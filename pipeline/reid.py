"""Stage F1: re-identification - link tracklets that are the same person
across an occlusion gap (the tracker assigned them different IDs because it
lost the object for a few frames).

Refuses on purpose: a tracklet with too few crops gets no embedding rather
than a noisy one, and two tracklets that overlap in time are never linked -
they cannot be the same physical object.
"""
import numpy as np
import cv2
from pipeline.video_io import seek_exact, open_capture


class NoopEmbedder:
    def embed(self, crop_bgr):
        return None


class OSNetEmbedder:
    def __init__(self, cfg):
        import torch
        from torchreid.utils import FeatureExtractor
        r = cfg["reid"]
        self.extractor = FeatureExtractor(
            model_name=r.get("model_name", "osnet_x1_0"),
            model_path=r.get("weights", ""),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def embed(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        feat = self.extractor([rgb])[0]
        v = feat.cpu().numpy() if hasattr(feat, "cpu") else np.asarray(feat)
        n = np.linalg.norm(v) + 1e-9
        return v / n


def crop_at(cap, t, cx, cy, w, h):
    """Crop from an ALREADY-OPEN capture. Opening a VideoCapture per crop
    meant ~40 tracklets x 3 crops = 120 open/seek/close cycles per run."""
    ok, frame = seek_exact(cap, t)
    if not ok:
        return None
    x0, y0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x1, y1 = int(cx + w / 2), int(cy + h / 2)
    crop = frame[y0:y1, x0:x1]
    return crop if crop.size else None


def tracklet_embedding(cap, tr, embedder, k=3, min_crops=2):
    """Average of k crops across the tracklet's lifetime. None if too thin
    to trust - a single frame is not a re-id signature."""
    n = tr.n()
    if n < min_crops:
        return None
    idxs = np.linspace(0, n - 1, min(k, n)).astype(int)
    embs = []
    for i in idxs:
        crop = crop_at(cap, tr.t[i], tr.cx[i], tr.cy[i], tr.w[i], tr.h[i])
        e = embedder.embed(crop)
        if e is not None:
            embs.append(e)
    if len(embs) < min_crops:
        return None
    v = np.mean(embs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def link_identities(tracks, video_path, cfg):
    """Returns (identity, linked): {track_id: identity_id} and the set of
    track_ids that were actually merged with something.

    Union-find, not chained assignment: with plain `identity[b] = identity[a]`
    a link found later could overwrite an earlier group, so a<->b<->c came out
    inconsistent depending on iteration order. Union-find makes the grouping
    order-independent and transitive.

    Tracklets with no usable embedding keep their own track_id and stay out of
    `linked` - no claim is made about them rather than a guessed one.
    """
    r = cfg["reid"]
    backend = r.get("backend", "none")
    embedder = NoopEmbedder() if backend == "none" else OSNetEmbedder(cfg)

    if backend == "osnet":
        person_classes = set(cfg.get("events", {}).get(
            "person_classes", [cfg.get("events", {}).get("person_class", 0)]))
        non_person = {tr.cls for tr in tracks.values()} - person_classes
        if non_person:
            print(f"[reid] WARNING: OSNet is trained for person re-ID; "
                  f"classes {sorted(non_person)} in this scene are not people. "
                  f"Their embeddings/links are unvalidated - treat vehicle "
                  f"re-id as experimental until measured.")

    cap = open_capture(video_path)           # one capture for every crop
    try:
        embs = {tid: tracklet_embedding(cap, tr, embedder, r.get("crops_per_track", 3))
                for tid, tr in tracks.items()}
    finally:
        cap.release()

    parent = {tid: tid for tid in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]     # path halving
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)  # lowest track_id wins as the id
            return True
        return False

    threshold = r.get("cosine_threshold", 0.7)
    linked = set()

    # #11: bucket by class BEFORE generating pairs, instead of generating
    # every cross-class pair and skipping it inside the loop. Same asymptotic
    # worst case (one giant class still pays full O(n^2)), but the common
    # case - several classes, none of them huge - drops from one O(n^2) over
    # everything to a sum of much smaller O(k^2) per class.
    from collections import defaultdict
    buckets = defaultdict(list)
    for tid, e in embs.items():
        if e is not None:
            buckets[tracks[tid].cls].append(tid)

    for ids in buckets.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ta, tb = tracks[a], tracks[b]
                overlaps = not (ta.t[-1] < tb.t[0] or tb.t[-1] < ta.t[0])
                if overlaps:
                    continue  # cannot be the same object if seen at the same time
                if float(np.dot(embs[a], embs[b])) >= threshold:
                    union(a, b)
                    linked.add(a)
                    linked.add(b)

    identity = {tid: find(tid) for tid in tracks}
    return identity, linked
