"""Stage F1: re-identification - link tracklets that are the same person
across an occlusion gap (the tracker assigned them different IDs because it
lost the object for a few frames).

Refuses on purpose: a tracklet with too few crops gets no embedding rather
than a noisy one, and two tracklets that overlap in time are never linked -
they cannot be the same physical object.
"""
import numpy as np
import cv2


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


def crop_at(video_path, t, cx, cy, w, h):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = int(cx + w / 2), int(cy + h / 2)
    x0, y0 = max(0, x0), max(0, y0)
    return frame[y0:y1, x0:x1]


def tracklet_embedding(video_path, tr, embedder, k=3, min_crops=2):
    """Average of k crops across the tracklet's lifetime. None if too thin
    to trust - a single frame is not a re-id signature."""
    n = tr.n()
    if n < min_crops:
        return None
    idxs = np.linspace(0, n - 1, min(k, n)).astype(int)
    embs = []
    for i in idxs:
        crop = crop_at(video_path, tr.t[i], tr.cx[i], tr.cy[i], tr.w[i], tr.h[i])
        e = embedder.embed(crop)
        if e is not None:
            embs.append(e)
    if len(embs) < min_crops:
        return None
    v = np.mean(embs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def link_identities(tracks, video_path, cfg):
    """Returns {track_id: identity_id}. Tracklets that never got a usable
    embedding keep their own track_id as their identity - no claim is made
    about them rather than a guessed one."""
    r = cfg["reid"]
    backend = r.get("backend", "none")
    embedder = NoopEmbedder() if backend == "none" else OSNetEmbedder(cfg)

    embs = {tid: tracklet_embedding(video_path, tr, embedder, r.get("crops_per_track", 3))
            for tid, tr in tracks.items()}

    identity = {tid: tid for tid in tracks}
    threshold = r.get("cosine_threshold", 0.7)
    ids = [tid for tid, e in embs.items() if e is not None]

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ta, tb = tracks[a], tracks[b]
            overlaps = not (ta.t[-1] < tb.t[0] or tb.t[-1] < ta.t[0])
            if overlaps:
                continue  # cannot be the same object if seen at the same time
            sim = float(np.dot(embs[a], embs[b]))
            if sim >= threshold:
                identity[b] = identity[a]

    return identity
