"""Fit the SCENE-SPECIFIC parts on footage you have never seen. Seconds, not epochs.

Run this first at the event, on their video. Nothing here trains; it measures.
Writes out/scene_fit.json (speed + density baselines) and out/normal_bank.npy
(embeddings of quiet frames, for cosine-distance anomaly scoring AND retrieval).

    python fit.py --video data/theirs.mp4 --fit-seconds 120
"""
import argparse, json, time, yaml
from pathlib import Path
import numpy as np
from pipeline.tracks import run_tracking
from pipeline.events import _speed_stats, detect_events


def src_fps_of(video_path):
    """The video's own fps. Tracklet timestamps are in SAMPLED time; the frame
    seeking in fit_embeddings is in SOURCE frames. Mixing them silently
    misplaces every retrieval timestamp."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps or 30.0


def fit_embeddings(video_path, n_frames, out_npy, max_seconds=None,
                    quiet_frames=None):
    """SigLIP over evenly spaced frames. One index, two jobs.

    `quiet_frames`: source frame numbers whose object count was at or below
    the scene median. The bank is meant to describe NORMAL, so sampling
    uniformly over footage that contains the anomalies puts the anomalies in
    the bank and quietly flattens every novelty score. When it is None (no
    tracking data) we fall back to uniform and say so.
    """
    import cv2
    try:
        import open_clip, torch
    except ImportError:
        print("[fit] open_clip not installed - skipping normal bank")
        return None
    try:
        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", pretrained="webli")
        tag = "ViT-B-16-SigLIP/webli"
    except Exception:
        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        tag = "ViT-B-32/openai"
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)   # retrieval needs SOURCE fps
    if max_seconds and src_fps:
        total = min(total, int(max_seconds * src_fps)) or total

    if quiet_frames:
        pool = np.asarray(sorted(q for q in quiet_frames if q < max(total, 1)))
        if len(pool) == 0:
            pool = np.asarray([0])
        pick = np.linspace(0, len(pool) - 1, min(n_frames, len(pool))).astype(int)
        idxs = pool[pick]
        source = f"{len(pool)} quiet frames"
    else:
        idxs = np.linspace(0, max(total - 1, 0), min(n_frames, max(total, 1))).astype(int)
        source = "uniform (no track data - bank may include anomalies)"
    from PIL import Image
    embs, kept = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        im = Image.fromarray(f[:, :, ::-1])
        with torch.no_grad():
            e = model.encode_image(pre(im).unsqueeze(0).to(dev))
        e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        embs.append(e); kept.append(int(i))
    cap.release()
    if not embs:
        return None
    np.save(out_npy, np.stack(embs))
    print(f"[fit] normal bank: {len(embs)} frames from {source}, {tag} -> {out_npy}")
    return {"encoder": tag, "n": len(embs), "frame_idx": kept, "src_fps": src_fps,
            "sampled_from": source}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", required=True)
    ap.add_argument("--fit-seconds", type=float, default=120.0,
                    help="how much footage to treat as 'mostly normal'")
    ap.add_argument("--bank-frames", type=int, default=200)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg["video"]["path"] = a.video
    # THE deployability number: only look at --fit-seconds of footage, so
    # "five minutes to adapt" holds on a two-hour file instead of quietly
    # processing all of it.
    cfg["video"]["max_seconds"] = a.fit_seconds
    from pipeline.validate import validate
    validate(cfg, out_path="out/scene_fit.json")
    Path("out").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tracks, n_frames, eff_fps = run_tracking(cfg)

    # per-class speed baseline; (None, None) where there is too little to say
    stats = _speed_stats(tracks)
    speed = {str(c): ({"mean": m, "std": s} if m is not None else None)
             for c, (m, s) in stats.items()}
    refused = [c for c, v in speed.items() if v is None]

    # density baseline: objects visible per sampled frame. Built over EVERY
    # sampled frame, zero-count ones included - a per_frame dict built only
    # from tracklet timestamps never contains a frame with zero objects, which
    # pulled mean/std up and made the live density threshold too lenient.
    hit_counts = {}
    for tr in tracks.values():
        for t in tr.t:
            hit_counts[round(t, 3)] = hit_counts.get(round(t, 3), 0) + 1
    all_ts = [round(i / eff_fps, 3) for i in range(n_frames)] if eff_fps else []
    per_frame = {t: hit_counts.get(t, 0) for t in all_ts} or hit_counts
    counts = np.asarray(list(per_frame.values()) or [0], float)
    density = {"mean": float(counts.mean()), "std": float(counts.std()),
               "p95": float(np.percentile(counts, 95))} if len(counts) >= 20 else None

    # quiet frames = at or below the median object count. Converted from
    # sampled-frame timestamps back to SOURCE frame numbers, which is the
    # space fit_embeddings seeks in.
    quiet = None
    if per_frame:
        median = float(np.median(counts))
        src_fps = src_fps_of(a.video)      # sampled seconds -> source frame number
        quiet = [int(round(t * src_fps))
                 for t, c in sorted(per_frame.items()) if c <= median]

    # sanity check on "this footage is mostly normal" (#9): a lenient event
    # pass on the SAME window that flags a large fraction of tracklets means
    # the fit window itself likely contains anomalies, not just normal scene
    # activity - the quiet-frame filter and baselines are unreliable then.
    # This can only warn, not fix itself: fit.py has no ground truth to check
    # against, so say so explicitly rather than silently trusting the window.
    lenient_cfg = {**cfg, "events": {**cfg["events"], "candidate_floor": 0.15}}
    preview_events = detect_events(tracks, lenient_cfg, stats, density)
    flagged_fraction = len(preview_events) / max(len(tracks), 1)
    window_looks_anomalous = flagged_fraction > 0.25
    if window_looks_anomalous:
        print(f"[fit] WARNING: {flagged_fraction:.0%} of tracklets in the fit "
              f"window look anomalous even at a lenient threshold. The "
              f"--fit-seconds window may not be 'mostly normal' - baselines "
              f"and the normal bank built from it are suspect. Pick a calmer "
              f"window or a longer --fit-seconds.")

    bank = fit_embeddings(a.video, a.bank_frames, "out/normal_bank.npy",
                          max_seconds=a.fit_seconds, quiet_frames=quiet)

    out = {"video": a.video, "sampled_frames": n_frames, "eff_fps": eff_fps,
           "fit_seconds_requested": a.fit_seconds,
           "fit_seconds_of_video": n_frames / eff_fps,
           "speed_by_class": speed, "speed_refused_classes": refused,
           "density": density, "normal_bank": bank,
           "window_looks_anomalous": window_looks_anomalous,
           "flagged_fraction_at_lenient_threshold": round(flagged_fraction, 3),
           "wall_seconds": round(time.time() - t0, 1)}
    json.dump(out, open("out/scene_fit.json", "w", encoding="utf-8"), indent=2)
    print(f"[fit] {len(tracks)} tracklets | density={density} | "
          f"{len(refused)} class(es) refused for lack of data")
    print(f"[fit] wrote out/scene_fit.json in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
