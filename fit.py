"""Fit the SCENE-SPECIFIC parts on footage you have never seen. Seconds, not epochs.

Run this first at the event, on their video. Nothing here trains; it measures.
Writes out/scene_fit.json (speed + density baselines) and out/normal_bank.npy
(embeddings of quiet frames, for cosine-distance anomaly scoring AND retrieval).

    python fit.py --video data/theirs.mp4 --fit-seconds 120
"""
import argparse, json, time, yaml
import numpy as np
from pipeline.tracks import run_tracking
from pipeline.events import _speed_stats


def fit_embeddings(video_path, n_frames, out_npy):
    """SigLIP over evenly spaced frames. One index, two jobs."""
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
    idxs = np.linspace(0, max(total - 1, 0), min(n_frames, max(total, 1))).astype(int)
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
    print(f"[fit] normal bank: {len(embs)} frames, {tag} -> {out_npy}")
    return {"encoder": tag, "n": len(embs), "frame_idx": kept}


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

    t0 = time.time()
    tracks, n_frames, eff_fps = run_tracking(cfg)

    # per-class speed baseline; (None, None) where there is too little to say
    stats = _speed_stats(tracks)
    speed = {str(c): ({"mean": m, "std": s} if m is not None else None)
             for c, (m, s) in stats.items()}
    refused = [c for c, v in speed.items() if v is None]

    # density baseline: objects visible per sampled frame
    per_frame = {}
    for tr in tracks.values():
        for t in tr.t:
            per_frame[round(t, 3)] = per_frame.get(round(t, 3), 0) + 1
    counts = np.asarray(list(per_frame.values()) or [0], float)
    density = {"mean": float(counts.mean()), "std": float(counts.std()),
               "p95": float(np.percentile(counts, 95))} if len(counts) >= 20 else None

    bank = fit_embeddings(a.video, a.bank_frames, "out/normal_bank.npy")

    out = {"video": a.video, "sampled_frames": n_frames, "eff_fps": eff_fps,
           "fit_seconds_of_video": n_frames / eff_fps,
           "speed_by_class": speed, "speed_refused_classes": refused,
           "density": density, "normal_bank": bank,
           "wall_seconds": round(time.time() - t0, 1)}
    json.dump(out, open("out/scene_fit.json", "w", encoding="utf-8"), indent=2)
    print(f"[fit] {len(tracks)} tracklets | density={density} | "
          f"{len(refused)} class(es) refused for lack of data")
    print(f"[fit] wrote out/scene_fit.json in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
