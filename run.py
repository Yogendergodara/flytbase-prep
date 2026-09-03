"""End-to-end: video -> sample -> YOLO -> track -> events -> VLM -> fuse -> alerts.

    python run.py --config config.yaml --video data/sample.mp4
    python run.py --video data/sample.mp4 --set vlm.backend=qwen detector.imgsz=1280
"""
import argparse, json, time, yaml
from pipeline.tracks import run_tracking
from pipeline.events import detect_events
from pipeline.openvocab import build_open_vocab
from pipeline.vlm_judge import build_judge, extract_frames
from pipeline.fuse import fuse_score, suppress


def apply_overrides(cfg, pairs):
    for p in pairs or []:
        k, v = p.split("=", 1)
        node = cfg
        *parents, leaf = k.split(".")
        for q in parents:
            node = node[q]
        node[leaf] = yaml.safe_load(v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video")
    ap.add_argument("--set", nargs="*", dest="overrides")
    ap.add_argument("--out", default="out/alerts.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.video:
        cfg["video"]["path"] = a.video
    apply_overrides(cfg, a.overrides)

    t0 = time.time()
    tracks, n_frames, eff_fps = run_tracking(cfg)
    t_track = time.time() - t0
    print(f"[1-2] {len(tracks)} tracklets over {n_frames} sampled frames "
          f"@{eff_fps:.1f}fps in {t_track:.1f}s "
          f"({n_frames / max(t_track, 1e-6):.1f} frames/s)")

    cands = detect_events(tracks, cfg)
    print(f"[3]   {len(cands)} candidate events "
          f"({100 * len(cands) / max(len(tracks), 1):.0f}% of tracklets)")

    open_vocab = build_open_vocab(cfg)
    ov_hits_total = 0
    for ev in cands:
        result = open_vocab.detect(cfg["video"]["path"], ev)
        ev.facts["open_vocab_hits"] = result["hits"]
        ov_hits_total += len(result["hits"])
    if cfg["open_vocab"]["backend"] != "none":
        print(f"[3b]  open-vocab: {ov_hits_total} text-prompted hits "
              f"across {len(cands)} candidate windows")

    judge = build_judge(cfg)
    scored, lat = [], []
    for ev in cands:
        s = time.time()
        frames = ([] if cfg["vlm"]["backend"] == "none"
                  else extract_frames(cfg["video"]["path"], ev.t_start, ev.t_end,
                                      cfg["vlm"]["frames_per_event"]))
        verdict = judge.judge(ev, frames)
        lat.append((time.time() - s) * 1000)
        score, mode = fuse_score(ev, verdict, cfg)
        scored.append((ev, verdict, score))
    if lat:
        print(f"[4]   vlm stage: {len(lat)} calls, mean {sum(lat)/len(lat):.0f}ms each")

    alerts = suppress(scored, cfg)
    print(f"[5]   {len(alerts)} alerts after hysteresis+cooldown "
          f"(suppressed {len(scored) - len(alerts)})")

    json.dump({"config": cfg, "eff_fps": eff_fps, "n_frames": n_frames,
               "alerts": alerts}, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"[6]   wrote {a.out}")
    for al in alerts[:10]:
        print(f"      {al['t_start']:7.1f}s  {al['kind']:<15} "
              f"id={al['track_id']:<4} score={al['score']:.2f}  {al.get('why') or ''}")


if __name__ == "__main__":
    main()
