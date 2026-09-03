"""End-to-end: video -> sample -> YOLO -> track -> events -> VLM -> fuse -> alerts.

    python run.py --config config.yaml --video data/sample.mp4
    python run.py --video data/sample.mp4 --set vlm.backend=qwen detector.imgsz=1280
"""
import argparse, json, os, time, yaml
from pathlib import Path
from pipeline.video_io import open_capture
from pipeline.tracks import run_tracking
from pipeline.events import detect_events
from pipeline.openvocab import build_open_vocab
from pipeline.vlm_judge import build_judge, extract_frames, judge_event_safe
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


# One command each on Saturday - no YAML edits under pressure. Applied
# BEFORE --set, so an explicit --set always wins over the preset.
PRESETS = {
    "day":      ["detector.conf=0.25", "night.enabled=false", "vlm.backend=qwen"],
    "night":    ["detector.conf=0.20", "night.enabled=true", "night.conf=0.15",
                 "night.clahe=true", "vlm.backend=qwen"],
    "fast":     ["video.target_fps=2", "detector.imgsz=960", "vlm.backend=none",
                 "open_vocab.backend=none"],
    "accurate": ["video.target_fps=5", "detector.imgsz=1280", "vlm.backend=qwen",
                 "open_vocab.backend=yoloworld", "vlm.frames_per_event=8"],
    # alerts during the pass instead of after it - the answer to "in real time"
    "live":     ["stream.enabled=true", "detector.conf=0.25", "vlm.backend=qwen"],
    # switch to the VisDrone-trained checkpoint AND its class semantics
    # together (#3) - combine with --set for anything else, e.g.:
    #   --preset visdrone --set night.enabled=true
    "visdrone": ["detector.weights=weights/aerial_night/weights/best.pt",
                 "detector.classes=[0,1,2,3,4,5,6,7,8,9]",
                 "events.person_classes=[0,1]"],
}


def _write_out(path, cfg, eff_fps, n_frames, alerts, wall, lat, t_track=None,
               candidates=None):
    """`alerts` is post-suppression (what the demo shows). `candidates` is
    EVERY judged event before suppression (#1 fix) - eval_run.py needs the
    raw timeline to sweep thresholds; scoring only the alerts that already
    passed `raise_threshold` made every threshold below it untestable by
    construction."""
    from pipeline.provenance import get_provenance, append_audit_log
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": cfg, "eff_fps": eff_fps, "n_frames": n_frames,
               "alerts": alerts, "candidates": candidates or [],
               "wall_seconds": round(wall, 2),
               "vlm_latency_ms": [round(x, 1) for x in lat],
               "provenance": get_provenance(cfg)}
    if t_track is not None:
        payload["track_seconds"] = round(t_track, 2)
    json.dump(payload, open(path, "w", encoding="utf-8"), indent=2)
    append_audit_log(alerts)


def _run_streaming(cfg, a, t_all0, speed_stats, density, novelty_fn):
    """Alerts emitted DURING the pass. This is the answer to "in real time" -
    the batch path cannot alert until the file ends."""
    from pipeline.stream import StreamingPipeline

    from pipeline.alert_sink import make_sink
    sink = make_sink(cfg)

    judge = build_judge(cfg)
    cap = open_capture(cfg["video"]["path"])
    first_alert_at = [None]

    def on_alert(al):
        if first_alert_at[0] is None:
            first_alert_at[0] = time.time() - t_all0
        print(f"  ALERT {al['t_start']:7.1f}s  {al['kind']:<15} "
              f"id={al['track_id']:<4} score={al['score']:.2f}  "
              f"{al.get('why') or ''}")
        if sink:
            sink(al)

    pipe = StreamingPipeline(cfg, judge, speed_stats, density,
                             on_alert=on_alert, score_novelty=novelty_fn)

    print(f"[1-5] streaming: window={pipe.window}s, "
          f"retire_after={pipe.retire_after}s - alerts appear as they are found")
    last = {"ts": 0.0, "tracks": None}

    def on_frame(ts, result, ids, tracks):
        last["ts"], last["tracks"] = ts, tracks
        pipe.on_frame(ts, result, ids, tracks, cap=cap)

    try:
        tracks, n_frames, eff_fps = run_tracking(cfg, on_frame=on_frame)
        alerts = pipe.finalize(last["tracks"] or tracks, last["ts"], cap=cap)
    finally:
        cap.release()

    wall = time.time() - t_all0
    print(f"[5]   {len(alerts)} alerts, {len(pipe.judged)} events judged, "
          f"{len(tracks)} tracklets live at end (retired the rest)")
    if first_alert_at[0] is not None:
        print(f"[5b]  first alert at {first_alert_at[0]:.1f}s wall - "
              f"batch mode could not have alerted before {wall:.1f}s")
    _write_out(a.out, cfg, eff_fps, n_frames, alerts, wall, pipe.vlm_latency_ms,
              candidates=pipe.scored)
    print(f"[6]   wrote {a.out}")
    print(f"[7]   end-to-end: {wall:.1f}s wall, "
          f"{n_frames / max(wall, 1e-6):.1f} FPS sustained, this machine")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video")
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--set", nargs="*", dest="overrides")
    ap.add_argument("--out", default="out/alerts.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.video:
        cfg["video"]["path"] = a.video
    if a.preset:
        apply_overrides(cfg, PRESETS[a.preset])
        print(f"[0]   preset={a.preset}")
    apply_overrides(cfg, a.overrides)

    from pipeline.validate import validate
    validate(cfg, out_path=a.out)   # #27 - fail before spending minutes on a bad config

    t_all0 = time.time()

    # the fit has to load BEFORE tracking - streaming mode needs the baselines
    # available from the first window
    fit_path = "out/scene_fit.json"
    speed_stats, density = None, None
    if os.path.exists(fit_path):
        fit = json.load(open(fit_path, encoding="utf-8"))
        speed_stats = {int(c): ((v["mean"], v["std"]) if v else (None, None))
                       for c, v in fit["speed_by_class"].items()}
        density = fit["density"]
        print(f"[0]   using scene fit from {fit['video']} "
              f"({fit['sampled_frames']} frames, {fit['wall_seconds']}s)")
    else:
        print("[0]   no out/scene_fit.json - run fit.py first for scene-fitted "
              "baselines; falling back to per-video stats")

    use_novelty = (cfg["fuse"].get("w_novelty", 0.0) > 0
                   and os.path.exists("out/normal_bank.npy"))
    novelty_fn = None
    if use_novelty:
        from pipeline.retrieve import score_frame_novelty
        novelty_fn = score_frame_novelty

    if cfg.get("stream", {}).get("enabled"):
        return _run_streaming(cfg, a, t_all0, speed_stats, density, novelty_fn)

    t0 = time.time()
    tracks, n_frames, eff_fps = run_tracking(cfg)
    t_track = time.time() - t0
    print(f"[1-2] {len(tracks)} tracklets over {n_frames} sampled frames "
          f"@{eff_fps:.1f}fps in {t_track:.1f}s "
          f"({n_frames / max(t_track, 1e-6):.1f} frames/s)")

    if cfg["reid"]["backend"] != "none":
        from pipeline.reid import link_identities
        identity, linked = link_identities(tracks, cfg["video"]["path"], cfg)
        for tid, tr in tracks.items():
            # -1 stays "no claim made" so facts can distinguish a real link
            # from a tracklet that simply kept its own id
            tr.identity = identity[tid] if tid in linked else -1
        n_identities = len(set(identity.values()))
        print(f"[2b]  re-id: {len(tracks)} tracklets -> {n_identities} identities "
              f"({len(linked)} tracklets linked across gaps)")

    cands = detect_events(tracks, cfg, speed_stats, density)
    print(f"[3]   {len(cands)} candidate events "
          f"({100 * len(cands) / max(len(tracks), 1):.0f}% of tracklets)")

    # open-vocab only runs when it is actually enabled - it was looping over
    # every candidate to call a noop before
    if cfg["open_vocab"]["backend"] != "none":
        open_vocab = build_open_vocab(cfg)
        ov_hits_total = 0
        for ev in cands:
            result = open_vocab.detect(cfg["video"]["path"], ev)
            ev.facts["open_vocab_hits"] = result["hits"]
            ov_hits_total += len(result["hits"])
        print(f"[3b]  open-vocab: {ov_hits_total} text-prompted hits "
              f"across {len(cands)} candidate windows")

    judge = build_judge(cfg)
    scored, lat = [], []
    # one capture for every frame read in this loop, instead of two or three
    # VideoCapture open/close cycles per event
    cap = open_capture(cfg["video"]["path"])
    try:
        for ev in cands:
            # #G-A: this used to have no exception boundary - a VLM OOM or a
            # corrupt frame anywhere in this loop aborted the whole run
            verdict, ms = judge_event_safe(judge, cfg["video"]["path"], ev, cfg,
                                           cap=cap, novelty_fn=novelty_fn)
            lat.append(ms)
            score, mode = fuse_score(ev, verdict, cfg)
            scored.append((ev, verdict, score))
    finally:
        cap.release()
    if lat:
        print(f"[4]   vlm stage: {len(lat)} calls, mean {sum(lat)/len(lat):.0f}ms each")

    alerts = suppress(scored, cfg)
    print(f"[5]   {len(alerts)} alerts after hysteresis+cooldown "
          f"(suppressed {len(scored) - len(alerts)})")

    from pipeline.alert_sink import make_sink
    sink = make_sink(cfg)
    if sink:
        for al in alerts:
            sink(al)

    # every judged event, pre-suppression - see _write_out docstring (#1)
    candidates = [{"kind": ev.kind, "track_id": ev.track_id, "cls": ev.cls,
                   "t_start": round(ev.t_start, 2), "t_end": round(ev.t_end, 2),
                   "score": round(score, 3), "facts": ev.facts,
                   "why": verdict.get("why")}
                  for ev, verdict, score in scored]

    t_total = time.time() - t_all0
    # vlm_latency_ms is persisted so F5 can report p50/p95 - it used to be
    # measured, printed, and thrown away
    _write_out(a.out, cfg, eff_fps, n_frames, alerts, t_total, lat, t_track,
              candidates=candidates)
    print(f"[6]   wrote {a.out}")
    print(f"[7]   end-to-end: {t_total:.1f}s wall, "
          f"{n_frames / max(t_total, 1e-6):.1f} FPS sustained, this machine")
    for al in alerts[:10]:
        print(f"      {al['t_start']:7.1f}s  {al['kind']:<15} "
              f"id={al['track_id']:<4} score={al['score']:.2f}  {al.get('why') or ''}")


if __name__ == "__main__":
    main()
