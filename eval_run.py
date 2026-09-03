"""F5: the full evaluation sweep - day/night AUC separately, threshold sweep,
FP/hour, and time-to-detection (an alert 4 minutes late is a report, not an
alarm - this matters more than raw accuracy for a live system).

The threshold sweep runs over `candidates` (every judged event, BEFORE
suppression) - not `alerts`. Scoring only post-suppression alerts made every
threshold below `fuse.raise_threshold` circular: an event that would have
passed a lower threshold was never scored in the first place, because it
never made it past the higher one to be judged... no - because `alerts.json`
only ever contained what already passed `raise_threshold`, so a sweep over it
could only ever rediscover the same threshold it was built at. `candidates`
is the raw pre-suppression timeline and is what a sweep actually needs.
The final operational report (FP/hour, latency) uses `alerts` - that IS the
threshold you're actually deploying at.

Needs frame-level ground truth. Convert whatever format the dataset ships
(CUHK Avenue ships .mat) into one JSON of anomalous time ranges per video:

    {"anomalous_ranges": [[12.0, 18.5], [40.0, 44.0]]}   # seconds, THIS video

Run day and night separately and report both, even if night is worse:

    python run.py --video avenue_01.avi --preset day   --out out/alerts_day.json
    python eval_run.py --alerts out/alerts_day.json --labels labels/avenue_01.json --tag day
    python run.py --video avenue_01_night.avi --preset night --out out/alerts_night.json
    python eval_run.py --alerts out/alerts_night.json --labels labels/avenue_01_night.json --tag night
"""
import argparse, json
import numpy as np
from pipeline.evaluate import frame_scores, report, sweep


def ranges_to_labels(ranges, n_frames, fps):
    labels = np.zeros(n_frames, dtype=int)
    for t0, t1 in ranges:
        i0, i1 = int(t0 * fps), int(t1 * fps) + 1
        labels[max(0, i0):min(n_frames, i1)] = 1
    return labels


def time_to_detection(alerts, ranges):
    """Seconds from an anomalous range's start to the first alert inside it.
    None where the range was never caught - a miss, not a zero."""
    out = []
    for t0, t1 in ranges:
        hits = [a["t_start"] for a in alerts if t0 <= a["t_start"] <= t1]
        out.append(min(hits) - t0 if hits else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--labels", required=True,
                     help='JSON: {"anomalous_ranges": [[t0,t1], ...]} in seconds')
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    data = json.load(open(a.alerts, encoding="utf-8"))
    alerts, n_frames, fps = data["alerts"], data["n_frames"], data["eff_fps"]
    candidates = data.get("candidates") or []
    ranges = json.load(open(a.labels, encoding="utf-8"))["anomalous_ranges"]
    labels = ranges_to_labels(ranges, n_frames, fps)
    threshold = data["config"]["fuse"]["raise_threshold"]

    if not candidates:
        print(f"[{a.tag}] WARNING: no 'candidates' in {a.alerts} (older run, "
              f"or nothing was judged) - sweep falls back to post-suppression "
              f"alerts and is circular below {threshold}. Re-run run.py to "
              f"get the raw timeline.")
    raw_scores = frame_scores(candidates or alerts, n_frames, fps)

    # operational report: at the threshold you actually deployed at, using
    # the alerts a human/demo would actually see
    alert_scores = frame_scores(alerts, n_frames, fps)
    metrics = report(alert_scores, labels, threshold, fps,
                     latency_ms=data.get("vlm_latency_ms") or None)
    print(f"[{a.tag}] operational (post-suppression, threshold={threshold}): {metrics}")

    # model-selection report: frame AUC over the RAW timeline is threshold-free
    # and is the number that should decide "is this detector any good", not
    # the operational one above
    if candidates:
        raw_metrics = report(raw_scores, labels, threshold, fps)
        print(f"[{a.tag}] raw candidate timeline (pre-suppression): "
              f"frame_auc={raw_metrics.get('frame_auc')}")

    fp_alerts = sum(1 for al in alerts
                    if not any(r0 <= al["t_start"] <= r1 for r0, r1 in ranges))
    hours = n_frames / fps / 3600.0 if fps else None
    print(f"[{a.tag}] fp_alerts_per_hour: "
          f"{round(fp_alerts / hours, 2) if hours else 'n/a'} "
          f"({fp_alerts} of {len(alerts)} alerts matched no labelled range) - "
          f"this is the operator-facing number; fp_frames_per_hour above is "
          f"a different, frame-level quantity")

    ttd = time_to_detection(alerts, ranges)
    caught = [t for t in ttd if t is not None]
    mean_ttd = f"{np.mean(caught):.1f}s" if caught else "n/a"
    print(f"[{a.tag}] time-to-detection: {len(caught)}/{len(ranges)} ranges "
          f"caught, mean {mean_ttd}")

    print(f"[{a.tag}] threshold sweep over the RAW candidate timeline "
          f"(precision/recall/f1):")
    for row in sweep(raw_scores, labels):
        print(f"   {row}")


if __name__ == "__main__":
    main()
