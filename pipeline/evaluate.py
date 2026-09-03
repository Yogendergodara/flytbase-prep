"""Stage 6: the numbers that go on the slide.

Frame-level ROC-AUC is the standard metric in the video-anomaly literature;
precision/recall/F1 are threshold-dependent, so always report the threshold.

`fp_frames_per_hour` is a FRAME-level count (how many labelled-normal frames
score above threshold) - it is what feeds AUC/F1 math but is NOT what an
operator asks about. `fp_alerts_per_hour` (computed in eval_run.py from the
alert list, not here) is the count an operator actually cares about: how many
times a human got pinged for nothing. Report both; they answer different
questions and the old single `fp_per_hour` name conflated them.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def frame_scores(alerts, n_frames, fps):
    s = np.zeros(n_frames)
    for a in alerts:
        i0 = int(a["t_start"] * fps); i1 = int(a["t_end"] * fps) + 1
        s[max(0, i0):min(n_frames, i1)] = np.maximum(
            s[max(0, i0):min(n_frames, i1)], a["score"])
    return s


def report(scores, labels, threshold, fps, latency_ms=None):
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"error": "labels are single-class - AUC undefined, refusing to report one"}
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary",
                                                  zero_division=0)
    hours = len(labels) / fps / 3600.0
    fp = int(((pred == 1) & (labels == 0)).sum())
    out = {"frame_auc": round(float(roc_auc_score(labels, scores)), 4),
           "threshold": threshold, "precision": round(float(p), 4),
           "recall": round(float(r), 4), "f1": round(float(f1), 4),
           "fp_frames": fp,
           "fp_frames_per_hour": round(fp / hours, 1) if hours else None}
    if latency_ms:
        out["p50_latency_ms"] = round(float(np.percentile(latency_ms, 50)), 1)
        out["p95_latency_ms"] = round(float(np.percentile(latency_ms, 95)), 1)
    return out


def sweep(scores, labels, lo=0.1, hi=0.95, n=18):
    """Threshold sweep. Refuses as a whole rather than returning rows that
    silently contain nothing but a threshold - report() errors on
    single-class labels, and the old dict comprehension swallowed that."""
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return [{"error": "labels are single-class - sweep is meaningless"}]
    return [{"threshold": round(t, 3),
             **{k: v for k, v in report(scores, labels, t, 1.0).items()
                if k in ("precision", "recall", "f1")}}
            for t in np.linspace(lo, hi, n)]
