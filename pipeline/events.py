"""Stage 3: tracklets -> candidate events. THIS is where the accuracy is.

Arithmetic only. No model runs here. Every event carries the numbers that
justified it, so the VLM prompt and the demo UI can both quote them.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CandidateEvent:
    kind: str
    track_id: int
    cls: int
    t_start: float
    t_end: float
    geo_score: float
    facts: dict = field(default_factory=dict)


def _point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            if x < xint:
                inside = not inside
    return inside


def detect_events(tracks, cfg, class_speed_stats=None):
    e = cfg["events"]
    out = []
    stats = class_speed_stats or _speed_stats(tracks)

    for tr in tracks.values():
        if tr.n() < e["min_track_frames"]:
            continue  # refuse: too little data to say anything

        dwell = tr.dwell_seconds(e["loiter_radius_px"])
        if dwell >= e["loiter_seconds"]:
            out.append(CandidateEvent(
                "loiter", tr.track_id, tr.cls, tr.t[0], tr.t[-1],
                min(1.0, dwell / (2 * e["loiter_seconds"])),
                {"dwell_s": round(dwell, 1), "radius_px": e["loiter_radius_px"]}))

        sp = tr.speeds_px_s()
        mu, sd = stats.get(tr.cls, (None, None))
        if sp is not None and sd not in (None, 0.0) and np.isfinite(sp).any():
            z = float((np.nanmax(sp) - mu) / sd)
            if abs(z) >= e["speed_z_threshold"]:
                out.append(CandidateEvent(
                    "speed_anomaly", tr.track_id, tr.cls, tr.t[0], tr.t[-1],
                    min(1.0, abs(z) / (2 * e["speed_z_threshold"])),
                    {"z": round(z, 2), "peak_px_s": round(float(np.nanmax(sp)), 1)}))

        for zi, poly in enumerate(e["restricted_zones"] or []):
            hits = [i for i in range(tr.n()) if _point_in_poly(tr.cx[i], tr.cy[i], poly)]
            if hits:
                dur = tr.t[hits[-1]] - tr.t[hits[0]]
                out.append(CandidateEvent(
                    "zone_intrusion", tr.track_id, tr.cls, tr.t[hits[0]], tr.t[hits[-1]],
                    min(1.0, 0.6 + dur / 20.0),
                    {"zone": zi, "seconds_inside": round(dur, 1)}))

    return [c for c in out if c.geo_score >= e["candidate_floor"]]


def _speed_stats(tracks):
    """Per-class speed baseline from THIS video. Scene-relative 'normal'."""
    buckets = {}
    for tr in tracks.values():
        sp = tr.speeds_px_s()
        if sp is None:
            continue
        buckets.setdefault(tr.cls, []).extend([float(s) for s in sp if np.isfinite(s)])
    stats = {}
    for c, vals in buckets.items():
        if len(vals) < 20:
            stats[c] = (None, None)   # not enough to define normal -> no claim
        else:
            a = np.asarray(vals)
            stats[c] = (float(a.mean()), float(a.std() + 1e-9))
    return stats
