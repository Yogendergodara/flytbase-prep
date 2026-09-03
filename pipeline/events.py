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


def clip_window(t_start, t_end, t_peak, clip_seconds):
    """Tighten an event window around the moment that justified it.

    Events used to carry the whole tracklet lifetime, so a loiterer's window
    was minutes long and the VLM's 6 frames got spread across all of it
    instead of the anomaly. `video.clip_seconds` existed for this and was
    dead config. Returns a window of at most clip_seconds centred on t_peak,
    clamped inside the tracklet.
    """
    if not clip_seconds or (t_end - t_start) <= clip_seconds:
        return t_start, t_end
    half = clip_seconds / 2.0
    lo = max(t_start, min(t_peak - half, t_end - clip_seconds))
    return lo, lo + clip_seconds


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


def detect_events(tracks, cfg, class_speed_stats=None, density=None):
    e = cfg["events"]
    clip = cfg.get("video", {}).get("clip_seconds")
    # list, not a single id: VisDrone's class map has TWO human classes
    # (0=pedestrian, 1=people). Hardcoding one id meant that switching from
    # COCO to a VisDrone-trained checkpoint would silently misclassify the
    # other human class as an "abandoned object". Back-compat with the old
    # singular `person_class` key.
    person_classes = set(e.get("person_classes", [e.get("person_class", 0)]))
    out = []
    stats = class_speed_stats or _speed_stats(tracks)

    for tr in tracks.values():
        if tr.n() < e["min_track_frames"]:
            continue  # refuse: too little data to say anything

        dwell = tr.dwell_seconds(e["loiter_radius_px"])
        still = tr.dwell_seconds(e["abandoned_radius_px"])

        # abandoned wins over loiter when both fire: the tighter radius is the
        # more specific claim. Restricted to non-person classes - a standing
        # person is loitering, not an abandoned object. No owner association,
        # so this means "object stopped moving", not "someone left it".
        is_abandoned = (still >= e["abandoned_seconds"] and tr.cls not in person_classes)

        if is_abandoned:
            t0, t1 = clip_window(tr.t[0], tr.t[-1], tr.t[-1], clip)
            facts = {"stationary_s": round(still, 1), "radius_px": e["abandoned_radius_px"],
                     "track_window": [round(tr.t[0], 2), round(tr.t[-1], 2)]}
            owner = _nearby_person_at_start(tracks, tr, person_classes)
            if owner:
                facts["owner_hint"] = owner   # a hint, not a claim of ownership
            out.append(CandidateEvent(
                "abandoned", tr.track_id, tr.cls, t0, t1,
                min(1.0, still / (2 * e["abandoned_seconds"])), facts))
        elif dwell >= e["loiter_seconds"]:
            t0, t1 = clip_window(tr.t[0], tr.t[-1], tr.t[-1], clip)
            out.append(CandidateEvent(
                "loiter", tr.track_id, tr.cls, t0, t1,
                min(1.0, dwell / (2 * e["loiter_seconds"])),
                {"dwell_s": round(dwell, 1), "radius_px": e["loiter_radius_px"],
                 "track_window": [round(tr.t[0], 2), round(tr.t[-1], 2)]}))

        sp = tr.speeds_px_s()
        mu, sd = stats.get(tr.cls, (None, None))
        if sp is not None and sd not in (None, 0.0) and np.isfinite(sp).any():
            peak_i = int(np.nanargmax(sp))
            z = float((np.nanmax(sp) - mu) / sd)
            if abs(z) >= e["speed_z_threshold"]:
                # centre the window on the peak, not the whole track
                t_peak = tr.t[min(peak_i + 1, tr.n() - 1)]
                t0, t1 = clip_window(tr.t[0], tr.t[-1], t_peak, clip)
                geo = min(1.0, abs(z) / (2 * e["speed_z_threshold"]))
                facts = {"z": round(z, 2), "peak_px_s": round(float(np.nanmax(sp)), 1),
                         "peak_at_s": round(t_peak, 2)}

                # #G-H: pixel speed conflates object motion with camera pan.
                # Opt-in (null threshold = off, zero cost) and self-contained -
                # no signature change for any caller. Discounts rather than
                # drops: never fabricate a 0, but say plainly when the camera,
                # not the object, is what moved.
                hover_thr = e.get("hover_pan_px_threshold")
                if hover_thr and cfg.get("video", {}).get("path"):
                    try:
                        from pipeline.camera_motion import pan_between
                        dt = 1.0 / max(cfg["video"].get("target_fps", 3), 1e-6)
                        pan = pan_between(cfg["video"]["path"],
                                          max(0.0, t_peak - dt), t_peak)
                    except Exception as ex:
                        pan = None
                        facts["camera_pan_error"] = str(ex)
                    if pan is not None:
                        facts["camera_pan_px"] = round(pan, 1)
                        if pan > hover_thr:
                            facts["camera_panning"] = True
                            geo *= 0.3   # discounted, not dropped

                out.append(CandidateEvent(
                    "speed_anomaly", tr.track_id, tr.cls, t0, t1, geo, facts))

        for zi, poly in enumerate(e["restricted_zones"] or []):
            hits = [i for i in range(tr.n()) if _point_in_poly(tr.cx[i], tr.cy[i], poly)]
            if hits:
                dur = tr.t[hits[-1]] - tr.t[hits[0]]
                t0, t1 = clip_window(tr.t[hits[0]], tr.t[hits[-1]], tr.t[hits[0]], clip)
                out.append(CandidateEvent(
                    "zone_intrusion", tr.track_id, tr.cls, t0, t1,
                    min(1.0, 0.6 + dur / 20.0),
                    {"zone": zi, "seconds_inside": round(dur, 1)}))

    if density and density.get("std"):
        out.extend(_density_events(tracks, density, e["density_z_threshold"], clip))

    # carry re-id identity into the facts the VLM prompt and demo UI quote
    for ev in out:
        tr = tracks.get(ev.track_id)
        if tr is not None and tr.identity != -1:
            ev.facts["identity"] = tr.identity

    return [c for c in out if c.geo_score >= e["candidate_floor"]]


def _density_events(tracks, density, z_threshold, clip_seconds=None):
    """Scene-level: contiguous runs where objects-per-frame beats fit.py's
    baseline. track_id=-1 - this is a crowding event, not an object event."""
    per_frame = {}
    for tr in tracks.values():
        for t in tr.t:
            per_frame[round(t, 3)] = per_frame.get(round(t, 3), 0) + 1
    if not per_frame:
        return []

    mu, sd = density["mean"], density["std"]
    times = sorted(per_frame)
    out, run = [], []
    for t in times + [None]:
        z = (per_frame[t] - mu) / sd if t is not None else -1e9
        if z >= z_threshold:
            run.append(t)
            continue
        if run:
            peak = max(per_frame[r] for r in run)
            peak_t = [r for r in run if per_frame[r] == peak][0]
            peak_z = (peak - mu) / sd
            # same clipping as every other event kind - a 5-minute crowding
            # run used to keep the whole run as its window instead of being
            # centred on the actual peak
            t0, t1 = clip_window(run[0], run[-1], peak_t, clip_seconds)
            out.append(CandidateEvent(
                "density_anomaly", -1, -1, t0, t1,
                min(1.0, abs(peak_z) / (2 * z_threshold)),
                {"peak_count": peak, "z": round(peak_z, 2),
                 "run_window": [round(run[0], 2), round(run[-1], 2)]}))
            run = []
    return out


def _nearby_person_at_start(tracks, tr, person_classes, radius_px=80):
    """#19: a cheap owner-proximity signal for `abandoned`. Not real
    association - just "was a person near this object right before it went
    stationary". None when there is nothing to say; never a guessed identity."""
    if tr.n() < 2:
        return None
    x0, y0, t0 = tr.cx[0], tr.cy[0], tr.t[0]
    best = None
    for other in tracks.values():
        if other.track_id == tr.track_id or other.cls not in person_classes:
            continue
        for i, t in enumerate(other.t):
            if abs(t - t0) > 2.0:
                continue
            d = ((other.cx[i] - x0) ** 2 + (other.cy[i] - y0) ** 2) ** 0.5
            if d <= radius_px and (best is None or d < best[1]):
                best = (other.track_id, d)
    if best is None:
        return None
    return {"possible_owner_track_id": best[0], "distance_px": round(best[1], 1)}


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
