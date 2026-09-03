"""Stage 5: fuse geometric + VLM scores, then suppress false positives.

Hysteresis and per-track cooldown are the two things that turn a noisy score
into an alert stream an operator will actually keep switched on.
"""


def fuse_score(ev, verdict, cfg):
    f = cfg["fuse"]
    g = ev.geo_score
    v = verdict.get("score")
    if v is None:
        return g, "geometric_only"          # None is not 0.0 - no VLM opinion
    tot = f["w_geometric"] + f["w_vlm"]
    return (f["w_geometric"] * g + f["w_vlm"] * v) / tot, "fused"


def suppress(scored, cfg):
    """scored: list of (event, verdict, score) in time order."""
    f = cfg["fuse"]
    kept, last_fired = [], {}
    for ev, verdict, score in sorted(scored, key=lambda x: x[0].t_start):
        if score < f["raise_threshold"]:
            continue
        if (ev.t_end - ev.t_start) < f["min_event_seconds"]:
            continue
        prev = last_fired.get(ev.track_id)
        if prev is not None and (ev.t_start - prev) < f["cooldown_seconds"]:
            continue
        last_fired[ev.track_id] = ev.t_end
        kept.append({"kind": ev.kind, "track_id": ev.track_id, "cls": ev.cls,
                     "t_start": round(ev.t_start, 2), "t_end": round(ev.t_end, 2),
                     "score": round(score, 3), "geo_score": round(ev.geo_score, 3),
                     "facts": ev.facts, "label": verdict.get("label"),
                     "why": verdict.get("why")})
    return kept
