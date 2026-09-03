"""Stage 5: fuse geometric + VLM scores, then suppress false positives.

Hysteresis and per-track cooldown are the two things that turn a noisy score
into an alert stream an operator will actually keep switched on.
"""


def fuse_score(ev, verdict, cfg):
    f = cfg["fuse"]
    parts = [(f["w_geometric"], ev.geo_score)]

    v = verdict.get("score")
    if v is not None:
        parts.append((f["w_vlm"], v))

    nov = ev.facts.get("novelty")               # from retrieve.score_frame_novelty
    if nov is not None and f.get("w_novelty", 0.0) > 0:
        parts.append((f["w_novelty"], nov))

    if len(parts) == 1:
        return ev.geo_score, "geometric_only"   # None is not 0.0 - no other opinion

    tot = sum(w for w, _ in parts)
    return sum(w * s for w, s in parts) / tot, "fused"


class HysteresisSuppressor:
    """The ONE raise/clear/cooldown state machine. #G-B: batch (`suppress`
    below) and streaming (`pipeline/stream.py`) used to each hand-roll their
    own copy of this - consistent by inspection, not by construction, and one
    could silently drift from the other. Both now call this same class, so
    "batch and streaming agree on the same clip" is structural, not a promise.

    Real hysteresis, per track: an EMA of the score has to cross
    `raise_threshold` to open an alert state, and only falls back below
    `clear_threshold` to close it. While a track is already raised, further
    events do not re-alert - that plus the cooldown is what stops a single
    loiterer producing forty rows.
    """

    def __init__(self, cfg):
        f = cfg["fuse"]
        self.alpha = f.get("ema_alpha", 0.4)
        self.raise_t = f["raise_threshold"]
        self.clear_t = f["clear_threshold"]
        self.min_event = f["min_event_seconds"]
        self.cooldown = f["cooldown_seconds"]
        self.ema, self.raised, self.last_fired = {}, {}, {}

    def consider(self, ev, verdict, score):
        """One (event, verdict, score) in time order. Returns an alert dict
        or None. State (EMA/raised/cooldown) persists across calls - this is
        what lets streaming carry hysteresis across window boundaries."""
        tid = ev.track_id
        prev_ema = self.ema.get(tid)
        s = (score if prev_ema is None
             else self.alpha * score + (1 - self.alpha) * prev_ema)
        self.ema[tid] = s

        was_raised = self.raised.get(tid, False)
        if was_raised and s < self.clear_t:
            self.raised[tid] = False                  # cleared low - re-armed
            return None
        if s < self.raise_t:
            return None                                # never reached the raise line
        if was_raised:
            return None                                # already alerting on this track

        if (ev.t_end - ev.t_start) < self.min_event:
            return None
        prev = self.last_fired.get(tid)
        if prev is not None and (ev.t_start - prev) < self.cooldown:
            return None

        self.raised[tid] = True
        self.last_fired[tid] = ev.t_end
        return {"kind": ev.kind, "track_id": tid, "cls": ev.cls,
                "t_start": round(ev.t_start, 2), "t_end": round(ev.t_end, 2),
                "score": round(score, 3), "ema": round(s, 3),
                "geo_score": round(ev.geo_score, 3),
                "facts": ev.facts, "label": verdict.get("label"),
                "why": verdict.get("why")}


def suppress(scored, cfg):
    """Batch entry point: one HysteresisSuppressor over the whole list, in
    time order. scored: list of (event, verdict, score). Returns alert dicts."""
    sup = HysteresisSuppressor(cfg)
    kept = []
    for ev, verdict, score in sorted(scored, key=lambda x: x[0].t_start):
        alert = sup.consider(ev, verdict, score)
        if alert is not None:
            kept.append(alert)
    return kept
