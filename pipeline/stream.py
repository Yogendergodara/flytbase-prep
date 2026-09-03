"""Streaming mode: emit alerts DURING the pass, not after it.

The batch path collects every tracklet for the whole file and only then runs
the event/VLM/fuse stages. On the brief's own footage - "long-form CCTV and
drone video, hours not clips" - that means zero alerts until the video ends
and a `tracks` dict that grows without bound. The brief asks whether a small
VLM can do this "in real time", so latency-to-alert is the question, not just
total cost.

This runs the same cascade on a sliding window:

    every window_seconds of sampled video
      -> detect_events over the live tracklets
      -> judge only candidates not already judged
      -> fuse + suppress with state carried ACROSS windows
      -> emit new alerts immediately

Tracklets idle longer than retire_after are dropped. That alone does NOT
bound memory (#2 fix): a person or vehicle visible continuously for hours
never goes idle, so every window ALSO trims each live tracklet's history to
`max_track_seconds` - the actual bound for the persistent case. Suppression
state (EMA, raised, cooldown) is kept for retired tracks so a re-appearing id
cannot re-alert instantly.

`judged` (#12 fix) stops the VLM being re-paid for a static event, but does
not freeze it forever: once an event's window has grown by
`rejudge_margin_seconds` (i.e. real new evidence accumulated - a loiterer who
kept loitering), it is sent back to the VLM.

#G-B: suppression is `fuse.HysteresisSuppressor`, the SAME class the batch
path uses - this used to be a hand-rolled second copy of the state machine,
consistent with batch by inspection only. #G-A: judging goes through
`vlm_judge.judge_event_safe`, so a VLM/decode failure degrades one event to
geometric_only instead of killing the stream. #G-C (partial): open-vocab now
runs here too, on the same candidates batch mode checks - re-ID still does
not (see class docstring below for why).
"""
from pipeline.events import detect_events
from pipeline.fuse import fuse_score, HysteresisSuppressor
from pipeline.vlm_judge import judge_event_safe


class StreamingPipeline:
    """Re-ID is still batch-only (#G-C, partial close): it needs a pass over
    each tracklet's own crops with its own VideoCapture, which is a real cost
    per window, not per new event like judging is. Wiring it in means
    deciding a re-id cadence (every window? every N?) and that decision needs
    measuring against a real stream before it is worth the complexity -
    unlike open-vocab, which reuses the exact same per-candidate cost the
    batch path already pays.
    """

    def __init__(self, cfg, judge, speed_stats=None, density=None,
                 on_alert=None, score_novelty=None):
        self.cfg = cfg
        self.judge = judge
        self.speed_stats = speed_stats
        self.density = density
        self.on_alert = on_alert or (lambda a: None)
        self.score_novelty = score_novelty

        s = cfg.get("stream", {})
        self.window = s.get("window_seconds", 30)
        self.retire_after = s.get("retire_after", 60)
        self.max_track_seconds = s.get("max_track_seconds", 180)
        self.rejudge_margin = s.get("rejudge_margin_seconds", 15)

        self.open_vocab = None
        if cfg["open_vocab"]["backend"] != "none":
            from pipeline.openvocab import build_open_vocab
            self.open_vocab = build_open_vocab(cfg)

        self.suppressor = HysteresisSuppressor(cfg)
        self.judged = {}             # (track_id, kind) -> t_end last judged
        self.alerts = []
        self.scored = []             # EVERY judged candidate, pre-suppression
                                      # (#1 - eval needs the raw timeline, not
                                      # just what survived suppression)
        self.next_eval = self.window
        self.vlm_latency_ms = []

    # ---- the two hooks run.py drives -------------------------------------

    def on_frame(self, ts, result, ids, tracks, cap=None):
        """Called per sampled frame. Evaluates once per window_seconds."""
        if ts < self.next_eval:
            return
        self.next_eval = ts + self.window
        self._evaluate(tracks, ts, cap)
        self._retire(tracks, ts)
        self._trim_all(tracks, ts)

    def finalize(self, tracks, ts, cap=None):
        """Last partial window - otherwise the tail of the video never gets
        judged."""
        self._evaluate(tracks, ts, cap)
        return self.alerts

    # ---- internals -------------------------------------------------------

    def _evaluate(self, tracks, now, cap):
        live = {tid: tr for tid, tr in tracks.items()
                if tr.n() and (now - tr.t[-1]) <= self.window}
        if not live:
            return

        cands = detect_events(live, self.cfg, self.speed_stats, self.density)
        path = self.cfg["video"]["path"]

        for ev in cands:
            key = (ev.track_id, ev.kind)
            last_end = self.judged.get(key)
            if last_end is not None and (ev.t_end - last_end) < self.rejudge_margin:
                continue            # no meaningful new evidence since last judged
            self.judged[key] = ev.t_end

            # #G-C (partial): same open-vocab check the batch path runs,
            # populated into facts before the VLM sees them - candidate
            # windows only, same as batch
            if self.open_vocab is not None:
                result = self.open_vocab.detect(path, ev)
                ev.facts["open_vocab_hits"] = result["hits"]

            # #G-A: exception boundary shared with the batch path - a VLM
            # OOM or corrupt frame degrades this one event, not the stream
            verdict, ms = judge_event_safe(self.judge, path, ev, self.cfg,
                                           cap=cap, novelty_fn=self.score_novelty)
            self.vlm_latency_ms.append(ms)
            score, _ = fuse_score(ev, verdict, self.cfg)
            self.scored.append({"kind": ev.kind, "track_id": ev.track_id,
                               "cls": ev.cls, "t_start": round(ev.t_start, 2),
                               "t_end": round(ev.t_end, 2), "score": round(score, 3),
                               "facts": ev.facts, "why": verdict.get("why")})

            # #G-B: the SAME suppressor class the batch path uses
            alert = self.suppressor.consider(ev, verdict, score)
            if alert is not None:
                self.alerts.append(alert)
                self.on_alert(alert)

    def _retire(self, tracks, now):
        """Drop IDLE tracklets. Suppression state is intentionally NOT
        cleared. This alone does not bound memory for a track that is always
        live - see _trim_all."""
        dead = [tid for tid, tr in tracks.items()
                if tr.n() and (now - tr.t[-1]) > self.retire_after]
        for tid in dead:
            del tracks[tid]
        return len(dead)

    def _trim_all(self, tracks, now):
        """#2: bound memory for tracks that NEVER go idle. retire_after only
        catches tracks that stop appearing; a continuously-visible object
        kept its full history for the whole file. max_track_seconds must
        stay above the longest event window (loiter/abandoned) that reads
        tracklet history, or a long-running event loses the evidence it's
        made of mid-detection."""
        cutoff = now - self.max_track_seconds
        for tr in tracks.values():
            tr.trim(cutoff)
