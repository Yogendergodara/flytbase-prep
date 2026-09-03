"""#26: pytest for the pure-python logic - no GPU, no video file, no
ultralytics/torch import needed. tracks/events/fuse/evaluate are all
import-safe without those (they import cv2/torch inside function bodies,
only when actually reading video or running a model), so this genuinely runs
anywhere, including CI.

NOT covered: pipeline.tracks.run_tracking, pipeline.reid, pipeline.vlm_judge,
pipeline.retrieve, pipeline.openvocab - all need a real video file and/or a
GPU model, which is exactly the "never executed" gap the rest of this repo's
docs are honest about. Testing the arithmetic that CAN be tested without
those is this suite's actual scope.
"""
import numpy as np
import pytest

from pipeline.tracks import Tracklet
from pipeline.events import (CandidateEvent, clip_window, _point_in_poly,
                             detect_events, _density_events)
from pipeline.fuse import fuse_score, suppress, HysteresisSuppressor
from pipeline.evaluate import frame_scores, report, sweep
from pipeline.vlm_judge import judge_event_safe


# ---------------------------------------------------------------- tracks --

def _tracklet(cls, ts, xs, ys):
    tr = Tracklet(track_id=1, cls=cls)
    for t, x, y in zip(ts, xs, ys):
        tr.t.append(t); tr.cx.append(x); tr.cy.append(y)
        tr.w.append(10.0); tr.h.append(10.0); tr.conf.append(0.9)
    return tr


def test_dwell_seconds_stationary_object():
    tr = _tracklet(0, [0, 1, 2, 3, 4], [100, 101, 100, 102, 101],
                   [100, 100, 101, 100, 99])
    assert tr.dwell_seconds(radius_px=10) == pytest.approx(4.0)


def test_dwell_seconds_moving_object_never_settles():
    tr = _tracklet(0, [0, 1, 2, 3], [0, 100, 200, 300], [0, 0, 0, 0])
    assert tr.dwell_seconds(radius_px=10) == 0.0


def test_dwell_seconds_matches_bruteforce_on_random_walk():
    rng = np.random.default_rng(0)
    n = 60
    ts = list(range(n))
    xs = list(np.cumsum(rng.normal(0, 3, n)))
    ys = list(np.cumsum(rng.normal(0, 3, n)))
    tr = _tracklet(0, ts, xs, ys)

    def bruteforce(radius_px):
        best = 0.0
        for i in range(n):
            for j in range(i, n):
                seg_x = np.asarray(xs[i:j + 1]); seg_y = np.asarray(ys[i:j + 1])
                if max(seg_x.max() - seg_x.min(), seg_y.max() - seg_y.min()) <= 2 * radius_px:
                    best = max(best, ts[j] - ts[i])
        return float(best)

    assert tr.dwell_seconds(25) == pytest.approx(bruteforce(25))


def test_speeds_px_s_returns_none_for_short_track():
    tr = _tracklet(0, [0], [0], [0])
    assert tr.speeds_px_s() is None


def test_trim_drops_old_points_but_keeps_first_seen():
    tr = _tracklet(0, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 0, 0, 0, 0])
    tr.first_seen = 0.0
    tr.trim(min_t=3)
    assert tr.t == [3, 4]
    assert tr.first_seen == 0.0   # survives trim - #2 fix


def test_trim_noop_when_nothing_is_old():
    tr = _tracklet(0, [5, 6, 7], [0, 1, 2], [0, 0, 0])
    tr.trim(min_t=0)
    assert tr.t == [5, 6, 7]


# ---------------------------------------------------------------- events --

def test_clip_window_centres_on_peak_when_track_is_long():
    lo, hi = clip_window(t_start=0, t_end=100, t_peak=50, clip_seconds=10)
    assert hi - lo == pytest.approx(10)
    assert lo <= 50 <= hi


def test_clip_window_passthrough_when_already_short():
    lo, hi = clip_window(t_start=0, t_end=5, t_peak=2, clip_seconds=10)
    assert (lo, hi) == (0, 5)


def test_clip_window_clamps_peak_near_the_edges():
    lo, hi = clip_window(t_start=0, t_end=100, t_peak=99, clip_seconds=10)
    assert lo >= 0 and hi <= 100
    assert hi - lo == pytest.approx(10)


def test_point_in_poly_square():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert _point_in_poly(5, 5, square) is True
    assert _point_in_poly(50, 50, square) is False


def _cfg(**overrides):
    cfg = {
        "video": {"clip_seconds": 4},
        "events": {"min_track_frames": 3, "loiter_seconds": 5,
                   "loiter_radius_px": 20, "abandoned_seconds": 8,
                   "abandoned_radius_px": 10, "speed_z_threshold": 2.5,
                   "density_z_threshold": 2.0, "person_classes": [0],
                   "restricted_zones": [], "candidate_floor": 0.0},
    }
    for k, v in overrides.items():
        cfg["events"][k] = v
    return cfg


def test_loitering_person_flagged_as_loiter_not_abandoned():
    tr = _tracklet(cls=0, ts=list(range(10)), xs=[50] * 10, ys=[50] * 10)
    out = detect_events({1: tr}, _cfg())
    kinds = {e.kind for e in out}
    assert "loiter" in kinds
    assert "abandoned" not in kinds


def test_stationary_non_person_flagged_as_abandoned():
    tr = _tracklet(cls=2, ts=list(range(10)), xs=[50] * 10, ys=[50] * 10)
    out = detect_events({1: tr}, _cfg())
    kinds = {e.kind for e in out}
    assert "abandoned" in kinds
    assert "loiter" not in kinds   # #4: abandoned wins, no double-fire


def test_abandoned_gets_owner_hint_when_person_was_nearby():
    bag = _tracklet(cls=2, ts=list(range(10)), xs=[50] * 10, ys=[50] * 10)
    person = _tracklet(cls=0, ts=[0, 1], xs=[52, 53], ys=[51, 50])
    person.track_id = 2
    out = detect_events({1: bag, 2: person}, _cfg())
    abandoned = [e for e in out if e.kind == "abandoned"][0]
    assert "owner_hint" in abandoned.facts
    assert abandoned.facts["owner_hint"]["possible_owner_track_id"] == 2


def test_speed_anomaly_discounted_when_camera_is_panning(monkeypatch):
    """#G-H: hover_pan_px_threshold is opt-in and self-contained - patch the
    optical-flow call so this needs no real video file."""
    import pipeline.camera_motion as camera_motion
    monkeypatch.setattr(camera_motion, "pan_between", lambda *a, **k: 40.0)

    tr = _tracklet(cls=2, ts=list(range(30)),
                   xs=[i * 50 for i in range(30)], ys=[0] * 30)
    stats = {2: (0.0, 1.0)}   # any real speed reads as a huge z-score
    cfg = _cfg(hover_pan_px_threshold=15)
    cfg["events"]["min_track_frames"] = 3
    cfg["video"]["path"] = "irrelevant.mp4"
    cfg["video"]["target_fps"] = 3

    out = detect_events({1: tr}, cfg, class_speed_stats=stats)
    speed_events = [e for e in out if e.kind == "speed_anomaly"]
    assert speed_events, "expected a speed_anomaly candidate to fire"
    ev = speed_events[0]
    assert ev.facts["camera_panning"] is True
    assert ev.facts["camera_pan_px"] == 40.0
    assert ev.geo_score < 0.5   # discounted, not dropped (still > 0)
    assert ev.geo_score > 0.0


def test_density_events_are_clipped_and_centred_on_the_peak():
    tracks = {}
    for i in range(30):
        tracks[i] = _tracklet(cls=0, ts=[10], xs=[i], ys=[i])
    density = {"mean": 1.0, "std": 1.0}
    out = _density_events(tracks, density, z_threshold=2.0, clip_seconds=4)
    assert len(out) == 1
    assert out[0].t_end - out[0].t_start <= 4


# ------------------------------------------------------------------ fuse --

def test_fuse_score_is_geometric_only_without_vlm_or_novelty():
    ev = CandidateEvent("loiter", 1, 0, 0, 5, 0.8, {})
    cfg = {"fuse": {"w_geometric": 0.45, "w_vlm": 0.55, "w_novelty": 0.0}}
    score, mode = fuse_score(ev, {"score": None}, cfg)
    assert score == 0.8
    assert mode == "geometric_only"


def test_fuse_score_blends_vlm_when_present():
    ev = CandidateEvent("loiter", 1, 0, 0, 5, 1.0, {})
    cfg = {"fuse": {"w_geometric": 0.5, "w_vlm": 0.5, "w_novelty": 0.0}}
    score, mode = fuse_score(ev, {"score": 0.0}, cfg)
    assert score == pytest.approx(0.5)
    assert mode == "fused"


def _fuse_cfg():
    return {"fuse": {"raise_threshold": 0.6, "clear_threshold": 0.4,
                     "min_event_seconds": 0.5, "cooldown_seconds": 10,
                     "ema_alpha": 1.0}}   # alpha=1 -> EMA == raw score, easy to reason about


def test_suppress_hysteresis_raises_once_then_blocks_until_cleared():
    cfg = _fuse_cfg()
    ev = lambda t0: CandidateEvent("loiter", 1, 0, t0, t0 + 1, 0.9, {})
    scored = [(ev(0), {}, 0.9), (ev(1), {}, 0.9)]   # still high - should NOT re-alert
    alerts = suppress(scored, cfg)
    assert len(alerts) == 1


def test_suppress_rearm_after_clearing_below_clear_threshold():
    cfg = _fuse_cfg()
    ev = lambda t0: CandidateEvent("loiter", 1, 0, t0, t0 + 1, 0.9, {})
    low = CandidateEvent("loiter", 1, 0, 5, 6, 0.1, {})
    scored = [(ev(0), {}, 0.9), (low, {}, 0.1), (ev(20), {}, 0.9)]
    alerts = suppress(scored, cfg)
    assert len(alerts) == 2   # raised, cleared, raised again


def test_suppress_respects_cooldown():
    cfg = _fuse_cfg()
    cfg["fuse"]["clear_threshold"] = 0.95   # never clears within this test
    ev = lambda t0: CandidateEvent("loiter", 1, 0, t0, t0 + 1, 0.9, {})
    scored = [(ev(0), {}, 0.9), (ev(2), {}, 0.9)]
    alerts = suppress(scored, cfg)
    assert len(alerts) == 1   # second one is within cooldown of the first


# -------------------------------------------------------------- evaluate --

def test_report_refuses_single_class_labels():
    scores = np.array([0.1, 0.9, 0.5])
    labels = np.array([0, 0, 0])
    out = report(scores, labels, threshold=0.5, fps=1.0)
    assert "error" in out


def test_sweep_refuses_single_class_labels():
    out = sweep(np.array([0.1, 0.9]), np.array([1, 1]))
    assert out == [{"error": "labels are single-class - sweep is meaningless"}]


# -------------------------------------------------------- exception boundary --

class _ExplodingJudge:
    """Simulates a VLM OOM / decode failure - #G-A."""
    def judge(self, ev, frames):
        raise RuntimeError("simulated CUDA OOM")


def test_judge_event_safe_degrades_one_event_instead_of_crashing():
    ev = CandidateEvent("loiter", 1, 0, 0, 5, 0.9, {})
    cfg = {"vlm": {"backend": "qwen", "frames_per_event": 6}}
    verdict, ms = judge_event_safe(_ExplodingJudge(), "does_not_matter.mp4", ev, cfg)
    assert verdict["score"] is None            # -> fuse_score's geometric_only path
    assert verdict["label"] == "stage_error"
    score, mode = fuse_score(ev, verdict, {"fuse": {"w_geometric": 1.0, "w_vlm": 0.0}})
    assert mode == "geometric_only"


def test_judge_event_safe_skips_frame_extraction_when_vlm_disabled():
    """backend=none skips extract_frames (no video file needed here), but
    .judge([]) is still called - the boundary still degrades cleanly even
    when the exception comes from the judge itself, not frame I/O."""
    ev = CandidateEvent("loiter", 1, 0, 0, 5, 0.9, {})
    cfg = {"vlm": {"backend": "none", "frames_per_event": 6}}
    verdict, ms = judge_event_safe(_ExplodingJudge(), "does_not_exist.mp4", ev, cfg)
    assert verdict["score"] is None
    assert verdict["label"] == "stage_error"


# -------------------------------------------------- shared suppression (#G-B) --

def test_hysteresis_suppressor_used_directly_matches_suppress_wrapper():
    """suppress() is a thin loop over HysteresisSuppressor - same object,
    batch and streaming can't drift apart by construction."""
    cfg = _fuse_cfg()
    events = [CandidateEvent("loiter", 1, 0, t, t + 1, 0.9, {}) for t in (0, 1)]
    scored = [(e, {}, 0.9) for e in events]

    via_wrapper = suppress(scored, cfg)

    sup = HysteresisSuppressor(cfg)
    via_direct = [a for e, v, s in scored if (a := sup.consider(e, v, s)) is not None]

    assert via_wrapper == via_direct


def test_frame_scores_paints_max_score_over_overlapping_alerts():
    alerts = [{"t_start": 0.0, "t_end": 2.0, "score": 0.3},
              {"t_start": 1.0, "t_end": 3.0, "score": 0.9}]
    s = frame_scores(alerts, n_frames=4, fps=1.0)
    assert s[1] == pytest.approx(0.9)   # overlap keeps the max, not the last write
