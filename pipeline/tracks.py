"""Stage 1-2: sampled frames -> YOLO -> tracker -> tracklets.

One pass of ultralytics does detection AND association. Everything after this
file works on tracklets, not pixels, which is why the pipeline is cheap.
"""
from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Tracklet:
    track_id: int
    cls: int
    t: list = field(default_factory=list)        # timestamps (seconds)
    cx: list = field(default_factory=list)
    cy: list = field(default_factory=list)
    w: list = field(default_factory=list)
    h: list = field(default_factory=list)
    conf: list = field(default_factory=list)
    identity: int = -1                            # set by pipeline/reid.py; -1 = unlinked
    first_seen: float = None                      # earliest ts ever, survives trim()

    def n(self):
        return len(self.t)

    def duration(self):
        return 0.0 if self.n() < 2 else self.t[-1] - self.t[0]

    def speeds_px_s(self):
        """Instantaneous speed. Returns None (not 0.0) when undefined."""
        if self.n() < 2:
            return None
        p = np.stack([self.cx, self.cy], axis=1).astype(float)
        dt = np.diff(np.asarray(self.t, dtype=float))
        dt[dt <= 0] = np.nan
        return np.linalg.norm(np.diff(p, axis=0), axis=1) / dt

    def trim(self, min_t):
        """Drop history older than min_t. #2: streaming retirement only
        dropped IDLE tracks, so a continuously-visible person/vehicle over
        hours of footage kept every point forever. `first_seen` is captured
        once (in _accumulate) and survives trimming, so "how long has this
        existed" stays answerable even after old points are gone; dwell/speed
        math over the RETAINED window stays correct as long as the retention
        length exceeds the longest event window that reads it.
        """
        if not self.t or self.t[0] >= min_t:
            return
        i = 0
        while i < len(self.t) and self.t[i] < min_t:
            i += 1
        if i == 0:
            return
        self.t = self.t[i:]; self.cx = self.cx[i:]; self.cy = self.cy[i:]
        self.w = self.w[i:]; self.h = self.h[i:]; self.conf = self.conf[i:]

    def dwell_seconds(self, radius_px):
        """Longest run that stayed inside a 2*radius_px box.

        O(n) via monotonic deques holding the window's running max/min of x
        and y. The old version recomputed a centroid distance over the whole
        window for every endpoint - O(n^2), which on hours of footage at 3fps
        (10k+ points on a persistent track) was minutes of numpy per tracklet.

        The metric is a box of side 2r, not a circle of radius r about the
        centroid. Slightly more permissive at the corners; monotone in window
        size, which is what makes the two-pointer valid.
        """
        n = self.n()
        if n < 2:
            return 0.0
        side = 2.0 * radius_px
        maxx, minx, maxy, miny = deque(), deque(), deque(), deque()
        best, i = 0.0, 0
        for j in range(n):
            x, y = self.cx[j], self.cy[j]
            while maxx and self.cx[maxx[-1]] <= x: maxx.pop()
            maxx.append(j)
            while minx and self.cx[minx[-1]] >= x: minx.pop()
            minx.append(j)
            while maxy and self.cy[maxy[-1]] <= y: maxy.pop()
            maxy.append(j)
            while miny and self.cy[miny[-1]] >= y: miny.pop()
            miny.append(j)

            # shrink from the left until the window fits in the box
            while (self.cx[maxx[0]] - self.cx[minx[0]] > side
                   or self.cy[maxy[0]] - self.cy[miny[0]] > side):
                i += 1
                if maxx[0] < i: maxx.popleft()
                if minx[0] < i: minx.popleft()
                if maxy[0] < i: maxy.popleft()
                if miny[0] < i: miny.popleft()

            if j > i:
                best = max(best, self.t[j] - self.t[i])
        return float(best)


def _accumulate(tracks, r, ts, on_frame):
    """One tracked frame's boxes -> tracklets. Shared by both read paths.

    on_frame(ts, result, ids, tracks) - `tracks` is the live dict, which is
    what lets streaming mode evaluate and retire mid-pass.
    """
    b = r.boxes
    if b is None or b.id is None:
        if on_frame:
            on_frame(ts, r, [], tracks)
        return
    ids = b.id.int().cpu().tolist()
    xywh = b.xywh.cpu().numpy()
    cls = b.cls.int().cpu().tolist()
    cf = b.conf.cpu().tolist()
    for tid, (cx, cy, w, h), c, s in zip(ids, xywh, cls, cf):
        tr = tracks.setdefault(tid, Tracklet(track_id=tid, cls=c))
        if tr.first_seen is None:
            tr.first_seen = ts
        tr.t.append(ts); tr.cx.append(float(cx)); tr.cy.append(float(cy))
        tr.w.append(float(w)); tr.h.append(float(h)); tr.conf.append(float(s))
    if on_frame:
        on_frame(ts, r, ids, tracks)


def _clahe_bgr(frame, clahe):
    """CLAHE on the L channel only - BGR->LAB->equalise L->back to BGR.
    Never touch B/G/R directly; that would shift colour, not contrast."""
    import cv2
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _resolve_tracker(d):
    """Merge detector.tracker_overrides onto the base tracker YAML and return
    a path to the merged file. Fixes #6: track_high_thresh, track_low_thresh,
    new_track_thresh and track_buffer previously appeared nowhere in this
    repo - tuning them meant editing Ultralytics' own packaged YAML by hand.
    Returns the base tracker name unchanged when no override is set, so the
    common case pays nothing extra.
    """
    overrides = {k: v for k, v in (d.get("tracker_overrides") or {}).items()
                 if v is not None}
    base = d["tracker"]
    if not overrides:
        return base

    import yaml
    from pathlib import Path

    try:
        import ultralytics
        base_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / base
        cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}   # base file not found at that path in this Ultralytics
                   # version - start from overrides alone rather than fail
    cfg.update(overrides)

    out_path = Path("out") / f"tracker_{Path(base).stem}_merged.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[tracker] {base} + {overrides} -> {out_path}")
    return str(out_path)


def _resolve_half(d):
    """half=True on CPU raises in several Ultralytics versions. Decide once."""
    if not d.get("half"):
        return False
    dev = d.get("device")
    if dev == "cpu":
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    return True


def run_tracking(cfg, on_frame=None):
    """Returns {track_id: Tracklet} plus frame count and effective fps.

    `on_frame(ts, result, ids, tracks)` fires per sampled frame - that hook is what
    run.py's streaming mode uses to emit alerts before the video ends.
    `video.max_seconds` caps how much footage is read (fit.py sets it from
    --fit-seconds; None means the whole file).
    """
    from ultralytics import YOLO
    import cv2
    from pipeline.video_io import open_capture
    from pipeline.retry import retry

    d = cfg["detector"]
    v = cfg["video"]
    n = cfg.get("night", {"enabled": False})
    model = retry(YOLO, d["weights"], label=f"load {d['weights']}")
    conf = n["conf"] if n.get("enabled") else d["conf"]
    half = _resolve_half(d)
    tracker = _resolve_tracker(d)

    cap = open_capture(v["path"])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / float(v["target_fps"]))))
    eff_fps = src_fps / stride

    max_seconds = v.get("max_seconds")
    max_frames = int(max_seconds * eff_fps) if max_seconds else None

    tracks, frames = {}, 0

    if n.get("enabled") and n.get("clahe"):
        # manual per-frame loop: CLAHE needs pixels before the detector sees
        # them, so we can't hand the path to model.track()'s own streaming.
        # model.track(frame, persist=True) is the documented custom-loop
        # pattern for keeping tracker state across independent calls.
        clahe = cv2.createCLAHE(clipLimit=n["clahe_clip"],
                                 tileGridSize=(n["clahe_grid"], n["clahe_grid"]))
        idx = 0
        while max_frames is None or frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue
            frame = _clahe_bgr(frame, clahe)
            r = model.track(frame, persist=True, tracker=tracker, conf=conf,
                             iou=d["iou"], imgsz=d["imgsz"], classes=d["classes"],
                             max_det=d["max_det"], half=half, device=d["device"],
                             verbose=False)[0]
            _accumulate(tracks, r, frames / eff_fps, on_frame)
            frames += 1
            idx += 1
        cap.release()
        return tracks, frames, eff_fps

    cap.release()
    stream = model.track(
        source=v["path"], stream=True, persist=True,
        tracker=tracker, conf=conf, iou=d["iou"],
        imgsz=d["imgsz"], classes=d["classes"], max_det=d["max_det"],
        half=half, device=d["device"], vid_stride=stride, verbose=False,
    )
    for i, r in enumerate(stream):
        frames += 1
        _accumulate(tracks, r, i / eff_fps, on_frame)
        if max_frames is not None and frames >= max_frames:
            break
    return tracks, frames, eff_fps
