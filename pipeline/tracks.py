"""Stage 1-2: sampled frames -> YOLO -> tracker -> tracklets.

One pass of ultralytics does detection AND association. Everything after this
file works on tracklets, not pixels, which is why the pipeline is cheap.
"""
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

    def dwell_seconds(self, radius_px):
        """Longest run kept inside `radius_px` of its own running centroid."""
        if self.n() < 2:
            return 0.0
        best = 0.0
        i = 0
        for j in range(1, self.n()):
            while i < j:
                xs = np.asarray(self.cx[i:j + 1], float)
                ys = np.asarray(self.cy[i:j + 1], float)
                r = np.max(np.hypot(xs - xs.mean(), ys - ys.mean()))
                if r <= radius_px:
                    break
                i += 1
            best = max(best, self.t[j] - self.t[i])
        return float(best)


def run_tracking(cfg, on_frame=None):
    """Yields nothing; returns {track_id: Tracklet} plus frame count and fps."""
    from ultralytics import YOLO

    d = cfg["detector"]
    v = cfg["video"]
    model = YOLO(d["weights"])

    import cv2
    cap = cv2.VideoCapture(v["path"])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, int(round(src_fps / float(v["target_fps"]))))
    eff_fps = src_fps / stride

    tracks, frames = {}, 0
    stream = model.track(
        source=v["path"], stream=True, persist=True,
        tracker=d["tracker"], conf=d["conf"], iou=d["iou"],
        imgsz=d["imgsz"], classes=d["classes"], max_det=d["max_det"],
        half=d["half"], device=d["device"], vid_stride=stride, verbose=False,
    )
    for i, r in enumerate(stream):
        frames += 1
        ts = i / eff_fps
        b = r.boxes
        if b is None or b.id is None:
            if on_frame:
                on_frame(ts, r, [])
            continue
        ids = b.id.int().cpu().tolist()
        xywh = b.xywh.cpu().numpy()
        cls = b.cls.int().cpu().tolist()
        cf = b.conf.cpu().tolist()
        for tid, (cx, cy, w, h), c, s in zip(ids, xywh, cls, cf):
            tr = tracks.setdefault(tid, Tracklet(track_id=tid, cls=c))
            tr.t.append(ts); tr.cx.append(float(cx)); tr.cy.append(float(cy))
            tr.w.append(float(w)); tr.h.append(float(h)); tr.conf.append(float(s))
        if on_frame:
            on_frame(ts, r, ids)
    return tracks, frames, eff_fps
