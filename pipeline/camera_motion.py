"""#G-H: pixel speed conflates object motion with platform motion. A drone
panning at 200 px/s makes every static object look like it is moving at
200 px/s, and `loiter_radius_px` is equally altitude-dependent. This does not
solve platform-motion compensation in general (that needs telemetry or full
frame registration) - it gives `events.py` a cheap, honest signal to check
before trusting a speed event: how much did the BACKGROUND move, not the
tracked object.
"""
import cv2
import numpy as np


def estimate_pan_px(prev_gray, curr_gray, max_points=60):
    """Median optical-flow displacement of well-tracked points between two
    frames. Median, not mean - a handful of points landing on a genuinely
    moving foreground object should not swing a background-motion estimate.
    Returns 0.0 (never fabricates a large number) when too few points track.
    """
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_points,
                                  qualityLevel=0.01, minDistance=20)
    if pts is None or len(pts) < 8:
        return 0.0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
    if nxt is None:
        return 0.0
    mask = status.flatten() == 1
    good_prev, good_next = pts[mask], nxt[mask]
    if len(good_prev) < 8:
        return 0.0
    disp = np.linalg.norm((good_next - good_prev).reshape(-1, 2), axis=1)
    return float(np.median(disp))


def pan_between(video_path, t0, t1):
    """Camera-pan estimate between two timestamps in one video. Opens its own
    capture - this is called only for candidate speed_anomaly events (a
    handful per run), never per frame, so it is not a hot path."""
    cap = cv2.VideoCapture(video_path)
    try:
        from pipeline.video_io import seek_exact
        ok1, f1 = seek_exact(cap, t0)
        ok2, f2 = seek_exact(cap, t1)
        if not (ok1 and ok2):
            return None
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        return estimate_pan_px(g1, g2)
    finally:
        cap.release()
