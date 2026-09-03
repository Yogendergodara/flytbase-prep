"""Shared, exact video frame access.

Problem this fixes (#7): every call site used to `cap.set(POS_MSEC, t*1000)`
then `read()`. On most compressed codecs the decoder snaps a millisecond seek
to the nearest keyframe, which can land up to a GOP away (often 0.25-2s) from
the frame actually requested - so the VLM's frame strip, a re-id crop, or an
open-vocab check could silently come from the wrong moment while `facts`
still describe the frame you asked for.

`seek_exact` does a coarse seek to a safe point slightly before the target,
then decodes forward frame-by-frame with `grab()` (cheap - no full decode)
until the exact target frame count, then `retrieve()`s only that one. This is
correct as long as the stream is forward-seekable, which every file source
here is.
"""


def open_capture(path, retries=3, delay=2.0):
    """#28 + #25 (RTSP): retry opening a source before giving up.

    cv2.VideoCapture already accepts an rtsp:// URL exactly like a file path
    - no separate "RTSP integration" is needed for a single connect. What IS
    missing, and stays missing here: automatic RECONNECT if a live stream
    drops mid-run, and a health/liveness check. Solving that properly needs a
    supervising process outside this CLI; retrying the initial connect is the
    honest, scoped piece of it this repo takes on.
    """
    import time
    import cv2
    last_err = None
    for attempt in range(retries):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            return cap
        cap.release()
        last_err = f"could not open '{path}' (attempt {attempt + 1}/{retries})"
        print(f"[video_io] {last_err}")
        if attempt < retries - 1:
            time.sleep(delay)
    raise RuntimeError(f"[video_io] giving up on '{path}' after {retries} attempts")


def seek_exact(cap, t_seconds, fps=None, max_walk=None):
    """Return (ok, frame) for the frame at t_seconds, exactly.

    `max_walk` caps how far forward we'll grab() before giving up (guards
    against a corrupt timestamp sending this into a near-infinite walk); it
    defaults to 3 seconds of frames, generous for anything used in this repo.
    """
    import cv2
    fps = fps or (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    target = int(round(max(0.0, t_seconds) * fps))
    max_walk = max_walk or int(3 * fps)

    coarse_t = max(0.0, t_seconds - 1.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, coarse_t * 1000.0)
    cur = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    walked = 0
    while cur < target and walked < max_walk:
        if not cap.grab():
            return False, None
        cur += 1
        walked += 1
    return cap.retrieve()
