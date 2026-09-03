"""Synthetic smoke-test clip. Proves the pipeline runs end to end without
depending on a real video download. Moving boxes are not real objects -
YOLO may detect nothing on them, and that is fine; this only proves no
stage crashes.
"""
import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 480, 30, 20
path = "data/sample.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(path, fourcc, FPS, (W, H))

rng = np.random.default_rng(0)
n = FPS * SECONDS
# one "loiterer" that barely moves, one fast "vehicle" crossing the frame
for i in range(n):
    frame = np.full((H, W, 3), 30, dtype=np.uint8)
    t = i / FPS
    lx = int(320 + 15 * np.sin(t * 0.5))
    ly = int(240 + 15 * np.cos(t * 0.5))
    cv2.rectangle(frame, (lx - 20, ly - 40), (lx + 20, ly + 40), (60, 180, 60), -1)
    vx = int((t / SECONDS) * (W + 100) - 50)
    cv2.rectangle(frame, (vx - 30, 200), (vx + 30, 240), (60, 60, 200), -1)
    vw.write(frame)
vw.release()
print(f"wrote {path}: {n} frames @ {FPS}fps, {SECONDS}s")
