"""#30 (partial): blur faces in a video before it leaves this machine -
demoing real drone footage of real people deserves a redaction option, not
just an unencrypted copy sitting in data/. Uses OpenCV's bundled Haar cascade
- no extra model, no extra download, works offline.

Honest limits: a Haar face detector is a coarse, frontal-face-biased
detector; it will miss profile faces, occluded faces, and small/distant faces
common in overhead drone footage. This raises the bar, it does not guarantee
every face is blurred - say that if you present it, don't claim it as a
compliance control.

    python scripts/redact_faces.py --video theirs.mp4 --out theirs_redacted.mp4
"""
import argparse

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blur-kernel", type=int, default=35)
    a = ap.parse_args()

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    k = a.blur_kernel | 1   # GaussianBlur needs an odd kernel size
    n_frames, n_faces = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, fw, fh) in faces:
            roi = frame[y:y + fh, x:x + fw]
            frame[y:y + fh, x:x + fw] = cv2.GaussianBlur(roi, (k, k), 0)
        n_faces += len(faces)
        writer.write(frame)
        n_frames += 1

    cap.release()
    writer.release()
    print(f"[redact] {n_frames} frames, {n_faces} face detections blurred -> {a.out}")
    print("[redact] Haar cascade is frontal-face-biased and WILL miss faces "
          "(profile, occluded, small/distant) - review the output before "
          "treating it as complete redaction.")


if __name__ == "__main__":
    main()
