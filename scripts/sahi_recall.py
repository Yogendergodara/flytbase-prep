"""Measure what SAHI tiling actually buys on aerial footage.

The brief's claim is "imgsz=1280 plus SAHI beats any version bump" - this
turns that into a number you can put on a slide instead of repeating it.

SAHI returns detections, not track IDs, so it is deliberately NOT inside
run_tracking: Ultralytics' tracker is coupled to model.track(), and feeding
sliced boxes into it would mean hand-rolling association. What SAHI is for
here is the recall argument and, if you want it, a higher-recall detector for
a single-frame demo.

    python scripts/sahi_recall.py --weights yolo11s.pt --video data/sample.mp4
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--video", default="data/sample.mp4")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        print("sahi not installed: pip install sahi")
        return

    plain = YOLO(a.weights)
    sliced = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=a.weights,
        confidence_threshold=a.conf, device="cuda:0")

    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(total - 1, 0), a.frames).astype(int)

    n_plain, n_sahi, small_plain, small_sahi = 0, 0, 0, 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue

        r = plain.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        if r.boxes is not None:
            wh = r.boxes.xywh.cpu().numpy()[:, 2:] if len(r.boxes) else np.empty((0, 2))
            n_plain += len(wh)
            small_plain += int((wh.max(axis=1) < 32).sum()) if len(wh) else 0

        # SAHI/ultralytics predict expects RGB; cv2.read() gives BGR - feeding
        # it BGR silently degrades the exact detections this script measures
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        s = get_sliced_prediction(
            frame_rgb, sliced, slice_height=a.slice, slice_width=a.slice,
            overlap_height_ratio=a.overlap, overlap_width_ratio=a.overlap,
            verbose=0)
        boxes = [o.bbox for o in s.object_prediction_list]
        n_sahi += len(boxes)
        small_sahi += sum(1 for b in boxes
                          if max(b.maxx - b.minx, b.maxy - b.miny) < 32)
    cap.release()

    print(f"[sahi] {len(idxs)} frames, slice={a.slice} overlap={a.overlap}")
    print(f"  plain @{a.imgsz}: {n_plain} detections ({small_plain} under 32px)")
    print(f"  sliced         : {n_sahi} detections ({small_sahi} under 32px)")
    if n_plain:
        print(f"  recall delta   : {100 * (n_sahi - n_plain) / n_plain:+.1f}% overall, "
              f"{small_sahi - small_plain:+d} small objects")
    print("  NOTE: more detections is not automatically better - eyeball a few "
          "frames for duplicate/spurious boxes before claiming this as recall.")


if __name__ == "__main__":
    main()
