"""F3: export the detector to TensorRT and measure sustained FPS - the two
numbers for the real-time slide. Run on Kaggle/the target GPU, not locally.

    python scripts/export_engine.py --weights weights/aerial_night/weights/best.pt
"""
import argparse, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--video", default="data/sample.mp4")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--frames", type=int, default=200)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)

    print(f"[export] {a.weights} -> TensorRT engine, imgsz={a.imgsz}, half=True")
    engine_path = model.export(format="engine", imgsz=a.imgsz, half=True)
    print(f"[export] wrote {engine_path}")

    engine = YOLO(engine_path)
    import cv2
    cap = cv2.VideoCapture(a.video)

    # warmup - first calls include CUDA context / kernel compile time
    for _ in range(a.warmup):
        ok, frame = cap.read()
        if not ok:
            break
        engine.predict(frame, imgsz=a.imgsz, verbose=False)

    n, t0 = 0, time.time()
    while n < a.frames:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop a short clip to fill --frames
            continue
        engine.predict(frame, imgsz=a.imgsz, verbose=False)
        n += 1
    elapsed = time.time() - t0
    cap.release()

    fps = n / elapsed
    print(f"[bench] {n} frames in {elapsed:.2f}s -> {fps:.1f} FPS sustained "
          f"(detector only, engine, this machine)")
    print("[bench] this is the detector's number, NOT the full pipeline's - "
          "run.py's [7] line is the one that includes tracking+events+VLM")


if __name__ == "__main__":
    main()
