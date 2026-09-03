"""Fine-tune the AERIAL detector - the one thing worth training before the event.

Viewpoint and scale transfer across cities; 'normal' does not. So this trains
top-down small-object detection (VisDrone) plus synthetic night, and nothing else.

    python train/make_night.py --images .../images/train --labels .../labels/train
    python train/train_aerial.py --preset laptop      # ~3-4 h, 6-8 GB
    python train/train_aerial.py --preset kaggle      # ~4-5 h on P100/T4

Launch it in the BACKGROUND at hour one and build the pipeline while it runs.
"""
import argparse

PRESETS = {
    # name         model         imgsz epochs batch freeze   ~wall
    "overnight": ("yolo11s.pt",  1280, 60,    8,    0),    # 8-10 h  <- THE one
    "laptop":    ("yolo11n.pt",  960,  25,    8,    10),   # 3-4 h
    "kaggle":    ("yolo11s.pt",  1024, 40,    12,   0),    # 5-6 h, free P100/T4
    "fast":      ("yolo11n.pt",  768,  12,    16,   10),   # 1-1.5 h, out of time
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="overnight", choices=list(PRESETS))
    ap.add_argument("--data", default="VisDrone.yaml",
                    help="ultralytics fetches VisDrone automatically on first use")
    ap.add_argument("--name", default="aerial_night")
    a = ap.parse_args()

    from ultralytics import YOLO
    model_name, imgsz, epochs, batch, freeze = PRESETS[a.preset]
    print(f"[train] {model_name} @ {imgsz}px, {epochs} epochs, batch {batch}, "
          f"freeze={freeze} backbone layers")

    YOLO(model_name).train(
        data=a.data, imgsz=imgsz, epochs=epochs, batch=batch,
        freeze=freeze or None,
        # augmentation aimed at aerial + low light, not at generic photos
        hsv_h=0.010, hsv_s=0.5, hsv_v=0.65,   # hsv_v high: brightness robustness
        degrees=10.0, translate=0.10, scale=0.6, shear=0.0,
        perspective=0.0, flipud=0.2, fliplr=0.5,
        mosaic=1.0, close_mosaic=8, mixup=0.05,
        patience=8, cos_lr=True, cache=False, workers=4,
        project="weights", name=a.name, exist_ok=True, plots=True,
    )
    print("[train] best weights -> weights/%s/weights/best.pt" % a.name)
    print("[train] now A/B it against stock yolo11n on a NIGHT clip before trusting it")


if __name__ == "__main__":
    main()
