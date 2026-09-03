"""Build a val YAML pointing at ONLY the synthetic night images, so F0 can
report day and night mAP separately instead of one averaged number.

make_night.py writes <stem>_night.<ext> beside each original. This collects
those into their own val list and emits a dataset YAML for `yolo val`.

    python train/make_night.py --images .../images/train --labels .../labels/train
    python scripts/make_night_yaml.py --images datasets/VisDrone/images/train \\
        --out VisDroneNight.yaml
    python scripts/ab_weights.py --tuned ... --night-data VisDroneNight.yaml
"""
import argparse
from pathlib import Path

# VisDrone's 10 classes, in Ultralytics' order
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                     help="the image dir make_night.py wrote into")
    ap.add_argument("--out", default="VisDroneNight.yaml")
    ap.add_argument("--list", default="night_val.txt")
    a = ap.parse_args()

    img_dir = Path(a.images).resolve()
    night = sorted(p for p in img_dir.iterdir()
                   if "_night" in p.stem
                   and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not night:
        print(f"no *_night.* images in {img_dir} - run train/make_night.py first")
        return

    list_path = Path(a.list).resolve()
    list_path.write_text("\n".join(str(p) for p in night), encoding="utf-8")

    # Ultralytics resolves val: against path:, and accepts a .txt of absolute
    # image paths. Labels are found by swapping /images/ -> /labels/.
    #
    # #13 fix: `train:` deliberately does NOT point at the same list as
    # `val:`. `yolo val` never reads `train:`, so the old version was not a
    # functional leak - but pointing both keys at the same images implied
    # this file could also be used to train on, which WOULD leak the val set
    # into training. Pointing train: at a path that does not exist makes this
    # YAML fail loudly if anyone ever runs `yolo train data=...` against it,
    # instead of silently doing the wrong thing.
    yaml_text = (f"# night-only VAL split, {len(night)} images. val-only:\n"
                 f"# `train:` is an intentionally broken placeholder - do not\n"
                 f"# `yolo train` against this file, it exists to A/B validate only.\n"
                 f"path: {img_dir.parent.parent}\n"
                 f"train: DO_NOT_TRAIN_ON_THIS_FILE_val_only.txt\n"
                 f"val: {list_path}\n"
                 f"names:\n"
                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    Path(a.out).write_text(yaml_text, encoding="utf-8")
    print(f"[night-yaml] {len(night)} night images -> {a.list}")
    print(f"[night-yaml] wrote {a.out}")
    print("[night-yaml] check `names:` matches your dataset if it isn't VisDrone")


if __name__ == "__main__":
    main()
