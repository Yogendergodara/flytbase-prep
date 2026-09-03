"""Synthesise night frames from daylight training images.

The event footage includes night flights. Rather than hope the detector
generalises, darken a fraction of the training set and train on both.
Gamma down + gain down + Poisson-ish noise + mild blur ~ a real low-light sensor.

    python train/make_night.py --images datasets/VisDrone/images/train \
        --labels datasets/VisDrone/labels/train --fraction 0.4
"""
import argparse, random, shutil
from pathlib import Path
import cv2
import numpy as np


def to_night(img, rng):
    gamma = rng.uniform(1.8, 3.0)          # crush the midtones
    gain = rng.uniform(0.35, 0.6)          # lose exposure
    x = (img.astype(np.float32) / 255.0) ** gamma * gain
    x = x * rng.uniform(0.85, 1.0)         # slight colour desaturation toward blue
    x[:, :, 2] *= rng.uniform(0.85, 1.0)
    x = np.clip(x * 255.0, 0, 255)
    sigma = rng.uniform(4.0, 12.0)         # sensor noise, the part that kills recall
    x = x + rng.normal(0, sigma, x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    imgs = sorted([p for p in Path(a.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    pick = random.sample(imgs, int(len(imgs) * a.fraction))
    lab_dir = Path(a.labels)
    made = 0
    for p in pick:
        src_lab = lab_dir / (p.stem + ".txt")
        if not src_lab.exists():
            continue                        # no label, no synthetic twin
        img = cv2.imread(str(p))
        if img is None:
            continue
        out_img = p.with_name(p.stem + "_night" + p.suffix)
        cv2.imwrite(str(out_img), to_night(img, rng))
        shutil.copyfile(src_lab, lab_dir / (p.stem + "_night.txt"))
        made += 1
    print(f"[night] {made} synthetic night images written beside {len(imgs)} originals")
    print("[night] geometry is unchanged, so labels copy across verbatim")


if __name__ == "__main__":
    main()
