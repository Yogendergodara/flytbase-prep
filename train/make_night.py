"""Synthesise night frames from daylight training images.

The event footage includes night flights. Rather than hope the detector
generalises, darken a fraction of the training set and train on both.
Gamma down + gain down + additive Gaussian noise + mild blur ~ a real
low-light sensor (not true Poisson shot noise - that would need per-pixel
signal-dependent variance; this is the cheap, still-useful approximation).

    python train/make_night.py --images datasets/VisDrone/images/train \
        --labels datasets/VisDrone/labels/train --fraction 0.4

Idempotent: re-running (e.g. after an interrupted Kaggle session) will not
re-darken its own output or inflate the fraction - see the exclusion filter
below.
"""
import argparse, random, shutil, time
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
    # exclude our OWN prior output from the candidate pool - without this, a
    # second run (interrupted session, or re-run by mistake) could re-darken
    # an already-synthetic frame into "_night_night", and would count those
    # synthetic frames toward `len(imgs)`, silently changing what --fraction
    # means run to run
    imgs = sorted([p for p in Path(a.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                   and not p.stem.endswith("_night")])
    already = sum(1 for p in Path(a.images).iterdir() if p.stem.endswith("_night"))
    if already:
        print(f"[night] {already} existing *_night.* file(s) found - excluded "
              f"from the source pool, not re-darkened")

    pick = random.sample(imgs, min(int(len(imgs) * a.fraction), len(imgs)))
    print(f"[night] darkening {len(pick)} of {len(imgs)} images "
          f"(CPU-only - no GPU used in this step)")
    lab_dir = Path(a.labels)
    made = 0
    t0 = time.time()
    for idx, p in enumerate(pick, 1):
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
        # progress every 200 images or every 10s, whichever comes first -
        # otherwise this loop is silent for minutes and looks hung
        if idx % 200 == 0 or idx == len(pick):
            rate = idx / max(time.time() - t0, 1e-6)
            print(f"[night] {idx}/{len(pick)} ({rate:.0f} img/s)")
    print(f"[night] {made} synthetic night images written beside {len(imgs)} originals "
          f"in {time.time() - t0:.1f}s")
    print("[night] geometry is unchanged, so labels copy across verbatim")


if __name__ == "__main__":
    main()
