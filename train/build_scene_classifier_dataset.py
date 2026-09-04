"""Build Pool 3 (DATASET_PLAN.md): a fire/smoke/flood/normal classification
dataset for the P20 scene-scan's auxiliary classifier.

FloodNet (100%, all-aerial) + a heavy FASDD_UAV sample (aerial fire/smoke) +
a capped D-Fire sample (ground-level, kept low - it exists only for
negative/hard-case diversity: fog, glare, night lights). Grouped by source
folder/scene, not by image, so near-duplicate frames from one clip can't
leak across train/val.

    python train/build_scene_classifier_dataset.py \\
        --floodnet datasets/FloodNet --fasdd datasets/FASDD_UAV --dfire datasets/D-Fire \\
        --out datasets/scene_hazard

Format handling, in order of confidence (no local copy of any of these to
test against - Kaggle is where these assumptions actually get checked):
  - FloodNet: the dataset's own Track-1 classification labels are used
    directly when present (a `Flooded`/`Non-Flooded` folder split, the
    known public layout for that track) - no pixel-value guessing needed.
    Only if that structure is absent does this fall back to thresholding
    the Track-2 segmentation masks by --flood-pixel-values, which IS a
    guess and prints a loud warning saying so.
  - FASDD_UAV: this is a detection dataset (published with COCO/YOLO/VOC
    bbox annotations for fire and smoke), not folder-per-class - handled
    with the same YOLO-label logic as D-Fire below. --fasdd-fire-id /
    --fasdd-smoke-id let you correct the class-id order if the release
    you downloaded differs; the printed per-class counts are the check.
  - D-Fire: YOLO-format bboxes (labels/*.txt, class 0=smoke, 1=fire per
    the dataset's published spec) - an image gets that class if any box
    exists, else 'normal'. D-Fire is a compiled set of independent stock
    photos, not extracted video frames, so there is no real "scene" to
    group by - a fabricated grouping here would be worse than an honest
    per-image split, so that's what this does.
"""
import argparse
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png")
CLASSES = ("fire", "smoke", "flood", "normal")


def _copy(src, dst_dir, cls_counts, cls):
    """Symlink rather than copy: three sources at ~24k images would otherwise
    duplicate several GB on a Kaggle disk that also has to hold the aerial
    set. Falls back to a real copy where symlinks aren't available."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    # prefix with the source dir so same-named files from two datasets
    # (both ship plenty of "00001.jpg") can't overwrite each other
    dst = dst_dir / f"{src.parent.name}__{src.name}"
    if dst.exists() or dst.is_symlink():
        cls_counts[cls] += 1
        return
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)
    cls_counts[cls] += 1


def _reset_out(out):
    """Re-running with a different seed/fraction must not leave the previous
    run's images behind - they'd silently inflate the class counts and put
    the same image on both sides of the split across runs. Same idempotency
    lesson as train/make_night.py."""
    if out.exists() and any(out.iterdir()):
        print(f"[scene-hazard] clearing previous build at {out}")
        shutil.rmtree(out)


def split_scenes(scene_to_items, val_frac, seed, source=""):
    """Hold out whole scenes until val reaches val_frac of the IMAGES, not of
    the scene count. Counting scenes instead is what a naive
    `n_val = max(1, len(keys) * val_frac)` does, and when a source has few
    scenes that floor of 1 hands a single huge scene - sometimes the entire
    source - to val, leaving nothing to train on. Train is always left at
    least one scene."""
    keys = sorted(scene_to_items)
    total = sum(len(scene_to_items[k]) for k in keys)
    if len(keys) < 2:
        print(f"[scene-hazard] {source}: only {len(keys)} scene(s) - can't hold any out "
              f"without giving one side everything, so all {total} images go to train "
              f"(this source contributes no val images)")
        return set(keys), set()

    rng = random.Random(seed)
    rng.shuffle(keys)
    target_val, val_keys, running = total * val_frac, set(), 0
    for k in keys:
        if running >= target_val or len(val_keys) >= len(keys) - 1:
            break
        val_keys.add(k)
        running += len(scene_to_items[k])
    return set(keys) - val_keys, val_keys


def _find_flooded_folders(root):
    """FloodNet's Track-1 classification release ships two folders whose
    names vary slightly by mirror (`Flooded`/`Non-Flooded`,
    `Flooded`/`Non Flooded`, lowercase variants). Match case/space-insensitively
    instead of hardcoding one spelling."""
    flooded = non_flooded = None
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        norm = d.name.lower().replace("-", " ").replace("_", " ").strip()
        if norm == "flooded":
            flooded = d
        elif norm in ("non flooded", "not flooded", "unflooded"):
            non_flooded = d
    return flooded, non_flooded


def build_floodnet(root, flood_pixel_values, out, val_frac, seed):
    root = Path(root)
    flooded_dir, non_flooded_dir = _find_flooded_folders(root)
    scene_to_items = defaultdict(list)

    if flooded_dir and non_flooded_dir:
        print(f"[scene-hazard] FloodNet: using Track-1 classification folders "
              f"({flooded_dir.name}/, {non_flooded_dir.name}/) - no mask guessing needed")
        for img_path in sorted(flooded_dir.glob("*")):
            if img_path.suffix.lower() in IMG_EXT:
                scene_to_items[img_path.stem].append((img_path, "flood"))
        for img_path in sorted(non_flooded_dir.glob("*")):
            if img_path.suffix.lower() in IMG_EXT:
                scene_to_items[img_path.stem].append((img_path, "normal"))
    else:
        img_dir, mask_dir = root / "images", root / "masks"
        if not img_dir.exists() or not mask_dir.exists():
            print(f"[scene-hazard] FloodNet not found at {root} (no Track-1 folders, "
                  f"no images/+masks/ either), skipping")
            return Counter()
        print(f"[scene-hazard] FloodNet: no Track-1 folders found - falling back to "
              f"thresholding Track-2 masks by pixel value {flood_pixel_values} - "
              f"THIS IS A GUESS, verify against your release's class list")
        flood_vals = set(flood_pixel_values)
        for img_path in sorted(img_dir.glob("*")):
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                continue
            mask = np.array(Image.open(mask_path))
            frac_flood = np.isin(mask, list(flood_vals)).mean()
            label = "flood" if frac_flood > 0.02 else "normal"
            scene_to_items[img_path.stem].append((img_path, label))

    train_keys, val_keys = split_scenes(scene_to_items, val_frac, seed, "FloodNet")
    counts = Counter()
    for key, items in scene_to_items.items():
        split = "val" if key in val_keys else "train"
        for img_path, label in items:
            _copy(img_path, out / split / label, counts, label)
    print(f"[scene-hazard] FloodNet: {dict(counts)}")
    return counts


def _yolo_box_label(lab_path, fire_id, smoke_id):
    """Detection -> per-image classification label: an image gets 'fire' if
    any box is the fire class, else 'smoke' if any box is the smoke class,
    else 'normal'. Fire takes priority when both appear in one frame."""
    if not lab_path.exists():
        return "normal"
    classes = {int(line.split()[0]) for line in lab_path.read_text().splitlines() if line.strip()}
    if fire_id in classes:
        return "fire"
    if smoke_id in classes:
        return "smoke"
    return "normal"


def build_fasdd(root, out, val_frac, seed, fire_id, smoke_id):
    """FASDD is published with detection annotations (fire/smoke bboxes),
    not classification folders - same YOLO-label conversion as D-Fire."""
    root = Path(root)
    img_dir, lab_dir = root / "images", root / "labels"
    if not img_dir.exists():
        print(f"[scene-hazard] FASDD_UAV not found at {root} (expected images/+labels/ "
              f"in YOLO detection format), skipping")
        return Counter()
    scene_to_items = defaultdict(list)
    # rglob, not glob: FASDD ships its images under train/val/test subdirs in
    # some releases, and a non-recursive scan silently finds zero of them
    label_by_stem = {p.stem: p for p in lab_dir.rglob("*.txt")} if lab_dir.exists() else {}
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXT:
            continue
        label = _yolo_box_label(label_by_stem.get(img_path.stem, lab_dir / "___missing.txt"),
                                fire_id, smoke_id)
        # FASDD_UAV frames come from UAV video capture, so consecutive frames
        # of one flight are near-duplicates - group by the filename prefix
        # before the trailing frame number to keep one flight on one side of
        # the split. A purely numeric stem has no prefix to group on, so fall
        # back to the containing directory rather than to "" (which would
        # collapse every such frame into a single giant scene).
        m = re.match(r"^(.+?)[_-]?(\d{3,})$", img_path.stem)
        scene = (m.group(1) if m and m.group(1) else img_path.parent.name)
        scene_to_items[scene].append((img_path, label))
    train_keys, val_keys = split_scenes(scene_to_items, val_frac, seed, "FASDD_UAV")
    counts = Counter()
    for key, items in scene_to_items.items():
        split = "val" if key in val_keys else "train"
        for img_path, label in items:
            _copy(img_path, out / split / label, counts, label)
    print(f"[scene-hazard] FASDD_UAV: {dict(counts)}")
    return counts


def build_dfire(root, sample_frac, out, val_frac, seed, fire_id, smoke_id):
    """D-Fire is a compiled set of independent stock photos (not video
    frames), so there is no real scene/clip to group by - a fabricated
    grouping would be worse than an honest per-image split."""
    root = Path(root)
    img_dir, lab_dir = root / "images", root / "labels"
    if not img_dir.exists():
        print(f"[scene-hazard] D-Fire not found at {root}, skipping")
        return Counter()
    label_by_stem = {p.stem: p for p in lab_dir.rglob("*.txt")} if lab_dir.exists() else {}
    items = []
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXT:
            continue
        label = _yolo_box_label(label_by_stem.get(img_path.stem, lab_dir / "___missing.txt"),
                                fire_id, smoke_id)
        items.append((img_path, label))

    rng = random.Random(seed)
    rng.shuffle(items)
    target = int(len(items) * sample_frac)
    picked = items[:target]
    n_val = max(1, int(len(picked) * val_frac))
    counts = Counter()
    for img_path, label in picked[:n_val]:
        _copy(img_path, out / "val" / label, counts, label)
    for img_path, label in picked[n_val:]:
        _copy(img_path, out / "train" / label, counts, label)
    print(f"[scene-hazard] D-Fire: sampled {len(picked)}/{len(items)} images -> {dict(counts)}")
    return counts


def downsample_normal(out, seed):
    for split in ("train", "val"):
        normal_dir = out / split / "normal"
        if not normal_dir.exists():
            continue
        hazard_max = max((len(list((out / split / c).glob("*")))
                           for c in ("fire", "smoke", "flood") if (out / split / c).exists()),
                          default=0)
        normal_files = list(normal_dir.glob("*"))
        if hazard_max and len(normal_files) > 10 * hazard_max:
            rng = random.Random(seed)
            keep = set(rng.sample(normal_files, 10 * hazard_max))
            removed = 0
            for f in normal_files:
                if f not in keep:
                    f.unlink()
                    removed += 1
            print(f"[scene-hazard] {split}/normal: downsampled by {removed} "
                  f"(was >10x the largest hazard class, {hazard_max})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floodnet", default=None)
    ap.add_argument("--fasdd", default=None)
    ap.add_argument("--dfire", default=None)
    ap.add_argument("--dfire-frac", type=float, default=0.20)
    ap.add_argument("--dfire-fire-id", type=int, default=1,
                     help="D-Fire's published class order is 0=smoke, 1=fire")
    ap.add_argument("--dfire-smoke-id", type=int, default=0)
    ap.add_argument("--fasdd-fire-id", type=int, default=0,
                     help="FASDD's class order isn't as widely documented as D-Fire's - "
                          "check the release's data.yaml/classes.txt and correct this "
                          "if the printed counts look inverted")
    ap.add_argument("--fasdd-smoke-id", type=int, default=1)
    ap.add_argument("--flood-pixel-values", default="4,5",
                     help="fallback only, used if FloodNet's Track-1 classification "
                          "folders aren't found - comma-separated mask pixel values "
                          "meaning 'flooded', VERIFY against your release's class list")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--keep-existing", action="store_true",
                     help="don't clear a previous build first (default is to clear, so a "
                          "re-run with a different seed can't leave stale images behind "
                          "that inflate the counts and straddle the split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="datasets/scene_hazard")
    a = ap.parse_args()

    out = Path(a.out)
    if not a.keep_existing:
        _reset_out(out)
    total = Counter()
    if a.floodnet:
        vals = [int(v) for v in a.flood_pixel_values.split(",")]
        total += build_floodnet(a.floodnet, vals, out, a.val_frac, a.seed)
    if a.fasdd:
        total += build_fasdd(a.fasdd, out, a.val_frac, a.seed, a.fasdd_fire_id, a.fasdd_smoke_id)
    if a.dfire:
        total += build_dfire(a.dfire, a.dfire_frac, out, a.val_frac, a.seed, a.dfire_fire_id, a.dfire_smoke_id)

    downsample_normal(out, a.seed)

    for cls in CLASSES:
        n_train = len(list((out / "train" / cls).glob("*"))) if (out / "train" / cls).exists() else 0
        n_val = len(list((out / "val" / cls).glob("*"))) if (out / "val" / cls).exists() else 0
        print(f"[scene-hazard] {cls}: train={n_train} val={n_val}")
    print(f"[scene-hazard] dataset ready at {out} "
          f"(yolo classify train data={out} model=yolo11n-cls.pt imgsz=224 epochs=15 batch=32)")


if __name__ == "__main__":
    main()
