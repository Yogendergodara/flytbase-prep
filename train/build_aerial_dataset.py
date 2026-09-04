"""Build Pool 1 (DATASET_PLAN.md): VisDrone + a stratified UIT-ADrone sample,
merged into one YOLO-detection dataset for the aerial detector.

VisDrone is used unchanged. UIT-ADrone contributes only a video-level sample
(never a frame-level one - a random 7% of frames would put near-duplicate
frames from the same clip on both sides of train/val) and only for classes
that actually exist in VisDrone's 10-class map; anything else is dropped, not
guessed onto the nearest box.

    python train/build_aerial_dataset.py \\
        --visdrone datasets/VisDrone \\
        --uit-json datasets/UIT-ADrone/train.json \\
        --uit-images datasets/UIT-ADrone/images \\
        --out datasets/aerial_combined

Sampled UIT-ADrone frames are SYMLINKED into `<out>/images/` with their
converted labels in `<out>/labels/`, rather than left in place. That is not
tidiness: Ultralytics finds a label by replacing the last `/images/` in the
image path with `/labels/`, so a label written anywhere else is silently
never found and those frames train as pure background - 40k unlabelled
images would quietly poison the run rather than fail it. Symlinks also keep
this working on Kaggle, where the downloaded dataset dir is read-only.

UNVERIFIED ASSUMPTIONS (this repo has never seen the real UIT-ADrone json -
DATASET_PLAN.md was written from the dataset's public description, not a
downloaded copy). Check the printed video-grouping key and class-remap table
against the first run's output before trusting it:
  - COCO-style json with top-level "images" (id, file_name, width, height,
    plus one of video_id/video/clip_id/sequence) and "annotations"
    (image_id, category_id, bbox=[x,y,w,h] absolute pixels) and "categories"
    (id, name).
  - If no video-grouping field exists, the frame's parent directory is used,
    then a "<video>_<frame>" filename pattern. If neither holds, this script
    refuses to guess and exits - grouping by frame instead would silently
    violate the repo's split-by-video rule.
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# VisDrone's 10 classes, in Ultralytics' order (must match scripts/make_night_yaml.py)
VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
                   "tricycle", "awning-tricycle", "bus", "motor"]

# name-substring synonyms -> VisDrone class. UNVERIFIED against UIT-ADrone's
# real category names - the printed mapping table is the actual check.
# Matched longest-key-first (see _remap_category): a plain "first match in
# insertion order" scan sends "motorbike" to bicycle (it contains "bike")
# and "awning-tricycle" to tricycle, silently mislabelling every box of
# those classes - exactly the corruption DATASET_PLAN.md warns about.
SYNONYMS = {
    "pedestrian": "pedestrian", "walker": "pedestrian",
    "person": "people", "people": "people", "human": "people",
    "bicycle": "bicycle", "bike": "bicycle", "cyclist": "bicycle",
    "car": "car", "sedan": "car",
    "van": "van",
    "truck": "truck", "lorry": "truck",
    "tricycle": "tricycle", "auto-rickshaw": "tricycle", "rickshaw": "tricycle",
    "awning-tricycle": "awning-tricycle", "awning tricycle": "awning-tricycle",
    "bus": "bus",
    "motor": "motor", "motorbike": "motor", "motorcycle": "motor", "scooter": "motor",
}
_SYNONYMS_BY_LENGTH = sorted(SYNONYMS.items(), key=lambda kv: -len(kv[0]))


def _video_key(img_entry):
    for field in ("video_id", "video", "clip_id", "sequence", "seq"):
        if field in img_entry:
            return str(img_entry[field])
    rel = Path(img_entry["file_name"])
    # a parent dir is the most reliable grouping when the json ships frames
    # as "<video>/<frame>.jpg" - and taking .stem alone would throw it away,
    # collapsing every numerically-named frame into one bucket
    if rel.parent != Path("."):
        return str(rel.parent)
    m = re.match(r"^(.*?)[_-](\d+)$", rel.stem) or re.match(r"^(.+?)(\d{3,})$", rel.stem)
    if m and m.group(1):
        return m.group(1)
    return None


def _remap_category(name):
    low = name.lower().strip()
    if low in VISDRONE_NAMES:
        return low
    for key, target in _SYNONYMS_BY_LENGTH:
        if key in low:
            return target
    return None


def build_class_map(categories):
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    remap, dropped = {}, []
    for cid, name in cat_id_to_name.items():
        target = _remap_category(name)
        if target is None:
            dropped.append(name)
        else:
            remap[cid] = VISDRONE_NAMES.index(target)
    print("[build-aerial] UIT-ADrone -> VisDrone class remap:")
    for cid, name in sorted(cat_id_to_name.items()):
        dest = VISDRONE_NAMES[remap[cid]] if cid in remap else "DROPPED (no VisDrone equivalent)"
        print(f"  {name!r} (id {cid}) -> {dest}")
    if dropped:
        print(f"[build-aerial] {len(dropped)} category name(s) dropped entirely: {dropped}")
    return remap


def group_videos(images):
    by_video = defaultdict(list)
    unresolved = 0
    for img in images:
        key = _video_key(img)
        if key is None:
            unresolved += 1
            continue
        by_video[key].append(img)
    if unresolved:
        raise SystemExit(
            f"[build-aerial] refusing to continue: {unresolved} of {len(images)} images "
            f"have no resolvable video id (no video_id/video/clip_id/sequence field, no "
            f"parent directory, and file_name doesn't match '<video>_<frame>'). Grouping "
            f"these by frame instead would violate the split-by-video rule - fix the "
            f"grouping key for this dataset release instead of ignoring this.")
    print(f"[build-aerial] {len(images)} frames grouped into {len(by_video)} videos")
    return by_video


def _video_classes(by_video, stratify_field):
    """One anomaly-class label per video, for stratified sampling. Returns
    None if the field isn't in this release's json - the caller then falls
    back to plain random sampling and says so, rather than pretending the
    'no class starved' guarantee in DATASET_PLAN.md was honoured."""
    if not stratify_field:
        return None
    out = {}
    for key, frames in by_video.items():
        vals = [f[stratify_field] for f in frames if stratify_field in f]
        if not vals:
            return None
        out[key] = str(Counter(map(str, vals)).most_common(1)[0][0])
    return out


def sample_videos(by_video, target_frames, seed, video_class=None):
    keys = sorted(by_video)  # sort before shuffling so the seed is reproducible
    rng = random.Random(seed)

    if video_class:
        # round-robin across anomaly classes so a rare class isn't starved by
        # a purely random draw (DATASET_PLAN.md's Pool 1 split rule)
        per_class = defaultdict(list)
        for k in keys:
            per_class[video_class[k]].append(k)
        for v in per_class.values():
            rng.shuffle(v)
        order, cursors = [], {c: 0 for c in per_class}
        while len(order) < len(keys):
            for cls in sorted(per_class):
                i = cursors[cls]
                if i < len(per_class[cls]):
                    order.append(per_class[cls][i])
                    cursors[cls] = i + 1
        print(f"[build-aerial] stratifying across {len(per_class)} anomaly classes: "
              f"{ {c: len(v) for c, v in sorted(per_class.items())} }")
    else:
        order = list(keys)
        rng.shuffle(order)
        print("[build-aerial] no stratify field - plain random video sampling "
              "(DATASET_PLAN.md's 'no class starved' guarantee is NOT in force)")

    picked, total = [], 0
    for k in order:
        if total >= target_frames:
            break
        picked.append(k)
        total += len(by_video[k])
    print(f"[build-aerial] sampled {len(picked)} of {len(keys)} videos, "
          f"{total} frames (target {target_frames})")
    return picked, total


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x,y,w,h] in absolute pixels -> YOLO normalized cx,cy,w,h, clipped
    to the frame. COCO boxes routinely run a few pixels past the edge; passing
    those through unclipped makes Ultralytics reject the label file."""
    x, y, w, h = bbox
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return ((x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h,
            (x1 - x0) / img_w, (y1 - y0) / img_h)


def _link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)  # Windows without dev-mode, or a filesystem without symlinks


def _image_size(img_entry, src_path):
    w, h = img_entry.get("width"), img_entry.get("height")
    if w and h:
        return int(w), int(h)
    import cv2
    im = cv2.imread(str(src_path))
    if im is None:
        return None
    return im.shape[1], im.shape[0]


def write_uit_subset(coco, picked_keys, by_video, remap, images_root, out_dir):
    images_root, out_dir = Path(images_root), Path(out_dir)
    dst_images, dst_labels = out_dir / "images", out_dir / "labels"
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    img_paths = []
    n_boxes = n_dropped_cls = n_dropped_geom = n_missing = n_no_label = 0
    for key in picked_keys:
        for img in by_video[key]:
            src_rel = Path(img["file_name"])
            src_path = images_root / src_rel
            if not src_path.exists():
                n_missing += 1
                continue
            size = _image_size(img, src_path)
            if size is None:
                n_missing += 1
                continue
            img_w, img_h = size
            lines = []
            for ann in ann_by_image.get(img["id"], []):
                if ann["category_id"] not in remap:
                    n_dropped_cls += 1
                    continue
                box = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                if box is None:
                    n_dropped_geom += 1
                    continue
                cx, cy, w, h = box
                lines.append(f"{remap[ann['category_id']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                n_boxes += 1
            if not lines:
                n_no_label += 1
                continue  # nothing usable left on this frame after the remap
            # image and label must mirror each other around /images/ and
            # /labels/ or Ultralytics will never pair them up
            dst_img = dst_images / src_rel
            _link_or_copy(src_path.resolve(), dst_img)
            dst_lab = dst_labels / src_rel.with_suffix(".txt")
            dst_lab.parent.mkdir(parents=True, exist_ok=True)
            dst_lab.write_text("\n".join(lines), encoding="utf-8")
            img_paths.append(str(dst_img.resolve()))

    print(f"[build-aerial] {len(img_paths)} UIT-ADrone frames linked into {dst_images} "
          f"with labels in {dst_labels}")
    print(f"[build-aerial] {n_boxes} boxes kept, {n_dropped_cls} dropped (unmapped class), "
          f"{n_dropped_geom} dropped (degenerate/out-of-frame box)")
    if n_no_label:
        print(f"[build-aerial] {n_no_label} frames skipped - no usable box left after remap")
    if n_missing:
        print(f"[build-aerial] WARNING: {n_missing} frames listed in the json are missing "
              f"under {images_root} (or unreadable) - check --uit-images points at the "
              f"root that file_name is relative to")
    return img_paths


def write_drone_anomaly_manifest(drone_anomaly_dir, out_dir):
    """Held-out eval only - never referenced by train_aerial.py's data.yaml."""
    src = Path(drone_anomaly_dir)
    if not src.exists():
        print(f"[build-aerial] --drone-anomaly {src} not found, skipping eval manifest")
        return
    frames = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = Path(out_dir) / "drone_anomaly_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.txt").write_text("\n".join(str(p.resolve()) for p in frames), encoding="utf-8")
    print(f"[build-aerial] {len(frames)} Drone-Anomaly frames -> {out}/manifest.txt "
          f"(eval only, frame-binary labels, no bboxes)")


def _list_images(d):
    return sorted(str(p.resolve()) for p in Path(d).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visdrone", required=True, help="existing datasets/VisDrone (unchanged)")
    ap.add_argument("--uit-json", required=True, help="UIT-ADrone train.json (COCO format)")
    ap.add_argument("--uit-images", required=True,
                     help="UIT-ADrone image root that the json's file_name is relative to")
    ap.add_argument("--drone-anomaly", default=None, help="optional Drone-Anomaly root, eval-only")
    ap.add_argument("--stratify-field", default="anomaly_class",
                     help="per-frame json field naming the anomaly class, used to spread the "
                          "video sample across all classes. Falls back to plain random "
                          "sampling (and says so) if the field isn't present.")
    ap.add_argument("--target-frames", type=int, default=40000,
                     help="DATASET_PLAN.md's own conservative estimate was 15000 "
                          "(~7% of UIT-ADrone) to avoid the detector overfitting to "
                          "UIT-ADrone's few camera locations; 40000 (~19%) trades some "
                          "of that safety margin for more data, per explicit ask. Above "
                          "~60000 you are approaching 1:1 with VisDrone itself and the "
                          "combined set risks being UIT-ADrone-flavored rather than "
                          "general - the mandatory A/B is what actually tells you if a "
                          "given size helped or hurt, don't assume from size alone.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="datasets/aerial_combined")
    a = ap.parse_args()

    visdrone_train_dir = Path(a.visdrone) / "images" / "train"
    visdrone_val_dir = Path(a.visdrone) / "images" / "val"
    for d in (visdrone_train_dir, visdrone_val_dir):
        if not d.is_dir():
            raise SystemExit(f"[build-aerial] {d} not found - point --visdrone at the "
                             f"VisDrone root containing images/train and images/val")

    out = Path(a.out)
    coco = json.loads(Path(a.uit_json).read_text(encoding="utf-8"))
    remap = build_class_map(coco["categories"])
    if not remap:
        raise SystemExit("[build-aerial] refusing to continue: not one UIT-ADrone category "
                         "mapped onto a VisDrone class. Fix SYNONYMS for this release's "
                         "category names - building the set anyway would add 40k frames "
                         "with no boxes at all.")
    by_video = group_videos(coco["images"])
    video_class = _video_classes(by_video, a.stratify_field)
    picked, _ = sample_videos(by_video, a.target_frames, a.seed, video_class)
    uit_paths = write_uit_subset(coco, picked, by_video, remap, a.uit_images, out)

    vd_train, vd_val = _list_images(visdrone_train_dir), _list_images(visdrone_val_dir)

    out.mkdir(parents=True, exist_ok=True)
    train_txt, val_txt = (out / "train.txt").resolve(), (out / "val.txt").resolve()
    train_txt.write_text("\n".join(vd_train + uit_paths), encoding="utf-8")
    # val stays VisDrone-only and untouched - the mandatory A/B (DATASET_PLAN.md
    # Phase D5) compares this run against the VisDrone-only run on the SAME val
    # set, which only works if neither run's val set moved.
    val_txt.write_text("\n".join(vd_val), encoding="utf-8")
    # absolute paths in data.yaml, not relative - Ultralytics resolves relative
    # train/val entries against different bases across versions (yaml dir vs
    # cwd vs `path:`); absolute paths sidestep that ambiguity entirely, same
    # as scripts/make_night_yaml.py already does for the night-only val list.
    yaml_text = (f"train: {train_txt}\nval: {val_txt}\nnames:\n"
                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(VISDRONE_NAMES)))
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"[build-aerial] train: {len(vd_train)} VisDrone + {len(uit_paths)} UIT-ADrone "
          f"= {len(vd_train) + len(uit_paths)} images")
    print(f"[build-aerial] val: {len(vd_val)} VisDrone images (unchanged)")
    print(f"[build-aerial] wrote {out}/data.yaml")

    if a.drone_anomaly:
        write_drone_anomaly_manifest(a.drone_anomaly, out.parent)


if __name__ == "__main__":
    main()
