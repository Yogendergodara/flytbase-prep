"""Merge the 5 downloaded AHC dataset mirrors into one complete copy.

Each of the 5 "mirror" downloads under datasets/ turned out to be a PARTIAL
Google Drive folder export - different classes are missing or near-empty in
each one (confirmed by inspection: e.g. Mirror 4 has 0 videos for
loitering_or_suspicious_presence despite the folder/csv existing, but
Mirror 5 has 184 of them). No single mirror is complete; the union of all
five is. This is not a guess - it was checked file-by-file per class before
writing this script.

    python train/consolidate_ahc_dataset.py --src datasets --out datasets/AHC_full

Dedupes by filename (video IDs like TR03107.mp4 are globally unique in this
dataset), so a clip present in two mirrors is copied once, not twice.
ground_truth.csv / videos.csv rows are concatenated and deduped by whatever
id column they use, keeping the first occurrence.
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def _find_mirrors(src):
    # each mirror's actual content sits one level inside the download
    # folder, under a re-nested "<Mirror Name>/" dir
    mirrors = []
    for d in sorted(src.iterdir()):
        if not d.is_dir():
            continue
        inner = [c for c in d.iterdir() if c.is_dir() and (c / "train").exists()]
        mirrors += inner if inner else ([d] if (d / "train").exists() else [])
    return mirrors


def _merge_csv(csv_paths, out_path):
    rows, seen, header = [], set(), None
    for p in csv_paths:
        if not p.exists():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            h = next(r, None)
            if h is None:
                continue
            header = header or h
            id_col = 0  # video_id is the first column in both videos.csv and ground_truth.csv
            for row in r:
                key = row[id_col] if row else None
                if key and key in seen and header == h:
                    continue  # same video_id already merged in from an earlier mirror
                if key:
                    seen.add(key)
                rows.append(row)
    if header:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    return len(rows)


def _merge_videos(video_dirs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, copied = set(), 0
    for vd in video_dirs:
        if not vd.exists():
            continue
        for p in sorted(vd.glob("*.mp4")):
            if p.name in seen:
                continue
            seen.add(p.name)
            dst = out_dir / p.name
            if not dst.exists():
                shutil.copyfile(p, dst)
            copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets", help="dir containing the 5 downloaded mirror folders")
    ap.add_argument("--out", default="datasets/AHC_full")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    mirrors = _find_mirrors(src)
    print(f"[consolidate] found {len(mirrors)} mirror(s): {[m.name for m in mirrors]}")
    if not mirrors:
        raise SystemExit(f"[consolidate] no mirror with a train/ dir found under {src}")

    classes = sorted({c.name for m in mirrors for c in (m / "train").iterdir() if c.is_dir()})
    print(f"[consolidate] {len(classes)} classes across all mirrors: {classes}")

    for cls in classes:
        vdirs = [m / "train" / cls / "videos" for m in mirrors]
        n_vid = _merge_videos(vdirs, out / "train" / cls / "videos")
        n_gt = _merge_csv([m / "train" / cls / "ground_truth.csv" for m in mirrors],
                          out / "train" / cls / "ground_truth.csv")
        n_vc = _merge_csv([m / "train" / cls / "videos.csv" for m in mirrors],
                          out / "train" / cls / "videos.csv")
        print(f"[consolidate] train/{cls}: {n_vid} videos, {n_gt} ground_truth rows, {n_vc} videos.csv rows")

    n_test_vid = _merge_videos([m / "test" / "videos" for m in mirrors], out / "test" / "videos")
    n_test_gt = _merge_csv([m / "test" / "ground_truth.csv" for m in mirrors], out / "test" / "ground_truth.csv")
    n_test_vc = _merge_csv([m / "test" / "videos.csv" for m in mirrors], out / "test" / "videos.csv")
    print(f"[consolidate] test: {n_test_vid} videos, {n_test_gt} ground_truth rows, {n_test_vc} videos.csv rows")
    print(f"[consolidate] done -> {out}. Originals under {src} untouched - "
          f"delete them yourself once you've spot-checked this output.")


if __name__ == "__main__":
    main()
