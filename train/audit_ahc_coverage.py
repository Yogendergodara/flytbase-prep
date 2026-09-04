"""Report exactly which labelled videos are MISSING from the download, so
they can be re-fetched by id instead of re-downloading everything blind.

The AHC ground_truth.csv files describe far more videos than the 5 Google
Drive mirror exports actually delivered (each export was partial, and
different mirrors truncated different classes). This script quantifies the
gap per class and writes the missing video_ids to a file.

    python train/audit_ahc_coverage.py --root datasets/AHC_full --out out/ahc_missing.txt

Recovering these is the single highest-value action available for accuracy:
it roughly DOUBLES the real labelled dataset. No amount of frame sampling or
augmentation substitutes for a video that was never downloaded.
"""
import argparse
import csv
from pathlib import Path


def audit_class(cls_dir, cls_name):
    gt = cls_dir / "ground_truth.csv"
    vids = cls_dir / "videos"
    if not gt.exists():
        return None
    rows = list(csv.DictReader(gt.open(encoding="utf-8")))
    wanted = {r["video_id"] for r in rows}
    have = {p.stem for p in vids.glob("*.mp4")} if vids.exists() else set()
    missing = sorted(wanted - have)
    return {
        "class": cls_name, "events": len(rows), "videos_wanted": len(wanted),
        "videos_have": len(wanted) - len(missing), "missing": missing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/AHC_full")
    ap.add_argument("--out", default="out/ahc_missing.txt")
    a = ap.parse_args()

    root = Path(a.root)
    reports = []
    for cls_dir in sorted((root / "train").iterdir()):
        if cls_dir.is_dir():
            r = audit_class(cls_dir, cls_dir.name)
            if r:
                reports.append(r)
    test_r = audit_class(root / "test", "__test__")
    if test_r:
        reports.append(test_r)

    print(f"{'class':34s} {'have':>6s} {'want':>6s} {'missing':>8s}  {'%':>5s}")
    tot_have = tot_want = 0
    for r in reports:
        pct = 100 * r["videos_have"] / r["videos_wanted"] if r["videos_wanted"] else 0
        tot_have += r["videos_have"]
        tot_want += r["videos_wanted"]
        flag = "  <-- badly incomplete" if pct < 50 else ""
        print(f"{r['class']:34s} {r['videos_have']:6d} {r['videos_wanted']:6d} "
              f"{len(r['missing']):8d}  {pct:4.0f}%{flag}")
    print(f"{'TOTAL':34s} {tot_have:6d} {tot_want:6d} {tot_want - tot_have:8d}  "
          f"{100 * tot_have / tot_want:4.0f}%")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in reports:
            for vid in r["missing"]:
                f.write(f"{r['class']},{vid}\n")
    print(f"\n[audit] {tot_want - tot_have} missing video ids -> {out}")
    print("[audit] these exist in the official labels but were never downloaded. "
          "Recovering them is worth more than any augmentation - it is real "
          "labelled data that currently contributes nothing.")


if __name__ == "__main__":
    main()
