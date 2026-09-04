"""Step 1 of the AHC VLM fine-tune pipeline: walk datasets/AHC_full class by
class, read each ground_truth.csv row by row, and turn each labelled event
into one or more TRAINING EXAMPLES, each with its own sampled frames.

    python train/extract_ahc_frames.py --root datasets/AHC_full \\
        --out datasets/AHC_frames --frames-per-crop 8 --crops-per-event 3

WHAT ACTUALLY MULTIPLIES TRAINING DATA HERE (read this before tuning knobs):
For the VLM classifier, ONE TRAINING EXAMPLE = ONE EVENT, with all its frames
fed in together as a single multi-image input. So raising frames-per-event
from 8 to 20 does NOT give more examples - it gives the same number of
examples with denser views of each. What genuinely raises the example count:

  --crops-per-event N   different temporal sub-windows of the same event.
                        Each crop is a genuinely different view (different
                        frames), so each is a real extra example. Crops are
                        only generated when the event is LONG ENOUGH for the
                        sub-windows to differ; a 3-second event yields one
                        crop, not N copies of itself. Faking variety by
                        duplicating identical frames would just be
                        oversampling wearing a disguise - finetune_ahc_vlm.py
                        does oversampling explicitly and says so, so this
                        script does not do it silently.
  Horizontal-flip augmentation is NOT done here. Profiled on this machine:
  video decoding is fast (~1.8s/video for 3 crops x 8 frames), but writing
  each crop's frames as separate JPEGs is not (~21 files/sec, filesystem/AV-
  scan overhead per small file dominates) - a flip variant materialized as
  its own set of on-disk files would roughly DOUBLE extraction wall-clock
  for a transform that is free to apply in memory. finetune_ahc_vlm.py
  applies the flip at training-example-build time instead - same augmentation
  benefit, zero extra extraction cost. This is a real measured tradeoff, not
  a style preference - see the profiling note in git history if it's ever
  worth revisiting on a faster filesystem.

THE BIGGEST LEVER IS NOT IN THIS SCRIPT. train/audit_ahc_coverage.py shows
~1,668 labelled videos (52% of the dataset) exist in the official
ground_truth.csv files but were never downloaded. Recovering those roughly
doubles the real data. No crop or flip setting substitutes for that.

Confirmed against the real merged dataset before writing this (not guessed):
  - video_id == the .mp4 filename stem exactly (TR01350 -> TR01350.mp4).
  - is_anomaly is the literal string "true"/"false", not a Python bool.
  - start/end are BLANK for every "normal" row and populated for anomaly
    rows, so blank reliably means "use the whole video".
  - all 3,173 train rows have a non-blank description_summary.
  - one video can carry several events (loitering: 184 videos, 300 events).
"""
import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# `python train/extract_ahc_frames.py` puts only train/ on sys.path, not the
# repo root - `from pipeline...` then fails with ModuleNotFoundError despite
# working fine under pytest (which adds the root itself). Confirmed by
# actually running this script, not assumed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.vlm_judge import extract_frames

# a sub-window shorter than this isn't a meaningful view of an event, and
# two sub-windows of a clip this short are the same frames twice
MIN_SUB_WINDOW_SEC = 1.5


def _video_duration(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    return (n_frames / fps) if fps > 0 else 0.0


def crop_windows(t0, t1, n_crops, coverage=0.6):
    """Sub-windows of [t0,t1] that tile the event with overlap. Returns ONE
    window when the event is too short for the crops to actually differ -
    the caller then produces one example, not n identical ones."""
    duration = t1 - t0
    if n_crops <= 1:
        return [(t0, t1)]
    sub = duration * coverage
    if sub < MIN_SUB_WINDOW_SEC or duration < MIN_SUB_WINDOW_SEC * 1.5:
        return [(t0, t1)]
    step = (duration - sub) / (n_crops - 1)
    return [(t0 + i * step, t0 + i * step + sub) for i in range(n_crops)]


def process_class(cls_dir, cls_name, out_dir, args, is_test):
    gt_path = cls_dir / "ground_truth.csv"
    video_dir = cls_dir / "videos"
    if not gt_path.exists():
        print(f"[extract] {cls_name}: no ground_truth.csv, skipping")
        return []

    rows = list(csv.DictReader(gt_path.open(encoding="utf-8")))
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)

    video_ids = sorted(by_video)
    if args.limit_per_class:
        video_ids = video_ids[:args.limit_per_class]

    manifest_rows, n_missing, n_bad, n_short = [], 0, 0, 0
    # test events are evaluation data - never crop them, or the reported
    # score stops describing the real 34-video public test set
    n_crops = 1 if is_test else args.crops_per_event

    for video_id in video_ids:
        events = by_video[video_id]
        vpath = video_dir / f"{video_id}.mp4"
        if not vpath.exists():
            n_missing += len(events)
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            n_bad += len(events)
            cap.release()
            continue
        duration = _video_duration(cap)

        for i, r in enumerate(events):
            t0_raw, t1_raw = r["start_time_sec"].strip(), r["end_time_sec"].strip()
            if t0_raw and t1_raw:
                t0, t1 = float(t0_raw), float(t1_raw)
                if t1 - t0 < 0.1:
                    # a handful of source rows give start==end (a flagged
                    # instant, not a range) - sampling k frames across a
                    # zero-length window gives the SAME frame k times, not a
                    # sequence. Pad symmetrically around the instant instead.
                    # This must NOT trigger on legitimately short real events
                    # (a genuine 1.2s anomaly window is valid data, not a
                    # bug) - 0.1s only catches the actual start==end
                    # pathology. Confirmed against the real data: exactly 7
                    # of 4,220 events are start==end; 60 have some other
                    # short-but-real duration and must be left alone.
                    pad = 1.5
                    t0, t1 = max(0.0, t0 - pad), min(duration or t1 + pad, t1 + pad)
            else:
                t0, t1 = 0.0, max(duration, 0.1)   # normal / level-1: whole clip

            windows = crop_windows(t0, t1, n_crops)
            if n_crops > 1 and len(windows) == 1:
                n_short += 1

            for c, (w0, w1) in enumerate(windows):
                frames = extract_frames(str(vpath), w0, w1, args.frames_per_crop, cap=cap)
                if not frames:
                    n_bad += 1
                    continue
                evt_dir = out_dir / cls_name / f"{video_id}__evt{i}__c{c}"
                evt_dir.mkdir(parents=True, exist_ok=True)
                frame_paths = []
                from PIL import Image
                for j, f in enumerate(frames):
                    p = evt_dir / f"frame_{j:02d}.jpg"
                    Image.fromarray(np.ascontiguousarray(f)).save(p, quality=90)
                    # relative to --out, NOT absolute: this manifest gets
                    # uploaded to Kaggle alongside the AHC_frames folder, and
                    # an absolute local-machine path (F:\hackthone\...) does
                    # not exist there. finetune_ahc_vlm.py / eval_ahc_vlm.py
                    # join this back with --frames-root at load time.
                    # .as_posix(), NOT str(): this repo builds on Windows but
                    # trains on Kaggle (Linux) - a Windows relative path
                    # stringifies with backslashes, which POSIX pathlib
                    # treats as a literal filename character, not a
                    # separator, so the join on Kaggle would silently look
                    # for one file with backslashes IN its name and fail.
                    frame_paths.append(p.resolve().relative_to(out_dir.resolve()).as_posix())

                manifest_rows.append({
                    "video_id": video_id, "event_index": i, "crop_index": c,
                    # the CSV row's OWN class_name, not the folder-level
                    # cls_name arg - identical for train (each class has its
                    # own folder), but for test every class shares one folder
                    # processed under the placeholder name "__test__", so
                    # cls_name here would have silently stored "__test__" as
                    # the ground-truth label for every one of the 32 test
                    # events - breaking eval_ahc_vlm.py's per-class scoring
                    # for the entire test set. Confirmed by actually
                    # inspecting the manifest, not assumed.
                    "class_name": r["class_name"],
                    "is_anomaly": r["is_anomaly"].strip().lower() == "true",
                    "t_start": w0, "t_end": w1,
                    "description_summary": r["description_summary"],
                    "frame_paths": frame_paths,
                    "split": "test" if is_test else "train",
                })
        cap.release()

    if n_missing:
        print(f"[extract] {cls_name}: {n_missing} event(s) skipped - video not "
              f"downloaded (see train/audit_ahc_coverage.py)")
    if n_bad:
        print(f"[extract] {cls_name}: {n_bad} event(s) skipped - unreadable file "
              f"or frame extraction failed")
    if n_short:
        print(f"[extract] {cls_name}: {n_short} event(s) too short for "
              f"{n_crops} distinct crops - 1 crop used instead of faking variety")
    n_events = len({(m['video_id'], m['event_index']) for m in manifest_rows})
    print(f"[extract] {cls_name}: {len(manifest_rows)} examples from {n_events} events")
    return manifest_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/AHC_full")
    ap.add_argument("--out", default="datasets/AHC_frames")
    ap.add_argument("--frames-per-crop", type=int, default=8,
                     help="frames per training example. More frames = denser view "
                          "of the SAME example, not more examples, and costs GPU "
                          "memory per training step. Use --crops-per-event to get "
                          "more examples.")
    ap.add_argument("--crops-per-event", type=int, default=3,
                     help="temporal sub-windows per event, each a separate training "
                          "example. Silently reduced to 1 for events too short for "
                          "the crops to genuinely differ.")
    ap.add_argument("--limit-per-class", type=int, default=0,
                     help="only process the first N videos per class - for a fast "
                          "verification run before committing to the full extraction")
    ap.add_argument("--classes", default=None,
                     help="comma-separated subset of classes to process (plus "
                          "'__test__' for the test split) - lets a full extraction "
                          "be run in chunks that fit under a single command's time "
                          "limit, each appended to the same manifest rather than "
                          "overwriting it")
    ap.add_argument("--append", action="store_true",
                     help="append to --manifest instead of overwriting it - use "
                          "with --classes to build the manifest up across "
                          "several chunked invocations")
    ap.add_argument("--manifest", default="train/ahc_manifest.jsonl")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)

    if not root.is_dir():
        raise SystemExit(
            f"[extract] --root {root} does not exist. On Kaggle the dataset "
            f"mount path is not the same as a local path - check "
            f"`ls /kaggle/input/` and point --root at the folder that "
            f"actually contains train/ and test/.")
    if not (root / "train").is_dir():
        raise SystemExit(
            f"[extract] {root} exists but has no train/ subdirectory. "
            f"Contents: {sorted(p.name for p in root.iterdir())[:10]}")

    # A killed/partial run leaves frame files on disk but writes no manifest
    # (the manifest is written once, at the end). A LATER run then produces a
    # manifest that can reference the earlier run's leftovers - which is how
    # a real fine-tune died on FileNotFoundError for a frame that was never
    # written. Clearing first makes frames and manifest describe the same
    # run. Same idempotency lesson as build_scene_classifier_dataset.py.
    if not a.append and out.exists() and any(out.iterdir()):
        print(f"[extract] clearing previous extraction at {out} so the frames "
              f"and the manifest describe the same run")
        shutil.rmtree(out)

    all_rows = []

    train_root = root / "train"
    classes = sorted(c.name for c in train_root.iterdir() if c.is_dir())
    wanted = set(a.classes.split(",")) if a.classes else None
    print(f"[extract] {len(classes)} classes total, {a.frames_per_crop} frames/example, "
          f"{a.crops_per_event} crops/event"
          + (f", running only: {sorted(wanted)}" if wanted else ""))
    # Write the manifest incrementally, one class at a time, and keep going
    # if a single class blows up. This is a 75-minute stage: previously the
    # manifest was written only after every class finished, so one malformed
    # CSV cell, one missing column, or a full disk anywhere in it left ~34k
    # orphan JPEGs and no manifest at all - and the next run cleared them
    # and started over.
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    manifest_f = open(a.manifest, "a" if a.append else "w", encoding="utf-8")

    def flush(rows):
        for r in rows:
            manifest_f.write(json.dumps(r) + "\n")
        manifest_f.flush()

    failed = []
    todo = [(train_root / c, c, False) for c in classes
            if not (wanted and c not in wanted)]
    test_dir = root / "test"
    if (test_dir / "ground_truth.csv").exists() and (wanted is None or "__test__" in wanted):
        todo.append((test_dir, "__test__", True))

    for cls_dir, cls_name, is_test in todo:
        try:
            rows = process_class(cls_dir, cls_name, out, a, is_test=is_test)
        except Exception as e:
            failed.append(cls_name)
            print(f"[extract] {cls_name}: FAILED ({type(e).__name__}: {e}) - "
                  f"continuing with the remaining classes; the work already "
                  f"written is kept")
            continue
        all_rows += rows
        flush(rows)

    manifest_f.close()
    if failed:
        print(f"[extract] WARNING: {len(failed)} class(es) failed entirely: "
              f"{failed}. The manifest holds every class that succeeded - "
              f"re-run with --classes {','.join(failed)} --append to add them.")

    train_rows = [r for r in all_rows if r["split"] == "train"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    n_frames = sum(len(r["frame_paths"]) for r in all_rows)
    n_events = len({(r["class_name"], r["video_id"], r["event_index"]) for r in train_rows})
    print(f"\n[extract] {len(train_rows)} train examples from {n_events} distinct "
          f"labelled events ({len(train_rows) / max(n_events, 1):.1f}x via crops - "
          f"flip augmentation happens at train time in finetune_ahc_vlm.py, doubling "
          f"this again with no extra extraction cost)")
    print(f"[extract] {len(test_rows)} test examples (no crops - held out)")
    print(f"[extract] {n_frames} frames total -> {a.manifest}")
    print("[extract] class distribution (train examples):")
    counts = defaultdict(int)
    for r in train_rows:
        counts[r["class_name"]] += 1
    for c in sorted(counts, key=counts.get):
        print(f"  {c:36s} {counts[c]}")


if __name__ == "__main__":
    main()
