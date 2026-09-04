"""Step 2: LoRA fine-tune a VLM on the real AHC hackathon labels (12-class
video anomaly classification), using train/extract_ahc_frames.py's manifest.

This is a DIFFERENT task from train/distill_vlm.py (which distills the
existing geometric-cascade judge's prompt onto teacher pseudo-labels).
Here the input is a raw labelled clip, the target is one of the 12 official
classes + a description - the actual hackathon scoring task. Do not point
this at the same output dir as distill_vlm.py; they are different adapters
for different jobs.

Recipe follows the hackathon primer's own Unsloth example exactly (frozen
vision encoder, LoRA on language layers only, r=16/alpha=16, all-linear
target modules, UnslothVisionDataCollator with train_on_responses_only,
dataset built as a list - NOT dataset.map(), which the primer says breaks on
multi-image samples). Kaggle-only, GPU required - never run locally
(CLAUDE.md: no local GPU).

    python train/finetune_ahc_vlm.py --manifest train/ahc_manifest.jsonl \\
        --base Qwen/Qwen2.5-VL-3B-Instruct --out weights/qwen_ahc_lora --epochs 3

Class balance in the real (downloaded) data is uneven - confirmed by
train/consolidate_ahc_dataset.py's own output, not assumed:
stalled_or_broken_down_vehicle has 14 source videos vs. hundreds for others.
--min-per-class oversamples (duplicates existing extracted frames, capped at
--max-oversample x) rather than silently training on whatever ratio the raw
counts happen to produce. This does not create new information for the
starved classes - say so on the slide, don't present it as a fix for having
too few stalled-vehicle videos.
"""
import argparse
import json
import math
import random
from collections import defaultdict

AHC_CLASSES = [
    "normal", "traffic_accident", "traffic_congestion",
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood",
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence",
]

AHC_PROMPT = """You are reviewing a short sequence of frames sampled from a fixed \
CCTV, dashcam, or drone camera feed.

Classify what is happening into EXACTLY one of these categories:
{classes}

Reply with ONLY this JSON:
{{"class_name": "<one category above, exact string>", "is_anomaly": true|false, \
"description": "<one sentence describing what you see>"}}"""


def load_manifest(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        rows.append(json.loads(line))
    return rows


def video_level_split(train_rows, val_fraction, seed):
    """Split by video_id, not by event - a video with two events must not
    have one event in train and the other in val (DATASET_PLAN.md's
    split-by-video rule, applied here for the same leakage reason)."""
    videos = sorted({r["video_id"] for r in train_rows})
    rng = random.Random(seed)
    rng.shuffle(videos)
    n_val = max(1, int(len(videos) * val_fraction))
    val_videos = set(videos[:n_val])
    train = [r for r in train_rows if r["video_id"] not in val_videos]
    val = [r for r in train_rows if r["video_id"] in val_videos]
    return train, val


def oversample(rows, min_per_class, max_oversample, seed):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class_name"]].append(r)
    rng = random.Random(seed)
    out = []
    for cls, items in by_class.items():
        n = len(items)
        factor = min(max_oversample, math.ceil(min_per_class / n)) if n < min_per_class else 1
        dup = list(items)
        for _ in range(factor - 1):
            dup += items
        rng.shuffle(dup)
        if factor > 1:
            reached = n * factor
            note = "" if reached >= min_per_class else f" (still short of {min_per_class})"
            print(f"[finetune] {cls}: {n} real events x{factor} -> {reached}{note}")
        out += dup
    rng.shuffle(out)
    return out


def resolve_paths(row, frames_root):
    """Manifest frame_paths are relative to wherever AHC_frames landed (the
    manifest and the frames travel to Kaggle separately). An already-absolute
    path (an older manifest) is left alone rather than double-joined."""
    from pathlib import Path
    return [p if Path(p).is_absolute() else str(Path(frames_root) / p)
            for p in row["frame_paths"]]


def to_record(row, frames_root, flip=False):
    """Decode one training example's frames and build its chat record.

    flip=True mirrors the frames in memory - no extra extraction, no extra
    disk files (extract_ahc_frames.py measured writing each crop's frames to
    disk at ~21 files/sec; materialising a flipped copy on disk would have
    roughly doubled a ~30min extraction for a transform that is free here).
    A mirrored collision is still a collision for all 12 classes; the one
    caveat is a description naming a direction ("moving left to right")
    would contradict the mirrored image - the AHC descriptions observed are
    not directional, but that risk is real, not fully eliminated.

    Called PER ITEM by LazyAHCDataset, never eagerly over the whole split -
    see that class's docstring for why that distinction is load-bearing.
    """
    from PIL import Image, ImageOps
    images = [Image.open(p).convert("RGB") for p in resolve_paths(row, frames_root)]
    if flip:
        images = [ImageOps.mirror(im) for im in images]
    target = json.dumps({
        "class_name": row["class_name"],
        "is_anomaly": row["is_anomaly"],
        "description": row["description_summary"],
    })
    prompt_text = AHC_PROMPT.format(classes=", ".join(AHC_CLASSES))
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt_text})
    return {
        "images": images,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
    }


class LazyAHCDataset:
    """Decodes frames on __getitem__ instead of up front.

    This exists because the eager version - `[to_record(r) for r in split]` -
    OOM-killed an entire Kaggle session. Each record holds 8 decoded PIL
    images; over ~8,400 flip-augmented records that is ~67,000 decoded
    images resident at once, tens of GB. Nothing about the model or the GPU
    was the problem: the dataset build alone exhausted system RAM before
    training could start.

    Implements the torch Dataset protocol (__len__/__getitem__), which
    transformers' Trainer accepts directly, so peak memory is
    batch_size x frames_per_example instead of the whole corpus. `flip`
    doubling is expressed as an index offset rather than a second
    materialised list, for the same reason.
    """

    def __init__(self, rows, frames_root, flip_augment):
        self.rows = rows
        self.frames_root = frames_root
        self.flip_augment = flip_augment

    def __len__(self):
        return len(self.rows) * (2 if self.flip_augment else 1)

    def __getitem__(self, idx):
        if idx >= len(self.rows):        # second half of the index space = flipped views
            return to_record(self.rows[idx - len(self.rows)], self.frames_root, flip=True)
        return to_record(self.rows[idx], self.frames_root, flip=False)


def drop_missing_frames(rows, frames_root, label):
    """Drop rows whose frame files are not on disk, and say how many.

    A single missing frame used to raise FileNotFoundError mid-run - after
    the model was loaded and training had begun - throwing away the whole
    job over one bad row. That happened for real: an interrupted extraction
    left a manifest referencing frames it never finished writing. Checking
    up front costs one stat() per frame and converts a hard crash into a
    counted, reported skip.
    """
    import os
    keep, dropped = [], 0
    for r in rows:
        if all(os.path.exists(p) for p in resolve_paths(r, frames_root)):
            keep.append(r)
        else:
            dropped += 1
    if dropped:
        print(f"[finetune] {label}: dropped {dropped} of {len(rows)} rows - frame "
              f"files missing on disk (incomplete extraction?). Re-run "
              f"extract_ahc_frames.py if this number is large.")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="train/ahc_manifest.jsonl")
    ap.add_argument("--frames-root", default="datasets/AHC_frames",
                    help="where the AHC_frames folder actually landed on THIS "
                         "machine/Kaggle session - the manifest's frame_paths "
                         "are relative to this, not absolute, since the "
                         "manifest and the frames travel separately")
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="matches config.yaml's vlm.model_id, so this adapter "
                         "can later attach to the same base pipeline/vlm_judge.py loads")
    ap.add_argument("--out", default="weights/qwen_ahc_lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16, help="LoRA rank (primer's example: 16)")
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--max-seq-length", type=int, default=8192,
                     help="raised from 4096 now that events carry 20 frames "
                          "each instead of 8 - each image costs a non-trivial "
                          "chunk of the sequence, so more frames/event needs "
                          "more headroom here or training silently truncates "
                          "images off the end of long examples. If this OOMs "
                          "on a T4/P100, lower --frames-per-event in "
                          "extract_ahc_frames.py and re-extract rather than "
                          "just lowering this - a truncated example teaches "
                          "the model to answer without seeing all its images.")
    ap.add_argument("--min-per-class", type=int, default=60,
                    help="oversample classes below this event count")
    ap.add_argument("--max-oversample", type=int, default=5,
                    help="cap on duplication factor - past this, a starved class "
                         "just isn't fixable by reweighting and should be reported as such")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flip-augment", dest="flip_augment", action="store_true", default=True,
                    help="double TRAIN examples with an in-memory horizontal flip "
                         "(free - no extra extraction cost, see to_record's docstring). "
                         "Never applied to val or test.")
    ap.add_argument("--no-flip-augment", dest="flip_augment", action="store_false")
    a = ap.parse_args()

    rows = load_manifest(a.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"[finetune] {len(train_rows)} train events, {len(test_rows)} held-out "
          f"test events (test is NEVER used here - eval_ahc_vlm.py handles that)")

    # validate before loading a 3B model, not after - see drop_missing_frames
    train_rows = drop_missing_frames(train_rows, a.frames_root, "train")
    if not train_rows:
        raise SystemExit(
            f"[finetune] every train row's frames are missing under "
            f"--frames-root {a.frames_root}. Either that path is wrong, or "
            f"extract_ahc_frames.py never finished - re-run it before this.")

    train_split, val_split = video_level_split(train_rows, a.val_fraction, a.seed)
    print(f"[finetune] video-level split: {len(train_split)} train events, "
          f"{len(val_split)} val events")

    train_split = oversample(train_split, a.min_per_class, a.max_oversample, a.seed)
    print(f"[finetune] after oversampling: {len(train_split)} train events")

    # lazy, not eager: the eager version OOM-killed a whole Kaggle session
    train_records = LazyAHCDataset(train_split, a.frames_root, a.flip_augment)
    val_records = LazyAHCDataset(val_split, a.frames_root, flip_augment=False)
    print(f"[finetune] {len(train_records)} train examples"
          f"{' (flip-augmented 2x)' if a.flip_augment else ''}, "
          f"{len(val_records)} val examples - decoded lazily per batch")

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastVisionModel.from_pretrained(
        a.base, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, r=a.r, lora_alpha=a.r,
        finetune_vision_layers=False,      # frozen encoder - primer's recipe
        finetune_language_layers=True,
        target_modules="all-linear",
        lora_dropout=0.0,
    )
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer, train_on_responses_only=True),
        train_dataset=train_records, eval_dataset=val_records,
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            num_train_epochs=a.epochs, learning_rate=a.lr,
            warmup_ratio=0.05, logging_steps=10, eval_strategy="epoch",
            output_dir=a.out, save_strategy="epoch", report_to="none",
            remove_unused_columns=False, dataset_text_field="",
            max_seq_length=a.max_seq_length,
        ),
    )
    trainer.train()

    model.save_pretrained(a.out)
    tokenizer.save_pretrained(a.out)
    print(f"[finetune] LoRA adapter saved to {a.out}")
    print(f"[finetune] MANDATORY next step: python train/eval_ahc_vlm.py "
          f"--adapter {a.out} --manifest {a.manifest} - scores against the "
          f"REAL held-out test split, never seen during this training run.")


if __name__ == "__main__":
    main()
