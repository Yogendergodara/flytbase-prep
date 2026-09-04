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
    """Duplicate examples of under-represented classes up to min_per_class.

    This reweights the loss; it does NOT create new information. A class
    with 14 source videos duplicated to 150 examples is still 14 videos, now
    seen ~11x per epoch - which risks memorising those clips. The counts
    printed below therefore distinguish EXAMPLES (crops, what training sees)
    from distinct VIDEOS (what the class actually knows about), because an
    earlier version reported crop counts while calling them "events" and
    made starved classes look ~3x healthier than they were.
    """
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class_name"]].append(r)
    rng = random.Random(seed)
    out = []
    for cls, items in sorted(by_class.items()):
        n = len(items)
        n_videos = len({r["video_id"] for r in items})
        factor = min(max_oversample, math.ceil(min_per_class / n)) if n < min_per_class else 1
        dup = list(items)
        for _ in range(factor - 1):
            dup += items
        rng.shuffle(dup)
        if factor > 1:
            reached = n * factor
            note = "" if reached >= min_per_class else f", still short of {min_per_class}"
            print(f"[finetune] {cls}: {n} examples from {n_videos} distinct videos "
                  f"x{factor} -> {reached}{note} (duplication, not new data)")
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


def _bf16_supported():
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def cap_pixels(im, max_pixels):
    """Downscale so W*H <= max_pixels, preserving aspect ratio.

    This is not cosmetic - it is the difference between training on the
    examples you think you are and training on truncated ones. Qwen2.5-VL
    spends roughly (W/28)*(H/28) visual tokens per image, so the 1280x720
    frames extract_ahc_frames.py writes cost ~1,175 tokens EACH; eight of
    them is ~9,400 tokens, which overflows --max-seq-length 8192 before a
    single word of text. The overflow is silent: frames (and potentially
    the assistant target itself) fall off the end of the sequence and the
    model learns from a partial example.

    max_pixels defaults to config.yaml's vlm.max_pixels (401408 = 512*28*28),
    so an image costs at most ~512 tokens and eight fit in ~4,096 - the same
    budget pipeline/vlm_judge.py already uses at inference, which also keeps
    train-time and inference-time framing consistent.
    """
    w, h = im.size
    if w * h <= max_pixels:
        return im
    scale = (max_pixels / (w * h)) ** 0.5
    return im.resize((max(1, int(w * scale)), max(1, int(h * scale))))


def to_record(row, frames_root, max_pixels=401408, flip=False):
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
    images = [cap_pixels(Image.open(p).convert("RGB"), max_pixels)
              for p in resolve_paths(row, frames_root)]
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

    def __init__(self, rows, frames_root, flip_augment, max_pixels=401408):
        self.rows = rows
        self.frames_root = frames_root
        self.flip_augment = flip_augment
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.rows) * (2 if self.flip_augment else 1)

    # Mirroring is label-preserving for 10 of the 12 classes, but NOT for
    # wrong_way_driving: this is left-hand-traffic footage, so a mirror turns
    # every frame into right-hand traffic. The cue a fixed camera actually
    # offers ("vehicle on the wrong side relative to the rest of the flow")
    # is inverted while the label is kept, i.e. contradictory evidence under
    # one label. Those rows are served unflipped instead of being dropped,
    # so the class keeps its (already thin) example count.
    NO_FLIP_CLASSES = {"wrong_way_driving"}

    def __getitem__(self, idx):
        if idx >= len(self.rows):        # second half of the index space = flipped views
            row = self.rows[idx - len(self.rows)]
            flip = row["class_name"] not in self.NO_FLIP_CLASSES
            return to_record(row, self.frames_root, self.max_pixels, flip=flip)
        return to_record(self.rows[idx], self.frames_root, self.max_pixels, flip=False)


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
    ap.add_argument("--min-per-class", type=int, default=150,
                    help="oversample classes with fewer than this many examples. "
                         "Raised from 60 because at 60 only ONE class "
                         "(stalled_or_broken_down_vehicle, 38 examples) qualified, "
                         "leaving a ~10:1 imbalance against traffic_accident (733) "
                         "- which biases a generative classifier toward the frequent "
                         "classes. 150 lifts the thin classes without pushing "
                         "duplication past --max-oversample.")
    ap.add_argument("--max-oversample", type=int, default=5,
                    help="cap on duplication factor - past this, a starved class "
                         "just isn't fixable by reweighting and should be reported as such")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flip-augment", dest="flip_augment", action="store_true", default=True,
                    help="double TRAIN examples with an in-memory horizontal flip "
                         "(free - no extra extraction cost, see to_record's docstring). "
                         "Never applied to val or test.")
    ap.add_argument("--no-flip-augment", dest="flip_augment", action="store_false")
    ap.add_argument("--max-pixels", type=int, default=401408,
                    help="per-image pixel cap (401408 = 512*28*28 = ~512 Qwen "
                         "visual tokens), matching config.yaml's vlm.max_pixels so "
                         "training and inference see comparably-scaled frames. "
                         "Raising this past ~800k with 8 frames/example will "
                         "overflow --max-seq-length and SILENTLY truncate images "
                         "off the end of examples - see cap_pixels' docstring.")
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
    train_records = LazyAHCDataset(train_split, a.frames_root, a.flip_augment, a.max_pixels)
    val_records = LazyAHCDataset(val_split, a.frames_root, False, a.max_pixels)
    n_frames = len(train_split[0]["frame_paths"]) if train_split else 0
    print(f"[finetune] {len(train_records)} train examples"
          f"{' (flip-augmented 2x)' if a.flip_augment else ''}, "
          f"{len(val_records)} val examples - decoded lazily per batch")
    print(f"[finetune] {n_frames} frames/example, capped at {a.max_pixels} px "
          f"(~{a.max_pixels // 784} visual tokens each, ~{n_frames * a.max_pixels // 784} "
          f"for the images) against --max-seq-length {a.max_seq_length}")
    if n_frames * (a.max_pixels // 784) > a.max_seq_length * 0.8:
        print(f"[finetune] WARNING: images alone may consume most of the sequence "
              f"budget - examples risk truncation. Lower --max-pixels or "
              f"re-extract with fewer --frames-per-crop.")

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

    # train_on_responses_only comes from the hackathon primer's own tip: loss
    # is then computed on the assistant answer only, not on the (identical
    # for every example) prompt. That matters for accuracy - without it a
    # large share of the loss is spent re-predicting boilerplate.
    #
    # This Unsloth version accepts the flag but then ASSERTS that
    # instruction_part/response_part are also given (unsloth_zoo
    # vision_utils.py: `assert(isinstance(instruction_part, str) and ...)`),
    # so passing the flag alone raised AssertionError - after 100 minutes of
    # extraction had already been spent. Pass Qwen's ChatML turn markers
    # explicitly, and catch AssertionError as well as TypeError so a
    # differently-shaped API degrades to plain collation instead of throwing
    # away the whole run.
    try:
        collator = UnslothVisionDataCollator(
            model, tokenizer,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("[finetune] collator: train_on_responses_only=True (loss on answer only)")
    except (TypeError, AssertionError) as e:
        collator = UnslothVisionDataCollator(model, tokenizer)
        print(f"[finetune] collator: response-only masking unavailable on this "
              f"Unsloth build ({type(e).__name__}) - loss also covers prompt "
              f"tokens. Training still works; expect slightly worse convergence.")

    # skip_prepare_dataset is REQUIRED for a pre-formatted vision dataset and
    # is part of Unsloth's own documented vision recipe. Without it TRL runs
    # its text-SFT preprocessing over the dataset, which calls .map() -
    # LazyAHCDataset deliberately has no .map (it is a lazy __getitem__
    # dataset, because materialising every record OOM-killed a session), so
    # that path would AttributeError at startup, after the 75-minute
    # extraction had already been paid for.
    cfg_kwargs = dict(
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=a.epochs, learning_rate=a.lr,
        warmup_ratio=0.05, logging_steps=10, eval_strategy="epoch",
        output_dir=a.out, save_strategy="epoch", report_to="none",
        remove_unused_columns=False, dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        # A T4 has no bf16, so fp16 + GradScaler is what keeps LoRA
        # gradients from underflowing against an fp16 base. Unsloth's own
        # recipe sets these explicitly rather than trusting a default, and
        # the failure mode if they are wrong is silent non-convergence -
        # a run that completes and produces a weak adapter.
        fp16=not _bf16_supported(), bf16=_bf16_supported(),
        seed=a.seed,
    )
    # TRL renamed max_seq_length -> max_length at some point, and Unsloth
    # patches SFTConfig on top of whatever TRL is installed. Try the name
    # this stack documents, then the other, then neither - a sequence-length
    # default is survivable, losing the run to a rejected kwarg is not.
    for extra in ({"max_seq_length": a.max_seq_length},
                  {"max_length": a.max_seq_length},
                  {}):
        try:
            sft_config = SFTConfig(**cfg_kwargs, **extra)
            if extra:
                print(f"[finetune] SFTConfig accepted {list(extra)[0]}="
                      f"{a.max_seq_length}")
            else:
                print(f"[finetune] WARNING: SFTConfig rejected both "
                      f"max_seq_length and max_length - using this TRL's "
                      f"default sequence length, which may truncate examples.")
            break
        except TypeError as e:
            last_err = e
    else:
        raise last_err

    # Trainer's `tokenizer=` kwarg was deprecated in transformers 4.46 and
    # REMOVED in 5.0 (this stack runs 5.5.0); the replacement is
    # `processing_class=`. Unsloth's patch layer has historically rewritten
    # the old name, and its own docs still show `tokenizer=`, so which one
    # works depends on whether that patch still covers TRL's current
    # signature. Try the modern name first, fall back to the legacy one.
    for kw in ("processing_class", "tokenizer"):
        try:
            trainer = SFTTrainer(
                model=model, data_collator=collator,
                train_dataset=train_records, eval_dataset=val_records,
                args=sft_config, **{kw: tokenizer},
            )
            print(f"[finetune] SFTTrainer accepted {kw}=")
            break
        except TypeError as e:
            last_err = e
    else:
        raise last_err

    trainer.train()

    model.save_pretrained(a.out)
    tokenizer.save_pretrained(a.out)
    print(f"[finetune] LoRA adapter saved to {a.out}")
    print(f"[finetune] MANDATORY next step: python train/eval_ahc_vlm.py "
          f"--adapter {a.out} --manifest {a.manifest} - scores against the "
          f"REAL held-out test split, never seen during this training run.")


if __name__ == "__main__":
    main()
