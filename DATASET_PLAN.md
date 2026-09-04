# Dataset Plan — building and training on the combined corpus

How we go from "here are 10 public datasets" to one dataset we actually train
on, on Kaggle, without corrupting the aerial detector's domain or wasting GPU
hours on data that doesn't help. Written, not run — same status as
`ENHANCEMENT_PLAN.md`.

**Core principle, stated once so it isn't relitigated per-dataset:** we do
NOT dump every dataset into one pool. Three separate pools exist because they
train three separate things, and mixing them is how you quietly *hurt*
accuracy instead of helping it (ground-level CCTV images fine-tuned into an
aerial detector introduce domain shift the detector then pays for on real
aerial footage).

---

## The three pools

### Pool 1 — Aerial YOLO detector (domain-pure, no exceptions)

| Source | Take | Count | Split rule |
|---|---|---|---|
| VisDrone | 100% (already running) | 9,059 train / 548 val | unchanged — this is the current Kaggle job |
| UIT-ADrone | ~7% stratified sample | ~15,000 of 206,194 frames | sample whole **videos**, never individual frames, across all 10 anomaly classes so no class is starved |
| Drone-Anomaly | **0% train, 100% eval** | 87,488 frames held out entirely | frame-binary labels only (no bboxes) — useless for detector training, valuable as an independent benchmark |
| Anything ground-level (DoTA, CCD, D-Fire, FloodNet) | **0%** | — | excluded, full stop — different camera angle, would degrade real aerial recall |

**Why 7% of UIT-ADrone, not more:** VisDrone is 9,059 images. Adding 15,000 aerial-but-different-city images at a ~1.6:1 ratio is already a meaningful domain nudge; going higher risks the detector learning UIT-ADrone's 3-roundabout visual signature instead of generalizing. This ratio is a judgment call, not a measured optimum — say so if asked, don't present it as tuned.

**Update:** per explicit ask, `train/build_aerial_dataset.py --target-frames` now defaults to 40,000 (~19%) instead of 15,000 — more data, traded against more of the above overfitting risk. This is still a guess, not a validated number; the mandatory A/B below is what actually settles it, not the frame count.

### Pool 2 — VLM prompt/distillation text corpus (small, curated, not bulk)

| Source | Take | Count | Why this size |
|---|---|---|---|
| A2Seek | ~2,000 curated examples, weighted toward night/twilight anomalous segments | subset of 42,000 keyframes | closest domain (aerial/campus + real night data) — weighted highest of the four |
| UCA | ~500 captions | ~2% of 23,542 | ground CCTV — used only for caption *phrasing* variety, not domain transfer |
| NVIDIA TAR | ~800 entries | ~2% of 44,040 | accident/traffic reasoning-style text (chain-of-thought, QA) |
| CUVA (optional) | ~370 rows | ~5% of 7,430 | cause/consequence phrasing, skip first if time is short |

**This is deliberately small.** It seeds few-shot examples inside
`pipeline/vlm_judge.PROMPT` and/or validates Phase 18 distillation output — it
is NOT a bulk fine-tuning corpus. Over-ingesting ground-CCTV text risks
biasing the judge's phrasing away from aerial framing, which is what it will
actually see at runtime.

### Pool 3 — Fire/smoke/flood mini-classifier (for the P20 scene-scan)

| Source | Take | Count | Why |
|---|---|---|---|
| FloodNet | 100% | 2,343 images | small, all-aerial, all relevant — no reason to subsample |
| FASDD_UAV | ~50% | ~17,500 of ~35,000 (estimate — UAV-only count isn't published separately from FASDD's 122k total, treat as approximate) | aerial fire/smoke, weighted heavy |
| D-Fire | ~20% | ~4,300 of 21,527 | ground-level — capped low, used only for negative/hard-case diversity (fog, glare, night lights) |

### Held out entirely — eval only, never trained on
**Drone-Anomaly** (Pool 1) and **CCD** (Car Crash Dataset) are reserved as
independent benchmarks. Training on data you also evaluate on is how the
1.0-precision/0.95-recall trap from earlier in this conversation happens —
don't repeat it here.

---

## Phase D1 — Download (tonight, in priority order)

Same order as already given, repeated here for completeness:
1. D-Fire (instant, CC0)
2. UCA annotations (git clone, MBs)
3. CCD (10GB, MIT, Drive)
4. FloodNet v1.0 (small, Drive/Dropbox)
5. NVIDIA TAR annotations (59MB, HF, CC BY 4.0)
6. A2Seek split (10.4GB, HF)
7. FASDD_UAV (part of FASDD, ScienceDB)
8. UIT-ADrone (Drive, ~40-60GB raw — this is the biggest pull, start it first if bandwidth is the bottleneck, not last)

**Do this on Kaggle, not locally** — Kaggle's network is faster and this repo has no local GPU/storage budget for it anyway.

## Phase D2 — Build Pool 1 (aerial detector)

**New file:** `train/build_aerial_dataset.py`

1. Load VisDrone's existing YOLO layout unchanged (`datasets/VisDrone/`).
2. From UIT-ADrone's `train.json`/`test.json` (COCO bbox format), group entries by source video, shuffle **videos** (not frames) with a fixed seed, take videos until the cumulative frame count reaches ~15,000, convert COCO boxes → YOLO `.txt` labels.
3. Remap UIT-ADrone's class ids onto VisDrone's 10-class map where the object type matches (car, motorcycle, pedestrian, etc.); classes with no VisDrone equivalent are dropped, not force-mapped — a wrong mapping silently corrupts every box of that class.
4. Write a merged `datasets/aerial_combined/` with `data.yaml` pointing at both sources' image dirs, one combined `train.txt`/`val.txt`.
5. Drone-Anomaly: write a separate `datasets/drone_anomaly_eval/` manifest — **never referenced by `train_aerial.py`**, only by a new eval step (below).

**Done when:** `datasets/aerial_combined/data.yaml` exists, `yolo val` can load it, and the class-remap table is printed and eyeballed for anything mapped wrong.

## Phase D3 — Build Pool 3 (fire/smoke/flood classifier)

**New file:** `train/build_scene_classifier_dataset.py`

1. Normalize FloodNet's segmentation masks down to a per-image binary label (`flooded` / `not_flooded`) if training a classifier rather than a segmenter — cheaper and matches the P20 scene-scan's grid-cell classification pattern already in `pipeline/zone_classify.py`.
2. Sample FASDD_UAV and D-Fire per the percentages above, **grouped by source folder/scene, not by individual image**, to avoid near-duplicate frames leaking across train/val.
3. Merge into one YOLO-classification layout: `datasets/scene_hazard/{fire,smoke,flood,normal}/`.

**Done when:** class counts are printed and are not wildly imbalanced (if `normal` ends up 10x any hazard class, downsample `normal`, don't upsample the hazard classes synthetically).

## Phase D4 — Build Pool 2 (VLM text corpus)

**New file:** `train/build_vlm_text_corpus.py`

1. Pull the specified sample counts from each source's caption/annotation files.
2. Normalize into one schema: `{source, video_ref, t_start, t_end, caption, reasoning}`.
3. Write `train/vlm_text_corpus.jsonl`.
4. Manually pick 3-5 of the clearest, most aerial-relevant entries and paste them into `pipeline/vlm_judge.PROMPT` as few-shot examples — this is the cheapest, lowest-risk use of this pool and should happen regardless of whether Phase 18 distillation is attempted.

**Done when:** the jsonl exists and the few-shot examples are in the prompt.

## Phase D5 — Train on Kaggle

**Pool 1 (aerial detector) — optional v2, only with GPU budget to spare:**
```
python train/train_aerial.py --preset kaggle --name aerial_combined_v2 \
    --data datasets/aerial_combined/data.yaml \
    --base weights/aerial_night/weights/best.pt --epochs 15
```
`--resume` is NOT used here — in `train_aerial.py` it means "continue an
interrupted run of the same `--name`" and silently ignores `--data`/`--base`
entirely, which would break this two-stage combined run. `--base` alone
already does the two-stage load (a converged checkpoint as the starting
point, lr0 dropped 10x). `--epochs 15` instead of the preset's 40: the
combined set is ~5.4x more images than VisDrone alone at the same batch
size, so epoch wall-clock scales accordingly (~2h measured for 33 epochs on
VisDrone alone → ~14h projected for 40 epochs on the combined set, likely
longer than one Kaggle GPU session) - fewer epochs is also appropriate
because this run starts from an already-converged base, not stock weights.
If a session is interrupted mid-run, `--resume` (with no other flags, same
`--name`) picks the SAME run back up from `last.pt`.

Then **mandatory A/B against the currently-training VisDrone-only run** (same as Phase 6's existing rule) — on VisDrone val AND on the held-out Drone-Anomaly eval set. If the combined version doesn't beat the VisDrone-only version, keep the simpler one and say so.

**Pool 3 (fire/smoke/flood classifier) — new, small, ~1-2h GPU:**
```
yolo classify train data=datasets/scene_hazard model=yolo11n-cls.pt \
    imgsz=224 epochs=15 batch=32
```
Small model, small images, few epochs — this is a cheap auxiliary classifier feeding the P20 scene-scan, not the main detector. Don't over-invest GPU hours here.

**Pool 2 (VLM text corpus)** — not trained directly unless Phase 18 is pursued; used as-is for prompt few-shot and as validation text for the distillation A/B.

## Time budget
| Phase | Estimate |
|---|---|
| D1 downloads | 2-4h (parallelize, start UIT-ADrone first — it's the biggest) |
| D2 build Pool 1 | ~1h |
| D3 build Pool 3 | ~45min |
| D4 build Pool 2 | ~30min |
| D5 Pool 3 training | ~1-2h GPU |
| D5 Pool 1 v2 training (optional) | ~5-6h GPU — only if the current run finishes early and quota remains |

**If GPU quota or time runs out: skip Pool 1 v2 entirely.** The currently-running VisDrone-only fine-tune already satisfies "a domain-general aerial detector" — Pool 1 v2 is an accuracy improvement attempt, not a requirement. Pool 3 (fire/smoke/flood) is the higher-value use of remaining GPU hours because it closes an actual coverage gap (Phase 15/P20's finding) that the current model has zero ability to detect at all.

## Governing rules carried over from CLAUDE.md
- Split by video/source, never by frame — every pool above follows this.
- Report results on the held-out sets (Drone-Anomaly, CCD), never on data trained on.
- A `None`/refused result beats a fabricated number — if a class has too few samples after sampling, drop it and say so rather than force a percentage that doesn't exist.
