# Kaggle run sheet — build both datasets, train both models

Everything here runs **on Kaggle**, in order, in one session. Nothing runs
locally (no GPU here). Copy-paste block by block and read the printed output
before moving to the next block — several steps print a check you are
supposed to eyeball, not skip.

Two models come out of this:
- **Pool 1** → aerial object detector (`yolo11s`, detection)
- **Pool 3** → fire/smoke/flood classifier (`yolo11n-cls`, classification)

Pool 2 (VLM text corpus) trains nothing — it feeds few-shot examples into
`pipeline/vlm_judge.PROMPT`. It is at the bottom, optional.

---

## 0. Locate the previously-trained checkpoint

The Phase 5 detector (`yolo11s`, mAP50-95 0.291 @ epoch 25) is attached as a
Kaggle Dataset, not sitting in the working dir — the original session's files
are gone.

```bash
find /kaggle/input -name "*.pt" | head -20
```

Use whatever path this prints wherever `$BASE` appears below. Prefer
`best.pt` over `last.pt`: training early-stopped at epoch 33, and those last
epochs were worse than epoch 25's best.

```bash
BASE=/kaggle/input/aerial-night-yolo11s-daynight/best.pt
ls -la $BASE     # confirm it exists before spending 5 GPU-hours on it
```

Also confirm the raw datasets are attached (Phase D1 in `DATASET_PLAN.md`):
VisDrone, UIT-ADrone, Drone-Anomaly, FloodNet, FASDD_UAV, D-Fire.

---

## 1. Build Pool 1 — the combined aerial dataset

```bash
python train/build_aerial_dataset.py \
    --visdrone datasets/VisDrone \
    --uit-json datasets/UIT-ADrone/train.json \
    --uit-images datasets/UIT-ADrone/images \
    --drone-anomaly datasets/Drone-Anomaly \
    --target-frames 40000 \
    --out datasets/aerial_combined
```

**Check the printed output before continuing:**
- The **class-remap table** — anything mapped to the wrong VisDrone class
  silently corrupts every box of that class. `helicopter`-style classes with
  no VisDrone equivalent should say `DROPPED`, not be forced onto something.
- **Frames grouped into N videos** — if N is 1, or equals the frame count,
  the video-grouping key is wrong and split-by-video isn't happening.
- **Whether stratification is on.** If it prints "no stratify field", the
  sample is plain random and the "no anomaly class starved" guarantee is not
  in force. Pass `--stratify-field <the real field name>` if the json has one.
- A large **"frames missing under ..."** warning means `--uit-images` isn't
  the root that `file_name` is relative to.

Expected: ~9,059 VisDrone + ~40,000 UIT-ADrone = ~49,000 train images,
548 val (VisDrone only, deliberately unchanged).

---

## 2. Train the detector

```bash
python train/train_aerial.py --preset kaggle --name aerial_combined_v2 \
    --data datasets/aerial_combined/data.yaml \
    --base $BASE --epochs 15
```

`--epochs 15`, not the preset's 40: the combined set is ~5.4x more images
than VisDrone alone, so at the preset's measured pace 40 epochs runs ~14h —
longer than a Kaggle session. Starting from an already-converged `--base`
also needs fewer epochs than training from stock weights.

Do **not** add `--resume` here — it means "continue an interrupted run of the
same `--name`" and ignores `--data`/`--base` entirely. If the session dies
mid-run, *then* restart with `--resume` alone and the same `--name`.

Check epoch-1 wall-clock and multiply out before walking away.

---

## 3. Build Pool 3 — fire/smoke/flood

```bash
python train/build_scene_classifier_dataset.py \
    --floodnet datasets/FloodNet \
    --fasdd datasets/FASDD_UAV \
    --dfire datasets/D-Fire \
    --out datasets/scene_hazard
```

**Check:**
- FloodNet should say **"using Track-1 classification folders"**. If it falls
  back to thresholding masks, that path is a guess — verify the pixel values.
- FASDD's fire/smoke counts. If they look inverted, that release numbers its
  classes the other way: re-run with `--fasdd-fire-id 1 --fasdd-smoke-id 0`.
- Per-class `train=`/`val=` counts. A class with `train=0` means that whole
  source landed in val, and a hazard class at 0 means the classifier can
  never learn it.

---

## 4. Train the classifier

```bash
yolo classify train data=datasets/scene_hazard model=yolo11n-cls.pt \
    imgsz=224 epochs=15 batch=32 project=weights name=scene_hazard
```

Small model, small images, few epochs — this is a cheap auxiliary classifier,
not the main detector. ~1-2h. Don't over-invest GPU hours here.

---

## 5. Mandatory A/B — did the bigger dataset actually help?

```bash
python scripts/ab_weights.py \
    --tuned weights/aerial_combined_v2/weights/best.pt \
    --stock $BASE \
    --data VisDrone.yaml
```

Both models are scored on the same untouched VisDrone val split. **If the
combined model doesn't beat 0.291 mAP50-95, keep the old one and say so** —
more data is a hypothesis, not a result. Report both numbers either way.

For the day/night breakdown the governing rules require, build a night-only
val YAML first and pass `--night-data`:

```bash
python scripts/make_night_yaml.py --images datasets/VisDrone/images/val \
    --out VisDroneNight.yaml
python scripts/ab_weights.py --tuned weights/aerial_combined_v2/weights/best.pt \
    --stock $BASE --data VisDrone.yaml --night-data VisDroneNight.yaml
```

---

## 6. Save the weights — Kaggle wipes everything else

```bash
ls -la weights/aerial_combined_v2/weights/best.pt weights/scene_hazard/weights/best.pt
```

Download **both** `best.pt` files before the session ends. Do **not** bother
downloading the built datasets (tens of thousands of images) — steps 1 and 3
rebuild them in minutes next session.

---

## Optional — Pool 2, the VLM text corpus

Trains nothing; produces few-shot examples for the judge prompt.

```bash
python train/build_vlm_text_corpus.py \
    --a2seek datasets/A2Seek/annotations.jsonl \
    --uca datasets/UCA/captions.json \
    --tar datasets/NVIDIA_TAR/tar.jsonl \
    --cuva datasets/CUVA/cuva.json \
    --out train/vlm_text_corpus.jsonl
```

A source reporting **0 usable rows** means its schema didn't match the
expected field names — go look at the file rather than assuming the sample
was just small. Then hand-pick 3-5 of the printed candidates into
`pipeline/vlm_judge.PROMPT`.
