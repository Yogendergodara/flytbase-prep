# Train all 3 models on Kaggle — run these cells in order

Every cell is copy-paste ready. `set -e` in each means a failure stops that
cell immediately instead of cascading into confusing downstream errors.

## Why this runs SEQUENTIALLY, not in parallel

Running Model 2 and Model 3 at the same time crashed a whole Kaggle session
("tried to allocate more memory than is available"). Two causes, both now
fixed in the code, but the sequencing advice stands:

- **The real bug (fixed):** `finetune_ahc_vlm.py` built every training
  record eagerly, each holding 8 decoded PIL images - ~8,400 records =
  ~15 GB of RAM before training even started. It now decodes lazily per
  batch (`LazyAHCDataset`), measured at ~1.75 MB/record instead.
- **Still true:** Model 2 takes ~15 min, Model 3 takes hours. Running them
  together saves ~15 minutes off a multi-hour job while stacking two
  models' memory in one session. Not worth the risk.

**Where both GPUs genuinely help: Model 1** (a 5-6h job) - that cell uses
`device=0,1`. For the short jobs, one GPU each, one at a time.

---

## Cell 1 — setup

```bash
%%bash
set -e
cd /kaggle/working
if [ ! -d flytbase-prep ]; then
  git clone https://github.com/Yogendergodara/flytbase-prep.git
fi
cd flytbase-prep
git fetch origin
git checkout enhancements
git pull
pip install -q ultralytics huggingface_hub unsloth
echo "=== GPUs ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| count:', torch.cuda.device_count())"
echo "=== attached inputs ==="
ls /kaggle/input/
```

**Required attachments** (Add Input → Datasets / Models):
- `ahc-frames` — the raw AHC videos (Model 3)
- `aerial-imagery-dataset-floodnet-challenge` — FloodNet (Model 2)
- `d-fire` — D-Fire (Model 2)

Check Cell 1's `ls /kaggle/input/` output matches the paths used below. If a
folder name differs, fix the paths in Cell 2/4 rather than guessing.

---

## Cell 2 — Model 2: build the hazard dataset

```bash
%%bash
set -e
mkdir -p /kaggle/working/datasets
ln -sfn "/kaggle/input/aerial-imagery-dataset-floodnet-challenge/FloodNet Challenge - Track 1/Train/Labeled" /kaggle/working/datasets/FloodNet
ln -sfn "/kaggle/input/d-fire" /kaggle/working/datasets/D-Fire

cd /kaggle/working/flytbase-prep
python train/build_scene_classifier_dataset.py \
    --floodnet /kaggle/working/datasets/FloodNet \
    --dfire /kaggle/working/datasets/D-Fire \
    --out /kaggle/working/datasets/scene_hazard
```

If FloodNet/D-Fire were attached via `kagglehub` rather than the UI, their
paths nest differently (`/kaggle/input/datasets/<owner>/<slug>/...`) - use
whatever Cell 1 printed.

**Read the printed class counts.** `flood` is expected to be small (~51
images in this FloodNet copy vs ~1,200 each for fire/smoke) - that class
will be measurably weaker, and that is a data limit, not a bug.

---

## Cell 3 — Model 2: train (~15-20 min)

```bash
%%bash
set -e
cd /kaggle/working/flytbase-prep
yolo classify train data=/kaggle/working/datasets/scene_hazard model=yolo11n-cls.pt \
    imgsz=224 epochs=15 batch=32 project=weights name=scene_hazard device=0
echo "DONE -> weights/scene_hazard/weights/best.pt"
```

No `cache=ram` (it added memory pressure for no real gain on a dataset this
small) and one GPU (this model used 565 MiB of 15 GB - a second GPU would
sit idle). Expect a warning about one corrupt D-Fire image; it is skipped
and training continues.

---

## Cell 4 — Model 3: extract frames (~30-40 min, CPU only)

```bash
%%bash
set -e
cd /kaggle/working/flytbase-prep
python train/extract_ahc_frames.py \
    --root /kaggle/input/ahc-frames/AHC_full \
    --out /kaggle/working/datasets/AHC_frames \
    --frames-per-crop 8 --crops-per-event 3 \
    --manifest train/ahc_manifest.jsonl
```

This clears any previous extraction first, so the frames and the manifest
always describe the same run - an interrupted earlier run left a manifest
pointing at frames it never wrote, which killed a fine-tune mid-job.

**Let this finish. Do not interrupt it or run other cells against
`AHC_frames` while it works.** Expect ~4,200 train + 32 test examples.

---

## Cell 5 — Model 3: fine-tune Qwen (~3-6h)

```bash
%%bash
set -e
cd /kaggle/working/flytbase-prep
python train/finetune_ahc_vlm.py \
    --manifest train/ahc_manifest.jsonl \
    --frames-root /kaggle/working/datasets/AHC_frames \
    --base Qwen/Qwen2.5-VL-3B-Instruct \
    --out weights/qwen_ahc_lora --epochs 3
```

Frames are decoded per batch now, so this no longer OOMs the session. It
also validates every manifest row against disk BEFORE loading the model,
and reports any it drops - so a bad row costs a warning, not a crash three
hours in.

If it OOMs on **GPU** (not system RAM), reduce frames per example and
re-extract - `--frames-per-crop 4` in Cell 4 - rather than lowering
`--max-seq-length`, which would silently truncate images off the end of
examples and teach the model to answer without seeing them.

---

## Cell 6 — Model 3: evaluate (mandatory)

```bash
%%bash
set -e
cd /kaggle/working/flytbase-prep
python train/eval_ahc_vlm.py \
    --manifest train/ahc_manifest.jsonl \
    --frames-root /kaggle/working/datasets/AHC_frames \
    --base Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter weights/qwen_ahc_lora
```

Scores the adapter against the 32 held-out test events AND against
zero-shot prompting of the same base model. **If the fine-tune doesn't beat
zero-shot, report both numbers and keep the simpler option** - same rule as
every other A/B in this repo.

---

## Cell 7 — Model 1: aerial detector (OPTIONAL, ~5-6h, both GPUs)

Only run this if you want to retrain. You already have a trained checkpoint
(Phase 5: P 0.598, R 0.464, mAP50-95 0.291), uploaded as the `daynight`
Kaggle Model. Re-running the same VisDrone-only recipe will land at roughly
the same numbers - it early-stopped at epoch 33/40, meaning it had already
converged.

```bash
%%bash
set -e
cd /kaggle/working/flytbase-prep
yolo settings datasets_dir=/kaggle/working/datasets
python train/train_aerial.py --preset kaggle --name aerial_v2 --device 0,1
```

To actually improve on 0.291, change one of these - not just re-run:
- **more data** (UIT-ADrone combined, via `build_aerial_dataset.py`) - still
  blocked, no verified download source found
- **bigger model** - `yolo11m`/`yolo11l` instead of `yolo11s` (edit
  `PRESETS` in `train_aerial.py`)
- **higher resolution** - above 1024px; VisDrone's difficulty is tiny
  objects, and resolution is the lever that most affects small-object recall

---

## Cell 8 — download everything before the session ends

```bash
%%bash
ls -la weights/scene_hazard/weights/best.pt 2>/dev/null || echo "Model 2 missing"
ls -la weights/qwen_ahc_lora 2>/dev/null || echo "Model 3 missing"
ls -la out/ahc_eval.json 2>/dev/null || echo "eval missing"
```

`/kaggle/working` is wiped when the session ends. Download:
- `weights/scene_hazard/weights/best.pt` (Model 2)
- `weights/qwen_ahc_lora/` — whole folder (Model 3 adapter + tokenizer)
- `out/ahc_eval.json` (the numbers you report)

---

## Expected end state

| Model | Output | Status after these cells |
|---|---|---|
| 1. Aerial detector | `daynight` Kaggle Model (already trained) | mAP50-95 0.291 — retrain only via Cell 7 |
| 2. Hazard classifier | `weights/scene_hazard/weights/best.pt` | trained; `flood` class weak by data limit |
| 3. AHC VLM | `weights/qwen_ahc_lora/` | trained; numbers in `out/ahc_eval.json` |

**Honest expectation on accuracy:** 90%+ across all 12 AHC classes is
unlikely — only 48% of the official videos were ever downloaded (1,668
missing, see `train/audit_ahc_coverage.py`), and some classes have very few
real examples. Binary `is_anomaly` and the well-represented classes (fire,
smoke, flood, fighting) are where strong numbers are realistic. Report
per-class results from Cell 6 rather than a single headline number.
