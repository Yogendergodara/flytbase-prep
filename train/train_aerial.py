"""Fine-tune the AERIAL detector - the one thing worth training before the event.

Viewpoint and scale transfer across cities; 'normal' does not. So this trains
top-down small-object detection (VisDrone) plus synthetic night, and nothing else.

    python train/make_night.py --images .../images/train --labels .../labels/train
    python train/train_aerial.py --preset laptop      # ~3-4 h, 6-8 GB
    python train/train_aerial.py --preset kaggle      # ~4-5 h on P100/T4

Launch it in the BACKGROUND at hour one and build the pipeline while it runs.

Two-stage by default: someone already paid for the aerial adaptation, so a
preset's PRETRAINED entry - a VisDrone-pretrained HF checkpoint matching that
preset's model size - is tried BEFORE the stock COCO weight named in PRESETS.
Only the night gap (40% synthetic dark twins) then has to be learned here,
for the same wall-clock budget. `--base` overrides this explicitly; `--no-pretrained`
forces stock COCO. A checkpoint that fails to download or fails to load falls
back to stock rather than aborting the night's only training run - training
on the wrong base is recoverable (A/B catches it); not training at all is not.
"""
import argparse
import sys
from pathlib import Path

PRESETS = {
    # name         model         imgsz epochs batch freeze   ~wall
    "overnight": ("yolo11s.pt",  1280, 60,    8,    0),    # 8-10 h  <- THE one
    "laptop":    ("yolo11n.pt",  960,  25,    8,    10),   # 3-4 h
    "kaggle":    ("yolo11s.pt",  1024, 40,    12,   0),    # 5-6 h, free P100/T4
    "fast":      ("yolo11n.pt",  768,  12,    16,   10),   # 1-1.5 h, out of time
}

# VisDrone-pretrained checkpoints on HF Hub, keyed by the stock weight name
# each preset would otherwise use. Unverified beyond "the repo exists with
# this name" - check the licence and A/B it (ab_weights.py) before trusting
# it over a from-scratch run.
PRETRAINED = {
    "yolo11s.pt": ("dronefreak/visdrone-yolov11s", "best.pt"),
    "yolo11n.pt": ("dronefreak/visdrone-yolov11n", "best.pt"),
}


def _resolve_pretrained(model_name):
    """Try the HF VisDrone checkpoint matching this preset's model size.
    Any failure - offline, repo renamed, filename mismatch - falls back to
    stock and says so, rather than crashing the one training run that
    cannot be re-launched if the night is spent."""
    repo = PRETRAINED.get(model_name)
    if repo is None:
        return model_name, False
    repo_id, filename = repo
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"[train] pretrained base: {repo_id}/{filename} (two-stage: aerial -> night)")
        return path, True
    except Exception as e:
        print(f"[train] could not fetch {repo_id}/{filename} ({e}) - "
              f"falling back to stock {model_name}")
        return model_name, False


def _lr0_for(two_stage):
    """Stock COCO weights want the default from-scratch-transfer lr0 (0.01).
    A VisDrone-pretrained base is already converged on this task; hitting it
    with that same lr0 for a full unfrozen fine-tune risks forgetting the
    aerial adaptation in the first few epochs before it recovers - spending
    the one overnight run relearning what was already paid for. 10x lower
    keeps this run's job to the night gap, not re-deriving VisDrone."""
    return 0.001 if two_stage else 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="overnight", choices=list(PRESETS))
    ap.add_argument("--data", default="VisDrone.yaml",
                    help="ultralytics fetches VisDrone automatically on first use")
    ap.add_argument("--name", default="aerial_night")
    ap.add_argument("--base", default=None,
                    help="Force a specific checkpoint (local path, HF Hub id, "
                         "or GitHub release) as the training base, overriding "
                         "the preset's own PRETRAINED lookup. Verify it really "
                         "was trained on VisDrone, check the licence, and A/B "
                         "it in F0 - a random community checkpoint can be "
                         "worse than stock.")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip the HF VisDrone lookup and train from stock "
                         "COCO weights, as before")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last.pt - a "
                         "Kaggle session dying at epoch 30 of 40 should not "
                         "cost you the whole night")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="'0' for one GPU, '0,1' for both T4s (DDP - Ultralytics "
                         "splits the batch across them, not a clean 2x speedup). "
                         "Default: Ultralytics auto-picks GPU 0.")
    a = ap.parse_args()

    from ultralytics import YOLO
    model_name, imgsz, epochs, batch, freeze = PRESETS[a.preset]

    if a.resume:
        last = f"weights/{a.name}/weights/last.pt"
        if not Path(last).exists():
            sys.exit(f"[train] --resume given but {last} does not exist - "
                     f"nothing to resume (check --name matches the original run)")
        print(f"[train] resuming from {last}")
        YOLO(last).train(resume=True)
        return

    if a.base:
        base = a.base
        two_stage = True
        print(f"[train] base checkpoint (forced): {base} (two-stage: aerial -> night)")
    elif a.no_pretrained:
        base = model_name
        two_stage = False
    else:
        base, two_stage = _resolve_pretrained(model_name)
    lr0 = _lr0_for(two_stage)
    print(f"[train] {base} @ {imgsz}px, {epochs} epochs, batch {batch}, "
          f"freeze={freeze} backbone layers, lr0={lr0}")

    YOLO(base).train(
        data=a.data, imgsz=imgsz, epochs=epochs, batch=batch,
        device=a.device,
        freeze=freeze or None, seed=a.seed, save_period=5, lr0=lr0,
        # augmentation aimed at aerial + low light, not at generic photos
        hsv_h=0.010, hsv_s=0.5, hsv_v=0.65,   # hsv_v high: brightness robustness
        degrees=10.0, translate=0.10, scale=0.6, shear=0.0,
        perspective=0.0, flipud=0.2, fliplr=0.5,
        mosaic=1.0, close_mosaic=8, mixup=0.05,
        patience=8, cos_lr=True, cache=False, workers=4,
        project="weights", name=a.name, exist_ok=True, plots=True,
    )
    print("[train] best weights -> weights/%s/weights/best.pt" % a.name)
    print("[train] now A/B it against stock yolo11n on a NIGHT clip before trusting it")


if __name__ == "__main__":
    main()
