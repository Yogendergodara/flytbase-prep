"""ONE-CELL Kaggle runner: builds every dataset and trains every model.

Paste this into a single Kaggle cell and run it:

    !git clone -b enhancements https://github.com/Yogendergodara/flytbase-prep.git /kaggle/working/flytbase-prep 2>/dev/null; \
     cd /kaggle/working/flytbase-prep && git pull -q && python RUN_ALL_KAGGLE.py

Nothing else to do by hand. Datasets are located if attached and DOWNLOADED
if not, so no UI attachment is required - a forgotten attachment silently
skipped the main model on a real run, and mount paths differ between
UI-attached (/kaggle/input/<slug>/) and kagglehub-fetched
(/kaggle/input/datasets/<owner>/<slug>/) datasets, which has broken others.

THE ONLY MANUAL STEP: Session options -> Accelerator -> GPU T4 x2.
Attaching the datasets is optional (it just saves the download time):
  ahc-frames (raw AHC videos), the FloodNet challenge dataset, d-fire

Stages run SEQUENTIALLY on purpose. Running the classifier and the VLM
together crashed a whole Kaggle session ("tried to allocate more memory than
is available"): two models' memory stacked in one 30GB box. The eager
dataset build that caused most of that is fixed, but the classifier only
takes ~15 min against the VLM's hours, so parallelism buys almost nothing
and risks the whole run. Model 1 (the long job) is where both GPUs are used.

Each stage is skipped if its output already exists, so re-running after an
interruption resumes instead of starting over. --force redoes everything.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/kaggle/working/flytbase-prep")
WORK = Path("/kaggle/working/datasets")
INPUT = Path("/kaggle/input")


def log(msg):
    print(f"\n{'=' * 70}\n[run-all] {msg}\n{'=' * 70}", flush=True)


def sh(cmd, label):
    """Run a command, streaming output. Returns True on success.

    A stage failing must not silently poison later stages that depend on it,
    but must also not abort stages that DON'T - so this reports rather than
    raises, and main() decides what a failure blocks.
    """
    print(f"\n$ {cmd}\n", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True, cwd=str(REPO))
    mins = (time.time() - t0) / 60
    print(f"\n[run-all] {label}: {'OK' if rc == 0 else f'FAILED (exit {rc})'} "
          f"after {mins:.1f} min", flush=True)
    return rc == 0


def find_dir(*name_fragments, must_contain=None):
    """Locate an attached dataset by fragments of its path.

    Kaggle mounts a UI-attached dataset at /kaggle/input/<slug>/ but a
    kagglehub-downloaded one at /kaggle/input/datasets/<owner>/<slug>/. Both
    have appeared in this project, and hardcoding either has broken a run.
    Search instead, and require a known child so a near-miss folder can't
    silently match.
    """
    if not INPUT.exists():
        return None
    for depth in ("*", "*/*", "*/*/*"):
        for cand in sorted(INPUT.glob(depth)):
            if not cand.is_dir():
                continue
            low = str(cand).lower()
            if all(f.lower() in low for f in name_fragments):
                if must_contain is None:
                    return cand
                for sub in must_contain:
                    if (cand / sub).exists():
                        return cand
    return None


def kagglehub_fetch(slug):
    """Download a dataset by slug when it isn't attached to this notebook.

    Dataset attachments do not carry over between notebooks, and forgetting
    one silently skipped the main model on a real run. Downloading is the
    difference between "you must remember a UI step" and "it just works":
    inside a Kaggle notebook the session is already authenticated as the
    owner, so this reaches private datasets too.
    """
    try:
        import kagglehub
    except ImportError:
        subprocess.call(f"{sys.executable} -m pip install -q kagglehub", shell=True)
        try:
            import kagglehub
        except ImportError:
            print("[run-all] kagglehub unavailable - cannot auto-download")
            return None
    try:
        print(f"[run-all] {slug} not attached - downloading with kagglehub "
              f"(this can take several minutes for a large dataset)", flush=True)
        return Path(kagglehub.dataset_download(slug))
    except Exception as e:
        print(f"[run-all] could not download {slug}: {type(e).__name__}: {e}")
        return None


def _ahc_from(base):
    """The AHC dataset has been uploaded both with and without an AHC_full/
    wrapper, so accept either shape rather than assuming one."""
    if base is None:
        return None
    for cand in (base / "AHC_full", base):
        if (cand / "train").is_dir():
            return cand
    for sub in base.rglob("*"):          # tolerate a deeper wrapper folder
        if sub.is_dir() and (sub / "train").is_dir() and (sub / "test").is_dir():
            return sub
    return None


def find_ahc_root(slug="yogendergodara/ahc-frames"):
    found = _ahc_from(find_dir("ahc", must_contain=["AHC_full", "train"]))
    if found is not None:
        return found
    return _ahc_from(kagglehub_fetch(slug))


def _floodnet_from(root):
    """FloodNet's Track-1 labelled split is nested several folders deep and
    the exact casing/spacing varies by upload, so find the folder that
    actually holds the Flooded/Non-Flooded pair rather than guessing a path."""
    if root is None:
        return None
    for d in root.rglob("*"):
        if d.is_dir() and d.name.lower().replace("-", " ") == "flooded":
            return d.parent          # the folder containing Flooded/ + Non-Flooded/
    return root


def find_floodnet(slug="aletbm/aerial-imagery-dataset-floodnet-challenge"):
    found = _floodnet_from(find_dir("floodnet"))
    if found is not None:
        return found
    return _floodnet_from(kagglehub_fetch(slug))


def find_dfire(slug="shubhamkarande13/d-fire"):
    found = (find_dir("d-fire", must_contain=["train", "test"])
             or find_dir("dfire", must_contain=["train", "test"])
             or find_dir("d-fire") or find_dir("dfire"))
    if found is not None:
        return found
    return kagglehub_fetch(slug)


def stage_hazard_dataset(force):
    out = WORK / "scene_hazard"
    if out.exists() and any(out.rglob("*.jpg")) and not force:
        log("Model 2 dataset already built - skipping")
        return True
    floodnet, dfire = find_floodnet(), find_dfire()
    if floodnet is None or dfire is None:
        print(f"[run-all] SKIP Model 2 - could not find or download its datasets "
              f"(floodnet={floodnet}, d-fire={dfire}).")
        return False
    log("Model 2: building fire/smoke/flood dataset")
    WORK.mkdir(parents=True, exist_ok=True)
    return sh(f'python train/build_scene_classifier_dataset.py '
              f'--floodnet "{floodnet}" --dfire "{dfire}" --out "{out}"',
              "hazard dataset")


def hazard_ckpt():
    """Ultralytics resolves project= against its own runs_dir setting, so
    `project=weights name=scene_hazard` actually landed in
    runs/classify/weights/scene_hazard/ - not weights/scene_hazard/ as the
    arguments suggest. Checking only the literal path meant a finished model
    looked missing and got retrained for another 20 minutes. Search both."""
    for pat in ("weights/scene_hazard/weights/best.pt",
                "runs/classify/weights/scene_hazard/weights/best.pt",
                "runs/classify/*/scene_hazard*/weights/best.pt",
                "**/scene_hazard*/weights/best.pt"):
        hits = sorted(REPO.glob(pat))
        if hits:
            return hits[0]
    return None


def stage_hazard_train(force):
    found = hazard_ckpt()
    if found and not force:
        log(f"Model 2 already trained ({found.relative_to(REPO)}) - skipping")
        return True
    if not (WORK / "scene_hazard").exists():
        return False
    log("Model 2: training hazard classifier (~15-20 min, 1 GPU)")
    # one GPU, no cache=ram: this model used 565MiB of 15GB, and cache=ram
    # added memory pressure that contributed to a session OOM for no gain
    return sh(f'yolo classify train data="{WORK}/scene_hazard" model=yolo11n-cls.pt '
              f'imgsz=224 epochs=15 batch=32 project=weights name=scene_hazard device=0',
              "hazard training")


def stage_ahc_extract(force):
    manifest = REPO / "train/ahc_manifest.jsonl"
    frames = WORK / "AHC_frames"
    if manifest.exists() and frames.exists() and not force:
        log("Model 3 frames already extracted - skipping")
        return True
    root = find_ahc_root()
    if root is None:
        print("[run-all] SKIP Model 3 - the AHC dataset was neither attached "
              "nor downloadable. Check the slug, or attach it manually: "
              "Add Input -> Datasets -> ahc-frames.")
        return False
    log(f"Model 3: extracting frames from {root} (~75 min measured on Kaggle, CPU only)")
    return sh(f'python train/extract_ahc_frames.py --root "{root}" '
              f'--out "{frames}" --frames-per-crop 8 --crops-per-event 3 '
              f'--manifest train/ahc_manifest.jsonl', "AHC extraction")


def stage_ahc_train(force):
    out = REPO / "weights/qwen_ahc_lora"
    if (out / "adapter_config.json").exists() and not force:
        log("Model 3 adapter already trained - skipping")
        return True
    if not (REPO / "train/ahc_manifest.jsonl").exists():
        return False
    log("Model 3: fine-tuning Qwen2.5-VL (~3-6h, 1 GPU)")
    return sh(f'python train/finetune_ahc_vlm.py --manifest train/ahc_manifest.jsonl '
              f'--frames-root "{WORK}/AHC_frames" --base Qwen/Qwen2.5-VL-3B-Instruct '
              f'--out weights/qwen_ahc_lora --epochs 3', "AHC fine-tune")


def stage_ahc_eval(force):
    if not (REPO / "weights/qwen_ahc_lora/adapter_config.json").exists():
        return False
    log("Model 3: evaluating on the held-out test split (mandatory)")
    return sh(f'python train/eval_ahc_vlm.py --manifest train/ahc_manifest.jsonl '
              f'--frames-root "{WORK}/AHC_frames" --base Qwen/Qwen2.5-VL-3B-Instruct '
              f'--adapter weights/qwen_ahc_lora', "AHC eval")


def stage_aerial(force):
    """Model 1 is OPT-IN. A trained checkpoint already exists (Phase 5:
    P 0.598 / R 0.464 / mAP50-95 0.291) and it early-stopped at epoch 33/40,
    meaning it had converged - re-running the same VisDrone-only recipe
    lands in the same place, for 5-6 GPU-hours. Improving it needs different
    data (UIT-ADrone, no verified source found), a bigger model, or higher
    resolution - not another identical run."""
    log("Model 1: aerial detector - training it (both GPUs, ~5-6h)")
    return sh('yolo settings datasets_dir=/kaggle/working/datasets && '
              'python train/train_aerial.py --preset kaggle --name aerial_v2 --device 0,1',
              "aerial training")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="redo stages whose outputs already exist")
    ap.add_argument("--with-aerial", action="store_true",
                    help="also retrain Model 1 (5-6h). Off by default because a "
                         "converged checkpoint already exists - see stage_aerial")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()

    log("Environment")
    sh("python -c \"import torch; print('cuda:', torch.cuda.is_available(), "
       "'| gpus:', torch.cuda.device_count())\"", "gpu check")
    print(f"[run-all] attached under {INPUT}: "
          f"{sorted(p.name for p in INPUT.glob('*')) if INPUT.exists() else 'NOTHING'}")
    print(f"[run-all] resolved AHC root : {find_ahc_root()}")
    print(f"[run-all] resolved FloodNet : {find_floodnet()}")
    print(f"[run-all] resolved D-Fire   : {find_dfire()}")

    results = {}
    results["model2_dataset"] = stage_hazard_dataset(a.force)
    if results["model2_dataset"]:
        results["model2_train"] = stage_hazard_train(a.force)

    results["model3_extract"] = stage_ahc_extract(a.force)
    if results["model3_extract"]:
        results["model3_train"] = stage_ahc_train(a.force)
        if results.get("model3_train") and not a.skip_eval:
            results["model3_eval"] = stage_ahc_eval(a.force)

    if a.with_aerial:
        results["model1_train"] = stage_aerial(a.force)

    log("FINAL REPORT")
    for k, v in results.items():
        print(f"  {k:20s} {'OK' if v else 'FAILED / SKIPPED'}")

    print("\n[run-all] artifacts to DOWNLOAD before the session ends "
          "(/kaggle/working is wiped):")
    hz = hazard_ckpt()
    print(f"  {'[x]' if hz else '[ ]'} "
          f"{hz.relative_to(REPO) if hz else 'scene_hazard best.pt (Model 2)'}")
    for p in ["weights/qwen_ahc_lora", "out/ahc_eval.json"]:
        print(f"  {'[x]' if (REPO / p).exists() else '[ ]'} {p}")
    aerial = sorted(REPO.glob("**/aerial_v2*/weights/best.pt"))
    print(f"  {'[x]' if aerial else '[ ]'} "
          f"{aerial[0].relative_to(REPO) if aerial else 'aerial_v2 best.pt (Model 1, optional)'}")

    if not a.with_aerial:
        print("\n[run-all] Model 1 (aerial detector) was NOT retrained - a "
              "converged checkpoint already exists (mAP50-95 0.291). Pass "
              "--with-aerial to retrain, but read stage_aerial's docstring "
              "first: the same recipe will reproduce the same number.")
    if not all(results.values()):
        print("\n[run-all] Something was skipped or failed - check the FINAL "
              "REPORT above. Most common cause: a dataset is not attached in "
              "this notebook.")
        sys.exit(1)


if __name__ == "__main__":
    main()
