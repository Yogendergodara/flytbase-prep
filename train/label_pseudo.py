"""P18 step 2: teacher labeling for VLM distillation.

ENHANCEMENT_PLAN.md Phase 18 step 1's feasibility check MUST pass before
this runs: a real teacher-model API endpoint (OpenAI-compatible chat
completions with image input) and a known budget for however many candidate
events get labelled. Qwen2.5-VL-72B needs ~140GB+ VRAM even 4-bit - use a
hosted API, not a local/Kaggle load.

Labels the SAME candidate-event windows the small model is judged on - not
random frames - using pipeline/vlm_judge.PROMPT verbatim, so the pseudo-labels
and the fine-tuned model's real inputs match exactly.

    # 1. Geometric-only pass to get the candidate list (no local VLM call):
    python run.py --video data/clip1.mp4 --set vlm.backend=none --out out/candidates_clip1.json

    # 2. Label those candidates with a teacher API:
    export TEACHER_API_BASE=https://api.<provider>.com/v1
    export TEACHER_API_KEY=...
    export TEACHER_MODEL=qwen2.5-vl-72b-instruct
    python train/label_pseudo.py --alerts out/candidates_clip1.json --video data/clip1.mp4
"""
import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

# same fix as train/extract_ahc_frames.py: `python train/label_pseudo.py`
# (this file's own documented invocation) puts only train/ on sys.path, not
# the repo root, so `from pipeline...` fails with ModuleNotFoundError -
# confirmed by actually hitting this running the sibling script, not assumed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.vlm_judge import PROMPT, extract_frames


def _b64_jpeg(frame_rgb, quality=85):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame_rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_teacher(prompt_text, frames_rgb, api_base, api_key, model, timeout=60):
    """OpenAI-compatible chat completions with image_url content blocks - the
    widest-compatibility shape across hosted Qwen2.5-VL-72B endpoints. Swap
    this function's body, not its signature, if your provider's API differs."""
    import requests
    content = [{"type": "text", "text": prompt_text}]
    for f in frames_rgb:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(f)}"}})
    resp = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": content}],
              "temperature": 0.0, "max_tokens": 200},
        timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_verdict(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", required=True,
                    help="out/*.json from a run with vlm.backend=none - its "
                         "'candidates' list is what gets labelled")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default="train/pseudo_data")
    ap.add_argument("--frames-per-event", type=int, default=6)
    ap.add_argument("--api-base", default=os.environ.get("TEACHER_API_BASE"))
    ap.add_argument("--api-key", default=os.environ.get("TEACHER_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL"))
    a = ap.parse_args()

    if not (a.api_base and a.api_key and a.model):
        raise SystemExit(
            "Feasibility check failed (ENHANCEMENT_PLAN.md Phase 18 step 1): "
            "TEACHER_API_BASE / TEACHER_API_KEY / TEACHER_MODEL are not all "
            "set. Confirm real API access and a labeling budget BEFORE "
            "running this - don't discover the gap mid-labeling.")

    d = json.load(open(a.alerts, encoding="utf-8"))
    candidates = d.get("candidates", [])
    if not candidates:
        raise SystemExit(f"no 'candidates' in {a.alerts} - re-run with "
                         f"--set vlm.backend=none first so the geometric "
                         f"stage's raw event list is written")

    img_dir = Path(a.out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(a.out_dir) / "pseudo_labels.jsonl"

    from PIL import Image
    n_ok, n_failed = 0, 0
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for i, ev in enumerate(candidates):
            frames = extract_frames(a.video, ev["t_start"], ev["t_end"],
                                     a.frames_per_event)
            if not frames:
                n_failed += 1
                continue
            prompt_text = PROMPT.format(
                cls=ev["cls"], tid=ev["track_id"], kind=ev["kind"],
                facts=ev["facts"], t0=ev["t_start"], t1=ev["t_end"])
            try:
                raw = call_teacher(prompt_text, frames, a.api_base, a.api_key, a.model)
            except Exception as e:
                print(f"[label_pseudo] event {i} ({ev['kind']} @{ev['t_start']:.1f}s) "
                     f"teacher call failed: {type(e).__name__}: {e}")
                n_failed += 1
                continue
            verdict = _parse_verdict(raw)
            if verdict is None:
                print(f"[label_pseudo] event {i}: unparseable teacher reply, skipped")
                n_failed += 1
                continue

            image_paths = []
            for j, f in enumerate(frames):
                p = img_dir / f"event{i:04d}_f{j}.jpg"
                Image.fromarray(f).save(p, quality=90)
                image_paths.append(str(p))

            manifest.write(json.dumps({
                "event_index": i, "kind": ev["kind"], "track_id": ev["track_id"],
                "t_start": ev["t_start"], "t_end": ev["t_end"],
                "facts": ev["facts"], "prompt": prompt_text,
                "image_paths": image_paths, "verdict": verdict,
            }) + "\n")
            n_ok += 1

    print(f"[label_pseudo] {n_ok} labelled, {n_failed} failed/skipped -> {manifest_path}")


if __name__ == "__main__":
    main()
