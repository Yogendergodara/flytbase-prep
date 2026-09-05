"""Produce AHC hackathon submission predictions from raw videos.

This is the actual deliverable: everything else in this repo either trains a
model or scores one. Nothing turned a folder of videos into the rows the
hackathon scores until this file. run.py emits out/alerts.json in this
repo's own internal cascade format, which is not the submission schema.

    # whole-video prediction (Level 1: is_anomaly + class, no timestamps)
    python predict_ahc.py --videos datasets/AHC_full/test/videos \
        --adapter weights/qwen_ahc_lora --out out/submission.csv

    # + temporal localisation (Levels 2-3: start/end filled in)
    python predict_ahc.py --videos datasets/AHC_full/test/videos \
        --adapter weights/qwen_ahc_lora --localize --out out/submission.csv

Output columns match test/ground_truth.csv exactly:
    video_id,level,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary

Degrades rather than failing:
  - no adapter -> runs the zero-shot base model and SAYS SO, so a missing or
    still-training fine-tune produces a submittable file instead of nothing
  - one unreadable video -> that row becomes `normal` with a note, and the
    remaining videos still get predicted. A crash on video 17 of 34 that
    discards the other 33 is the worst possible outcome this close to a
    deadline.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))

from finetune_ahc_vlm import AHC_CLASSES, AHC_PROMPT, cap_pixels
from pipeline.vlm_judge import extract_frames

FIELDS = ["video_id", "level", "is_anomaly", "class_name",
          "start_time_sec", "end_time_sec", "description_summary"]


def _parse_json(raw):
    """The model emits prose sometimes. Take the first balanced object rather
    than a greedy {.*} span, which swallows trailing text and fails to parse."""
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _coerce_class(name):
    """Map whatever the model said onto one of the 12 official strings.

    A prediction that isn't an exact class string scores as wrong no matter
    how right the model was, so normalise case/spacing first and only then
    fall back. Falling back to `normal` (not to a random anomaly class) is
    deliberate: on an unparseable answer, claiming "nothing happened" is the
    conservative error, and `normal` is also the single most common label.
    """
    if not name:
        return None
    n = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if n in AHC_CLASSES:
        return n
    for c in AHC_CLASSES:                       # tolerate partial/verbose answers
        if c in n or n in c:
            return c
    return None


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def video_duration(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (n / fps) if fps > 0 else 0.0


class HazardScreen:
    """Cheap always-on stage: the trained yolo11n-cls fire/smoke/flood model.

    The brief calls for "a lightweight always-on stage with a heavier
    verification step", and for inference cheap enough to run across many
    feeds at once. This is that first stage - measured at 1.2ms/image
    during its own training run, against ~seconds per call for the 3B VLM.

    It does NOT short-circuit the VLM. Fire/smoke/flood are 3 of 12 classes,
    so a confident hazard call cannot answer the other 9, and letting a
    small classifier veto the VLM would trade precision for latency in the
    direction the brief explicitly warns about ("false alarms matter as
    much as missed detections"). It is used two ways instead: as a recorded
    second opinion, and to rescue the case where the VLM says `normal` but
    this model is highly confident a hazard is present - a missed detection
    the cheap stage can catch for almost no cost.
    """

    HAZARD = {"fire", "smoke", "flood"}
    # flood -> the official class string; the classifier's own label set is
    # {fire, smoke, flood, normal}, which is not the 12-class vocabulary
    TO_AHC = {"fire": "fire", "smoke": "smoke", "flood": "waterlogging_or_flood"}

    def __init__(self, weights, conf=0.80):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf = conf
        self.calls = 0
        self.seconds = 0.0

    def screen(self, video_path, t0, t1, n_frames=3):
        """Returns (ahc_class, confidence) or (None, 0.0)."""
        import numpy as np
        frames = extract_frames(str(video_path), t0, t1, n_frames)
        if not frames:
            return None, 0.0
        t = time.time()
        best_name, best_p = None, 0.0
        for f in frames:
            r = self.model.predict(f, verbose=False)[0]
            probs = r.probs
            if probs is None:
                continue
            name = r.names[int(probs.top1)]
            p = float(probs.top1conf)
            if name in self.HAZARD and p > best_p:
                best_name, best_p = name, p
        self.seconds += time.time() - t
        self.calls += 1
        if best_name and best_p >= self.conf:
            return self.TO_AHC[best_name], best_p
        return None, best_p


class AHCPredictor:
    """Loads the model once, predicts many videos.

    Loading a 3B model per video would dominate runtime on a 34-video test
    set; this keeps one instance alive across the whole run.
    """

    def __init__(self, base, adapter=None, max_pixels=401408, max_frames=3,
                 load_4bit=True):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.max_pixels, self.max_frames = max_pixels, max_frames

        kwargs = {"device_map": "auto"}
        if load_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16)
            except Exception as e:
                print(f"[predict] 4-bit unavailable ({type(e).__name__}) - "
                      f"loading full precision. The adapter was TRAINED in "
                      f"4-bit, so predictions may differ slightly.")
        # transformers 5.0 renamed torch_dtype -> dtype; try both so this
        # works on either side of that change
        for dt in ("dtype", "torch_dtype", None):
            try:
                kw = dict(kwargs)
                if dt and "quantization_config" not in kw:
                    kw[dt] = "auto"
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, **kw)
                break
            except TypeError:
                continue
        else:
            raise RuntimeError("[predict] could not load the base model")

        self.zero_shot = True
        if adapter and Path(adapter, "adapter_config.json").exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            n_lora = sum(1 for n, _ in self.model.named_modules() if "lora" in n.lower())
            if n_lora == 0:
                # loading "successfully" while matching nothing would mean
                # silently submitting base-model predictions labelled as
                # fine-tuned - refuse rather than mislead
                raise SystemExit(
                    f"[predict] adapter at {adapter} matched NO modules - "
                    f"predictions would silently be the base model's. Check "
                    f"--base matches what training used.")
            print(f"[predict] fine-tuned adapter loaded ({n_lora} LoRA modules)")
            self.zero_shot = False
        else:
            print(f"[predict] WARNING: no adapter at {adapter} - using the "
                  f"ZERO-SHOT base model. This still produces a valid "
                  f"submission, but it is NOT your fine-tuned model.")

        # prefer the processor training saved next to the adapter: Unsloth
        # mutates it (pixel bounds, padding side, chat template), and using
        # the stock one is a train/inference mismatch
        src = adapter if (adapter and Path(adapter, "preprocessor_config.json").exists()) else base
        self.proc = AutoProcessor.from_pretrained(src)
        print(f"[predict] processor from {src}")
        self.model.eval()

    def predict_window(self, video_path, t0, t1):
        from PIL import Image
        frames = extract_frames(str(video_path), t0, t1, self.max_frames)
        if not frames:
            return None
        images = [cap_pixels(Image.fromarray(f).convert("RGB"), self.max_pixels)
                  for f in frames]
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text",
                        "text": AHC_PROMPT.format(classes=", ".join(AHC_CLASSES))})
        chat = self.proc.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[chat], images=images,
                           return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        raw = self.proc.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0]
        return _parse_json(raw)


def predict_video(pred, path, localize, n_windows):
    """One video -> one submission row.

    Without --localize this is a single whole-video call (Level 1: the class
    and is_anomaly, timestamps left blank, which is what Level 1 expects).
    With it, the video is split into overlapping windows and the longest
    contiguous run of same-class anomalous windows becomes the event span -
    real timestamps rather than a guess spanning the whole clip.
    """
    duration = video_duration(path)
    if duration <= 0:
        return None, "unreadable"

    if not localize:
        out = pred.predict_window(path, 0.0, duration)
        return (out, duration), None

    win = duration / n_windows
    results = []
    for i in range(n_windows):
        t0, t1 = i * win, min(duration, (i + 1) * win)
        r = pred.predict_window(path, t0, t1)
        cls = _coerce_class((r or {}).get("class_name"))
        results.append((t0, t1, cls, r))

    # longest contiguous run of the same non-normal class
    best = None
    i = 0
    while i < len(results):
        cls = results[i][2]
        if cls and cls != "normal":
            j = i
            while j + 1 < len(results) and results[j + 1][2] == cls:
                j += 1
            span = (results[i][0], results[j][1], cls, results[i][3], j - i + 1)
            if best is None or span[4] > best[4]:
                best = span
            i = j + 1
        else:
            i += 1

    if best is None:                       # every window looked normal
        whole = pred.predict_window(path, 0.0, duration)
        return (whole, duration), None
    t0, t1, cls, raw, _ = best
    raw = dict(raw or {})
    raw["class_name"], raw["is_anomaly"] = cls, True
    return (raw, duration), (t0, t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True,
                    help="folder of .mp4 files, or a videos.csv listing them")
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="weights/qwen_ahc_lora",
                    help="fine-tuned LoRA. If absent, falls back to zero-shot "
                         "and says so - a submittable file beats no file")
    ap.add_argument("--out", default="out/submission.csv")
    ap.add_argument("--max-pixels", type=int, default=401408)
    ap.add_argument("--max-frames", type=int, default=3,
                    help="must match training's --max-frames")
    ap.add_argument("--localize", action="store_true",
                    help="split each video into windows to produce real "
                         "start/end times (Levels 2-3). Costs --windows x "
                         "more inference per video.")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--level", type=int, default=None,
                    help="value for the `level` column; defaults to 2 with "
                         "--localize (timestamps present) and 1 without")
    ap.add_argument("--limit", type=int, default=0, help="first N videos only (smoke test)")
    ap.add_argument("--hazard-weights", default="weights/scene_hazard/weights/best.pt",
                    help="the cheap always-on fire/smoke/flood screen. Runs as a "
                         "second opinion and can rescue a missed hazard the VLM "
                         "called normal; never vetoes the VLM. Pass '' to disable.")
    ap.add_argument("--hazard-conf", type=float, default=0.80,
                    help="only override a VLM `normal` when the cheap screen is at "
                         "least this confident - set high on purpose, since the "
                         "brief weighs false alarms as heavily as misses")
    a = ap.parse_args()

    src = Path(a.videos)
    if src.is_file() and src.suffix == ".csv":
        rows = list(csv.DictReader(src.open(encoding="utf-8")))
        videos = [(r["video_id"], src.parent / r["filename"]) for r in rows]
    else:
        videos = [(p.stem, p) for p in sorted(src.glob("*.mp4"))]
    if a.limit:
        videos = videos[:a.limit]
    if not videos:
        raise SystemExit(f"[predict] no videos found under {src}")
    print(f"[predict] {len(videos)} videos, localize={a.localize}")

    pred = AHCPredictor(a.base, a.adapter, a.max_pixels, a.max_frames)
    level = a.level if a.level is not None else (2 if a.localize else 1)

    screen = None
    if a.hazard_weights and Path(a.hazard_weights).exists():
        screen = HazardScreen(a.hazard_weights, a.hazard_conf)
        print(f"[predict] cheap hazard screen active ({a.hazard_weights})")
    elif a.hazard_weights:
        print(f"[predict] no hazard screen at {a.hazard_weights} - VLM only")

    total_video_seconds = 0.0
    n_rescued = 0

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_anom = n_failed = 0
    t_start = time.time()

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, (vid, path) in enumerate(videos, 1):
            try:
                result, span = predict_video(pred, path, a.localize, a.windows)
            except Exception as e:
                # never let one video cost the other 33
                print(f"[predict] {vid}: FAILED ({type(e).__name__}: {e})")
                result, span = None, None
            if result is None:
                n_failed += 1
                w.writerow({"video_id": vid, "level": level, "is_anomaly": "false",
                            "class_name": "normal", "start_time_sec": "",
                            "end_time_sec": "",
                            "description_summary": "prediction failed for this video"})
                fh.flush()
                continue

            raw, duration = result
            total_video_seconds += duration
            cls = _coerce_class((raw or {}).get("class_name")) or "normal"
            is_anom = _as_bool((raw or {}).get("is_anomaly")) if raw else False

            # cheap-stage rescue: only ever promotes `normal` -> a hazard the
            # small model is highly confident about. It cannot change one
            # anomaly class into another, and cannot silence the VLM.
            if screen is not None and cls == "normal":
                hz, p = screen.screen(path, 0.0, duration, a.max_frames)
                if hz:
                    print(f"[predict]   cheap screen overrode normal -> {hz} (p={p:.2f})")
                    cls, n_rescued = hz, n_rescued + 1
                    raw = dict(raw or {})
                    raw["description"] = (raw.get("description")
                                          or f"{hz} detected by hazard screen")
            # keep the two fields consistent: a class of `normal` with
            # is_anomaly=true (or the reverse) is self-contradictory and the
            # scorer reads both
            is_anom = (cls != "normal")
            desc = str((raw or {}).get("description", "") or "").strip() \
                   or ("no target anomaly observed" if not is_anom else f"{cls} observed")

            t0 = t1 = ""
            if is_anom and span:
                t0, t1 = f"{span[0]:.3f}", f"{span[1]:.3f}"
            elif is_anom and a.localize:
                t0, t1 = "0.000", f"{duration:.3f}"

            n_anom += is_anom
            w.writerow({"video_id": vid, "level": level,
                        "is_anomaly": "true" if is_anom else "false",
                        "class_name": cls, "start_time_sec": t0, "end_time_sec": t1,
                        "description_summary": desc.replace("\n", " ")})
            fh.flush()          # partial file survives a session dying mid-run
            print(f"[predict] {i}/{len(videos)} {vid}: {cls} "
                  f"{'(' + t0 + '-' + t1 + 's)' if t0 else ''}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n[predict] wrote {out_path} - {len(videos)} rows, {n_anom} anomalous, "
          f"{n_failed} failed, in {elapsed/60:.1f} min")
    if n_rescued:
        print(f"[predict] cheap hazard screen rescued {n_rescued} video(s) the "
              f"VLM had called normal")

    # The brief's central constraint is real-time on limited GPU, so state a
    # measured number rather than a claim. Realtime factor > 1 means the
    # system processes footage faster than it plays.
    if total_video_seconds > 0:
        rtf = total_video_seconds / elapsed
        print(f"\n[predict] === LATENCY (measured, not estimated) ===")
        print(f"  video processed   : {total_video_seconds/60:.1f} min of footage")
        print(f"  wall clock        : {elapsed/60:.1f} min")
        print(f"  realtime factor   : {rtf:.2f}x "
              f"({'faster' if rtf > 1 else 'SLOWER'} than realtime)")
        print(f"  per video         : {elapsed/max(len(videos),1):.1f} s")
        if screen is not None and screen.calls:
            print(f"  cheap screen      : {screen.seconds/screen.calls*1000:.0f} ms/call "
                  f"over {screen.calls} calls "
                  f"({100*screen.seconds/elapsed:.1f}% of total time)")
        print(f"  concurrent feeds  : ~{rtf:.1f} on this one GPU at this sampling rate")
        print(f"  NOTE: this samples {a.max_frames} frames per clip, it does not "
              f"decode every frame - that is the design, but say so rather than "
              f"implying full-framerate processing.")
    if pred.zero_shot:
        print("[predict] REMINDER: this used the ZERO-SHOT base model, not a "
              "fine-tuned adapter. Re-run once training finishes.")


if __name__ == "__main__":
    main()
