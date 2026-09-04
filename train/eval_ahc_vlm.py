"""Step 3, MANDATORY: score the fine-tuned adapter against the AHC test
split - the real held-out set, never touched by finetune_ahc_vlm.py.

Governing rule (CLAUDE.md): report on held-out data, never on data trained
on. This is that check for the AHC fine-tune, the same way scripts/ab_weights.py
is that check for the aerial detector.

    python train/eval_ahc_vlm.py --adapter weights/qwen_ahc_lora \\
        --manifest train/ahc_manifest.jsonl --base Qwen/Qwen2.5-VL-3B-Instruct

Also runs the BASE model with zero-shot prompting on the same test events, so
the adapter's number is reported against something, not presented alone -
same reasoning as the aerial detector's mandatory stock-vs-tuned A/B.
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from finetune_ahc_vlm import AHC_CLASSES, AHC_PROMPT


def load_test_rows(manifest_path):
    rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    test = [r for r in rows if r["split"] == "test"]
    if not test:
        raise SystemExit(f"[eval] no split=='test' rows in {manifest_path} - "
                         f"did train/extract_ahc_frames.py find datasets/AHC_full/test?")
    return test


def _parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _as_bool(v):
    """The model emits JSON text, so is_anomaly can come back as the string
    "true" rather than a real boolean. Comparing that to a Python bool with
    == is always False, which would UNDERSTATE is_anomaly accuracy - a
    measurement bug that makes the model look worse than it is. Normalise
    both sides before comparing."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v) if v is not None else None


def predict(model, tokenizer, row, frames_root, max_pixels=401408, max_new_tokens=150):
    from pathlib import Path
    from PIL import Image
    import torch
    from finetune_ahc_vlm import cap_pixels
    # same relative-path convention as finetune_ahc_vlm.py's to_record -
    # the manifest and the AHC_frames folder travel to Kaggle separately
    paths = [p if Path(p).is_absolute() else str(Path(frames_root) / p) for p in row["frame_paths"]]
    # same pixel cap as training: scoring the adapter on differently-scaled
    # frames than it was trained on would measure the wrong thing (and can
    # overflow the context with 8 full-res frames)
    images = [cap_pixels(Image.open(p).convert("RGB"), max_pixels) for p in paths]
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": AHC_PROMPT.format(classes=", ".join(AHC_CLASSES))})
    messages = [{"role": "user", "content": content}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[chat], images=images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = tokenizer.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return _parse(raw)


def score(model, tokenizer, test_rows, label, frames_root, max_pixels=401408):
    n_correct_class = n_correct_anomaly = n_parsed = 0
    per_class = defaultdict(lambda: [0, 0])  # [correct, total]
    confusion = Counter()
    n_errors = 0
    for row in test_rows:
        # One truncated JPEG or one CUDA OOM must not discard a completed
        # eval - the fine-tuned run is the number that matters and it comes
        # AFTER the base run, so an unhandled raise here throws away both.
        try:
            pred = predict(model, tokenizer, row, frames_root, max_pixels)
        except Exception as e:
            print(f"[eval] {row['video_id']} failed ({type(e).__name__}: {e}) "
                  f"- counted as unparsed")
            pred, n_errors = None, n_errors + 1
        truth_cls = row["class_name"]
        per_class[truth_cls][1] += 1
        if pred is None:
            confusion[(truth_cls, "UNPARSED")] += 1
            continue
        n_parsed += 1
        # normalise both sides: the model may emit a class with different
        # case/whitespace, or is_anomaly as the string "true" - none of which
        # are real errors, but all of which would be scored as errors on a
        # raw == comparison
        pred_cls = (pred.get("class_name") or "").strip().lower()
        pred_anom = _as_bool(pred.get("is_anomaly"))
        confusion[(truth_cls, pred_cls or "EMPTY")] += 1
        if pred_cls == truth_cls.strip().lower():
            n_correct_class += 1
            per_class[truth_cls][0] += 1
        if pred_anom == _as_bool(row["is_anomaly"]):
            n_correct_anomaly += 1

    n = len(test_rows)
    print(f"\n=== {label} ===")
    print(f"parsed: {n_parsed}/{n}" + (f" ({n_errors} hard errors)" if n_errors else ""))
    print(f"class accuracy: {n_correct_class}/{n} = {n_correct_class/n:.3f}")
    print(f"is_anomaly accuracy: {n_correct_anomaly}/{n} = {n_correct_anomaly/n:.3f}")
    # n=32 over 12 classes is ~2-3 examples per class: one flipped example
    # moves overall accuracy by ~3 points and a per-class row by 33-50. The
    # counts are printed next to every ratio precisely so nobody quotes
    # these as precise numbers.
    print(f"NOTE: n={n} total. At this size one example is "
          f"{100/n:.1f} points of overall accuracy - treat per-class rows "
          f"below as indicative, not measured.")
    print("per-class recall (correct/total):")
    for cls in AHC_CLASSES:
        c, t = per_class.get(cls, [0, 0])
        if t:
            print(f"  {cls:36s} {c}/{t} = {c/t:.2f}")
    return {"class_acc": n_correct_class / n, "anomaly_acc": n_correct_anomaly / n,
            "confusion": {f"{a}->{b}": v for (a, b), v in confusion.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="train/ahc_manifest.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="weights/qwen_ahc_lora")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the zero-shot base-model comparison run (faster, "
                         "but then the adapter's number has nothing to compare against)")
    ap.add_argument("--out", default="out/ahc_eval.json")
    ap.add_argument("--max-pixels", type=int, default=401408,
                    help="must match finetune_ahc_vlm.py's --max-pixels - scoring "
                         "on differently-scaled frames than training used measures "
                         "the wrong thing")
    ap.add_argument("--frames-root", default="datasets/AHC_frames",
                    help="where AHC_frames actually landed on THIS machine/Kaggle "
                         "session - same convention as finetune_ahc_vlm.py")
    a = ap.parse_args()

    test_rows = load_test_rows(a.manifest)
    print(f"[eval] {len(test_rows)} held-out test events (public test set, "
          f"never trained on)")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    def load_base():
        """Load the base the SAME way training did: 4-bit NF4.

        finetune_ahc_vlm.py trains with load_in_4bit=True, so the LoRA is fit
        against NF4-quantised frozen weights. Attaching that adapter to an
        unquantised fp16 base - which this used to do - is a known and
        uncontrolled accuracy loss: the number reported would not describe
        the model that was trained. Also: transformers 5.0 removed
        `torch_dtype=` in favour of `dtype=`, so pass the new name and fall
        back, rather than having it silently swallowed and the model land in
        fp32 (~12GB, which does not fit a 15GB T4 twice over).
        """
        kwargs = {"device_map": "auto"}
        try:
            from transformers import BitsAndBytesConfig
            import torch
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
        except Exception as e:
            print(f"[eval] 4-bit unavailable ({type(e).__name__}) - loading full "
                  f"precision. NOTE: the adapter was trained in 4-bit, so this "
                  f"comparison is not exactly the trained model.")
            kwargs["dtype"] = "auto"
        for dt in ("dtype", "torch_dtype", None):
            try:
                kw = dict(kwargs)
                if dt and "quantization_config" not in kw:
                    kw[dt] = "auto"
                return Qwen2_5_VLForConditionalGeneration.from_pretrained(a.base, **kw)
            except TypeError:
                continue
        raise RuntimeError("[eval] could not load the base model")

    def load_proc(prefer):
        """Prefer the processor training SAVED. Unsloth mutates the processor
        it returns (pixel bounds, padding side, chat-template details), and
        finetune_ahc_vlm.py deliberately saves it next to the adapter. Scoring
        with the stock base processor instead is a train/eval preprocessing
        mismatch that shows up as a depressed fine-tuned score for no
        modelling reason."""
        try:
            p = AutoProcessor.from_pretrained(prefer)
            print(f"[eval] processor loaded from {prefer}")
            return p
        except Exception:
            print(f"[eval] no processor at {prefer} - falling back to {a.base}")
            return AutoProcessor.from_pretrained(a.base)

    results = {}
    if not a.skip_base:
        base_model = load_base()
        base_model.eval()
        proc = load_proc(a.base)
        results["zero_shot_base"] = score(base_model, proc, test_rows,
                                          "zero-shot base (no fine-tune)", a.frames_root, a.max_pixels)
        # `del` alone does not return VRAM to the allocator, so the second
        # model load below would stack on top of the first and can OOM a
        # 15GB T4 - the adapter run is the one that matters, and losing it
        # to a memory error after the base run already succeeded would be
        # an avoidable waste.
        del base_model
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    tuned_model = load_base()
    tuned_model = PeftModel.from_pretrained(tuned_model, a.adapter)
    tuned_model.eval()

    # Verify the adapter actually attached to something. Unsloth remaps the
    # base repo and transformers reorganised Qwen2.5-VL's submodule paths
    # across 4.5x -> 5.x, so PeftModel.from_pretrained can load while
    # matching ZERO target modules - in which case the "fine-tuned" score is
    # just the base model again, and the verdict below is meaningless. Fail
    # loudly instead of reporting a fake comparison.
    n_lora = sum(1 for n, _ in tuned_model.named_modules() if "lora" in n.lower())
    if n_lora == 0:
        raise SystemExit(
            f"[eval] the adapter at {a.adapter} matched NO modules in this base "
            f"model - every 'fine-tuned' number would silently be the base "
            f"model's. This usually means the base repo or the module naming "
            f"differs from training. Re-check --base matches what "
            f"finetune_ahc_vlm.py used.")
    print(f"[eval] adapter attached: {n_lora} LoRA modules injected")
    proc = load_proc(a.adapter)
    results["finetuned"] = score(tuned_model, proc, test_rows,
                                 f"fine-tuned ({a.adapter})", a.frames_root, a.max_pixels)

    if "zero_shot_base" in results:
        b, f = results["zero_shot_base"]["class_acc"], results["finetuned"]["class_acc"]
        n = len(test_rows)
        gap_examples = abs(f - b) * n
        winner = "fine-tuned" if f > b else ("zero-shot base" if b > f else "tie")
        print(f"\n[verdict] {winner} leads on class accuracy "
              f"(finetuned={f:.3f} vs base={b:.3f}) on n={n}")
        # Declaring a winner from a 1-2 example gap on 32 samples is noise,
        # not a result. Say so rather than letting the label stand alone.
        if gap_examples < 3:
            print(f"[verdict] the gap is only {gap_examples:.0f} example(s) of "
                  f"{n} - that is within noise at this sample size. Do NOT "
                  f"present either as the winner; report both numbers and the n.")
        elif f <= b:
            print("[verdict] fine-tune did NOT beat zero-shot prompting - "
                  "report both numbers, do not present the adapter as the pick.")

    import os
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[eval] full results -> {a.out}")


if __name__ == "__main__":
    main()
