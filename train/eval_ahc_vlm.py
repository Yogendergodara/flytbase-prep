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


def predict(model, tokenizer, row, frames_root, max_new_tokens=150):
    from pathlib import Path
    from PIL import Image
    import torch
    # same relative-path convention as finetune_ahc_vlm.py's to_record -
    # the manifest and the AHC_frames folder travel to Kaggle separately
    paths = [p if Path(p).is_absolute() else str(Path(frames_root) / p) for p in row["frame_paths"]]
    images = [Image.open(p).convert("RGB") for p in paths]
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": AHC_PROMPT.format(classes=", ".join(AHC_CLASSES))})
    messages = [{"role": "user", "content": content}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[chat], images=images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = tokenizer.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return _parse(raw)


def score(model, tokenizer, test_rows, label, frames_root):
    n_correct_class = n_correct_anomaly = n_parsed = 0
    per_class = defaultdict(lambda: [0, 0])  # [correct, total]
    confusion = Counter()
    for row in test_rows:
        pred = predict(model, tokenizer, row, frames_root)
        truth_cls = row["class_name"]
        per_class[truth_cls][1] += 1
        if pred is None:
            confusion[(truth_cls, "UNPARSED")] += 1
            continue
        n_parsed += 1
        pred_cls = pred.get("class_name")
        pred_anom = pred.get("is_anomaly")
        confusion[(truth_cls, pred_cls)] += 1
        if pred_cls == truth_cls:
            n_correct_class += 1
            per_class[truth_cls][0] += 1
        if pred_anom == row["is_anomaly"]:
            n_correct_anomaly += 1

    n = len(test_rows)
    print(f"\n=== {label} ===")
    print(f"parsed: {n_parsed}/{n}")
    print(f"class accuracy: {n_correct_class}/{n} = {n_correct_class/n:.3f}")
    print(f"is_anomaly accuracy: {n_correct_anomaly}/{n} = {n_correct_anomaly/n:.3f}")
    print("per-class recall:")
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
    ap.add_argument("--frames-root", default="datasets/AHC_frames",
                    help="where AHC_frames actually landed on THIS machine/Kaggle "
                         "session - same convention as finetune_ahc_vlm.py")
    a = ap.parse_args()

    test_rows = load_test_rows(a.manifest)
    print(f"[eval] {len(test_rows)} held-out test events (public test set, "
          f"never trained on)")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    results = {}
    if not a.skip_base:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            a.base, device_map="auto", torch_dtype="auto")
        base_model.eval()
        proc = AutoProcessor.from_pretrained(a.base)
        results["zero_shot_base"] = score(base_model, proc, test_rows, "zero-shot base (no fine-tune)", a.frames_root)
        del base_model

    tuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.base, device_map="auto", torch_dtype="auto")
    tuned_model = PeftModel.from_pretrained(tuned_model, a.adapter)
    tuned_model.eval()
    proc = AutoProcessor.from_pretrained(a.base)
    results["finetuned"] = score(tuned_model, proc, test_rows, f"fine-tuned ({a.adapter})", a.frames_root)

    if "zero_shot_base" in results:
        b, f = results["zero_shot_base"]["class_acc"], results["finetuned"]["class_acc"]
        winner = "fine-tuned" if f > b else ("zero-shot base" if b > f else "tie")
        print(f"\n[verdict] {winner} wins on class accuracy (finetuned={f:.3f} vs base={b:.3f})")
        if f <= b:
            print("[verdict] fine-tune did NOT beat zero-shot prompting - "
                  "report both numbers, do not present the adapter as the pick.")

    import os
    os.makedirs("out", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[eval] full results -> {a.out}")


if __name__ == "__main__":
    main()
