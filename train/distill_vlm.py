"""P18 step 3: LoRA/QLoRA distillation of the small VLM judge on teacher
pseudo-labels (train/label_pseudo.py's output). Uses Unsloth's FastVisionModel
against the SAME model id already in config.yaml's vlm.model_id, so the
resulting adapter drops onto exactly the model pipeline/vlm_judge.py loads.

Kaggle-only, GPU required - do not run locally (CLAUDE.md: no local GPU).
This is the highest-risk phase in ENHANCEMENT_PLAN.md: do not start it
before Phase 5/6 (detector fine-tune + A/B) and Phase 13's dry runs are
solid, and do not skip the mandatory before/after comparison in step 5.

    python train/distill_vlm.py --data train/pseudo_data/pseudo_labels.jsonl \
        --base Qwen/Qwen2.5-VL-3B-Instruct --out weights/qwen_distilled_lora --epochs 2
"""
import argparse
import json


def load_dataset(jsonl_path):
    """One record per labelled event, using the SAME prompt text the teacher
    was shown - the student must learn the identical input, not a
    paraphrased one, or the distillation isn't measuring what it claims to."""
    from PIL import Image
    records = []
    for line in open(jsonl_path, encoding="utf-8"):
        row = json.loads(line)
        images = [Image.open(p).convert("RGB") for p in row["image_paths"]]
        target = json.dumps(row["verdict"])
        records.append({
            "images": images,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": row["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ],
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/pseudo_data/pseudo_labels.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="must match config.yaml's vlm.model_id - the adapter "
                         "is loaded on top of exactly this base at inference")
    ap.add_argument("--out", default="weights/qwen_distilled_lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    a = ap.parse_args()

    records = load_dataset(a.data)
    if len(records) < 20:
        print(f"[distill] WARNING: only {len(records)} labelled events - a "
              f"LoRA fine-tune on this few examples is unlikely to beat the "
              f"zero-shot prompted baseline. Step 5's mandatory A/B "
              f"(eval_run.py) will say either way; do not skip it because "
              f"the sample is small.")

    n_val = max(1, int(len(records) * a.val_fraction))
    train_records, val_records = records[:-n_val], records[-n_val:]

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastVisionModel.from_pretrained(
        a.base, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, r=a.r, lora_alpha=a.r * 2, lora_dropout=0.05,
        finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True)
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_records, eval_dataset=val_records,
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            num_train_epochs=a.epochs, learning_rate=a.lr,
            warmup_ratio=0.05, logging_steps=5, eval_strategy="epoch",
            output_dir=a.out, save_strategy="epoch", report_to="none",
            remove_unused_columns=False, dataset_text_field="",
            max_seq_length=2048,
        ),
    )
    trainer.train()

    model.save_pretrained(a.out)
    tokenizer.save_pretrained(a.out)
    print(f"[distill] LoRA adapter saved to {a.out}")
    print(f"[distill] MANDATORY next step: run both vlm.backend=qwen and "
          f"vlm.backend=qwen_distilled through eval_run.py on the SAME "
          f"held-out clips and compare AUC/F1/latency - "
          f"ENHANCEMENT_PLAN.md Phase 18 step 5. If distilled loses or ties, "
          f"keep the prompted version and say so; do not present this "
          f"adapter as a differentiator without that comparison.")


if __name__ == "__main__":
    main()
