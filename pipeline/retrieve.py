"""Level 3: natural-language search over the footage.

Reuses the SAME embeddings fit.py already built for the normal-bank anomaly
score - one index, two features. No extra model, no extra pass over the
video.
"""
import json
import numpy as np


def load_bank(fit_json="out/scene_fit.json", bank_npy="out/normal_bank.npy"):
    meta = json.load(open(fit_json, encoding="utf-8"))
    bank_meta = meta.get("normal_bank")
    if not bank_meta:
        return None, None, None, "no normal bank in scene_fit.json - run fit.py first"
    embs = np.load(bank_npy)
    return embs, bank_meta["frame_idx"], bank_meta["encoder"], None


def _load_text_encoder(encoder_tag):
    import open_clip
    name, pretrained = encoder_tag.split("/")
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(name)
    model.eval()
    return model, tokenizer


def search(query, fit_json="out/scene_fit.json", bank_npy="out/normal_bank.npy",
           eff_fps=None, top_k=5):
    """Returns ranked frames for a text query, or an explicit refusal."""
    embs, frame_idx, encoder_tag, err = load_bank(fit_json, bank_npy)
    if err:
        return {"results": [], "reason": err}

    import torch
    model, tokenizer = _load_text_encoder(encoder_tag)
    with torch.no_grad():
        tok = tokenizer([query])
        q = model.encode_text(tok)
        q = (q / q.norm(dim=-1, keepdim=True)).numpy()[0]

    sims = embs @ q
    order = np.argsort(-sims)[:top_k]
    results = []
    for rank, i in enumerate(order):
        frame_number = frame_idx[i]
        entry = {"rank": rank + 1, "frame_index": int(frame_number),
                 "similarity": round(float(sims[i]), 4)}
        if eff_fps:
            entry["approx_seconds"] = round(frame_number / eff_fps, 2)
        results.append(entry)
    return {"results": results, "reason": None}
