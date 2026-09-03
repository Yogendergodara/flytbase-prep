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
        return None, None, "no normal bank in scene_fit.json - run fit.py first"
    return np.load(bank_npy), bank_meta, None


_CACHE = {}


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_encoder(encoder_tag):
    """Load once, reuse. score_frame_novelty is called per candidate event, so
    reloading SigLIP from disk each time meant a full model load per event.
    Also moves to GPU - fit.py built the bank on GPU while queries were being
    encoded on CPU."""
    if encoder_tag in _CACHE:
        return _CACHE[encoder_tag]
    import open_clip
    name, pretrained = encoder_tag.split("/")
    model, _, pre = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model.eval()                       # or embeddings drift between runs
    dev = _device()
    model = model.to(dev)
    tokenizer = open_clip.get_tokenizer(name)
    _CACHE[encoder_tag] = (model, pre, tokenizer, dev)
    return _CACHE[encoder_tag]


def novelty(frame_emb, bank, k=5):
    """1 - mean cosine to the k nearest normal-bank frames.

    High = unlike anything fit.py saw as 'normal'. This is the other half of
    the dual-use index: fit.py builds the bank once, retrieve.search() reads
    it for text queries, this reads it for a zero-shot anomaly score - same
    embeddings, no extra model, no training (the WinCLIP/AnomalyCLIP framing).
    """
    k = min(k, len(bank))
    sims = bank @ frame_emb
    top = np.sort(sims)[-k:]
    return float(max(0.0, min(1.0, 1.0 - top.mean())))


def score_frame_novelty(frame_rgb, fit_json="out/scene_fit.json",
                         bank_npy="out/normal_bank.npy", k=5):
    """Encode one RGB frame with the SAME encoder the bank was built with,
    and return (score, None) - or (None, reason) if no bank exists."""
    embs, bank, err = load_bank(fit_json, bank_npy)
    if err:
        return None, err
    import torch
    from PIL import Image
    model, pre, _, dev = _load_encoder(bank["encoder"])
    with torch.inference_mode():
        e = model.encode_image(pre(Image.fromarray(frame_rgb)).unsqueeze(0).to(dev))
    e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    return novelty(e, embs, k), None


def search(query, fit_json="out/scene_fit.json", bank_npy="out/normal_bank.npy",
           src_fps=None, top_k=5):
    """Returns ranked frames for a text query, or an explicit refusal.

    `frame_idx` holds SOURCE frame numbers, so seconds needs the video's own
    fps - never the sampled eff_fps, which would overstate every timestamp.
    """
    embs, bank, err = load_bank(fit_json, bank_npy)
    if err:
        return {"results": [], "reason": err}
    frame_idx = bank["frame_idx"]
    fps = src_fps or bank.get("src_fps") or 0.0

    import torch
    model, _, tokenizer, dev = _load_encoder(bank["encoder"])
    with torch.inference_mode():
        q = model.encode_text(tokenizer([query]).to(dev))
        q = (q / q.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

    sims = embs @ q
    order = np.argsort(-sims)[:top_k]
    results = []
    for rank, i in enumerate(order):
        frame_number = frame_idx[i]
        entry = {"rank": rank + 1, "frame_index": int(frame_number),
                 "similarity": round(float(sims[i]), 4)}
        if fps:
            entry["approx_seconds"] = round(frame_number / fps, 2)
        results.append(entry)
    return {"results": results, "reason": None}
