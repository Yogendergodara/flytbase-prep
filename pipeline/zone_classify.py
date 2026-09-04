"""P17: one-time, zero-shot scene classification per camera view - "is this
grid cell a driving lane, a parking lot, a footpath, or a restricted area?"
- so a new feed doesn't need someone to click zone polygons by hand before a
demo starts.

Reuses the SAME CLIP-family encoder fit.py already loads for the normal bank
(SigLIP ViT-B-16, falling back to ViT-B-32/openai) - no second CLIP
dependency, no extra model download.

This is a SUGGESTION tool. It proposes; `zones.py --auto` still requires the
operator to press `s` to save, same as manual clicking. Never silently
overwrites a hand-drawn zone.
"""
import numpy as np

ZONE_PROMPTS = [
    "a driving lane with vehicle traffic",
    "a parking lot or parking spot",
    "a footpath or sidewalk for pedestrians",
    "a restricted or fenced-off area",
    "open ground with no clear purpose",
]

# which labels are worth turning into a restricted_zones polygon by default -
# "footpath" and "open ground" are informational only, not intrusion-worthy
_INTRUSION_WORTHY = {
    "a driving lane with vehicle traffic",
    "a restricted or fenced-off area",
}

_CACHE = {}


def _load_encoder():
    """Same fallback order as fit.py:fit_embeddings - SigLIP first, ViT-B-32
    if unavailable. Cached module-level so repeated calls in one process
    don't reload the model."""
    if "encoder" in _CACHE:
        return _CACHE["encoder"]
    import open_clip
    import torch
    try:
        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", pretrained="webli")
        tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
        tag = "ViT-B-16-SigLIP/webli"
    except Exception:
        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tag = "ViT-B-32/openai"
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    _CACHE["encoder"] = (model, pre, tokenizer, dev, tag)
    return _CACHE["encoder"]


def classify_regions(frame_rgb, grid=(4, 4), confidence_floor=0.3):
    """Split frame_rgb (H, W, 3) into a grid, zero-shot classify each cell
    against ZONE_PROMPTS. Returns a list of
    {bbox: [x0,y0,x1,y1], label, confidence, intrusion_worthy: bool}
    for cells at or above confidence_floor. Below the floor, a cell is
    omitted entirely - never guessed as a default label."""
    import torch
    from PIL import Image

    model, pre, tokenizer, dev, tag = _load_encoder()
    h, w = frame_rgb.shape[:2]
    gx, gy = grid
    cell_w, cell_h = w // gx, h // gy

    with torch.inference_mode():
        text_feats = model.encode_text(tokenizer(ZONE_PROMPTS).to(dev))
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        results = []
        for j in range(gy):
            for i in range(gx):
                x0, y0 = i * cell_w, j * cell_h
                x1 = w if i == gx - 1 else x0 + cell_w
                y1 = h if j == gy - 1 else y0 + cell_h
                cell = frame_rgb[y0:y1, x0:x1]
                if cell.size == 0:
                    continue
                im = pre(Image.fromarray(cell)).unsqueeze(0).to(dev)
                feat = model.encode_image(im)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                sims = (feat @ text_feats.T).cpu().numpy()[0]
                # SigLIP/CLIP logits aren't a calibrated probability - softmax
                # over the closed prompt set gives a relative confidence,
                # which is all "confidence_floor" needs to mean here
                probs = np.exp(sims * 100) / np.exp(sims * 100).sum()
                best = int(np.argmax(probs))
                conf = float(probs[best])
                if conf < confidence_floor:
                    continue
                label = ZONE_PROMPTS[best]
                results.append({
                    "bbox": [x0, y0, x1, y1],
                    "label": label,
                    "confidence": round(conf, 3),
                    "intrusion_worthy": label in _INTRUSION_WORTHY,
                })
    return results
