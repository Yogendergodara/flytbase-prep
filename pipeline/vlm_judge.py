"""Stage 4: only candidate events reach a VLM. Constrained JSON out.

Cost control: N events x frames_per_event images, NOT every frame.
`backend: none` returns a neutral verdict so the whole pipeline runs on CPU.
"""
import json, re, cv2

PROMPT = """You are reviewing a short frame sequence from a fixed security camera / drone feed.

Detector facts (trust these, they are measured):
- object class: {cls}
- track id: {tid}
- event flagged by geometry: {kind}
- measurements: {facts}
- window: {t0:.1f}s to {t1:.1f}s

Decide whether this is genuinely anomalous for this scene, or ordinary activity
that the geometry rule over-triggered on.

Reply with ONLY this JSON:
{{"anomalous": true|false, "score": 0.0-1.0, "label": "<3-6 words>", "why": "<one sentence citing what you see>"}}"""


def extract_frames(video_path, t0, t1, k):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times = [t0 + (t1 - t0) * i / max(1, k - 1) for i in range(k)]
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


class NoopJudge:
    def judge(self, ev, frames):
        return {"anomalous": None, "score": None, "label": "not judged",
                "why": "vlm.backend=none - geometric score only"}


class QwenJudge:
    def __init__(self, cfg):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        v = cfg["vlm"]
        self.cfg = v
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            v["model_id"], torch_dtype="auto", device_map="auto")
        self.proc = AutoProcessor.from_pretrained(v["model_id"], max_pixels=v["max_pixels"])

    def judge(self, ev, frames):
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": PROMPT.format(
            cls=ev.cls, tid=ev.track_id, kind=ev.kind, facts=ev.facts,
            t0=ev.t_start, t1=ev.t_end)})
        msgs = [{"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
        ids = self.model.generate(**inputs, max_new_tokens=self.cfg["max_new_tokens"],
                                  do_sample=False)
        raw = self.proc.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0]
        return _parse(raw)


def _parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"anomalous": None, "score": None, "label": "unparsed",
                "why": raw[:160]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"anomalous": None, "score": None, "label": "unparsed", "why": raw[:160]}
    s = d.get("score")
    d["score"] = None if s is None else max(0.0, min(1.0, float(s)))
    return d


def build_judge(cfg):
    b = cfg["vlm"]["backend"]
    if b == "none":
        return NoopJudge()
    if b == "qwen":
        return QwenJudge(cfg)
    raise ValueError("unknown vlm backend: " + b)
