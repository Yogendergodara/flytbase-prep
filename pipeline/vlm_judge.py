"""Stage 4: only candidate events reach a VLM. Constrained JSON out.

Cost control: N events x frames_per_event images, NOT every frame.
`backend: none` returns a neutral verdict so the whole pipeline runs on CPU.
"""
import json, re, time, cv2
from pipeline.video_io import seek_exact

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

FORENSIC_PROMPT = """You are writing a short forensic summary of a monitored
window, based on multiple already-flagged events. Each event's facts were
measured by the detector/tracker; trust them. Do not invent anything not
present in the facts below - if the facts don't say it, don't write it.

Events in this window ({t0:.1f}s to {t1:.1f}s):
{events}

Write ONE paragraph (3-5 sentences) describing what happened across these
events, citing timestamps explicitly."""


def _format_events(alerts):
    lines = []
    for a in alerts:
        line = (f"- [{a['t_start']:.1f}s-{a['t_end']:.1f}s] {a['kind']} "
                f"(track {a['track_id']}, score {a['score']:.2f}): {a['facts']}")
        if a.get("why"):
            line += f" - VLM said: {a['why']}"
        lines.append(line)
    return "\n".join(lines)


def extract_frames(video_path, t0, t1, k, cap=None):
    """k frames spanning [t0, t1] as RGB. k=1 returns the MIDDLE frame, not
    t0 - a single frame taken from the very start of a window is the least
    informative one available. Pass `cap` to reuse an open capture."""
    own = cap is None
    if own:
        cap = cv2.VideoCapture(video_path)
    if k == 1:
        times = [(t0 + t1) / 2.0]
    else:
        times = [t0 + (t1 - t0) * i / (k - 1) for i in range(k)]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    for t in times:
        ok, f = seek_exact(cap, t, fps)
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if own:
        cap.release()
    return frames


class NoopJudge:
    def judge(self, ev, frames):
        return {"anomalous": None, "score": None, "label": "not judged",
                "why": "vlm.backend=none - geometric score only"}

    def summarize(self, alerts, t0, t1):
        return None, "vlm.backend=none - no forensic summary"


def _quant_config(v):
    """4-bit NF4 - Qwen2.5-VL-3B drops ~7 GB fp16 -> ~2.5 GB. Returns None
    (plain fp16/bf16) when 4-bit is off, bitsandbytes is absent, or there is
    no CUDA device: bitsandbytes has no usable CPU backend, so silently
    'succeeding' on CPU would just crash deeper in."""
    if not v.get("load_4bit"):
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("[vlm] bitsandbytes/transformers 4-bit unavailable - loading fp16")
        return None
    if not torch.cuda.is_available():
        print("[vlm] no CUDA - 4-bit skipped, loading on CPU (slow)")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class _HFVisionJudge:
    """Shared judge body. Subclasses only supply the model class + id, so the
    Qwen path and the SmolVLM2 fallback cannot drift apart."""

    def _load(self, cfg, model_cls, model_id):
        from transformers import AutoProcessor
        v = cfg["vlm"]
        self.cfg = v
        kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        qc = _quant_config(v)
        if qc is not None:
            kwargs["quantization_config"] = qc
            kwargs.pop("torch_dtype")          # compute dtype comes from the config
        from pipeline.retry import retry
        self.model = retry(model_cls.from_pretrained, model_id, attempts=3,
                          label=f"load {model_id}", **kwargs)
        self.model.eval()                      # embeddings must not drift between runs
        self.proc = AutoProcessor.from_pretrained(model_id, max_pixels=v["max_pixels"])
        self.quantized = qc is not None

    def _generate(self, msgs, imgs, max_new_tokens):
        import torch
        chat = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        # images=None (not []) for the text-only path - an empty list makes
        # some processors build a zero-length pixel tensor and fail in the model
        inputs = self.proc(text=[chat], images=imgs or None,
                           return_tensors="pt").to(self.model.device)
        # inference_mode, not just no_grad: forgetting it builds an autograd
        # graph nothing consumes and OOMs a GPU with room to spare
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=self.cfg.get("temperature", 0.0) > 0,
                temperature=self.cfg.get("temperature", 0.0) or None)
        return self.proc.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]

    def judge(self, ev, frames):
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": PROMPT.format(
            cls=ev.cls, tid=ev.track_id, kind=ev.kind, facts=ev.facts,
            t0=ev.t_start, t1=ev.t_end)})
        raw = self._generate([{"role": "user", "content": content}], imgs,
                             self.cfg["max_new_tokens"])
        return _parse(raw)

    def summarize(self, alerts, t0, t1):
        """F6: one paragraph across several already-flagged events. Text only -
        no images, no extra frame extraction, cheap on top of the per-event
        judging that already ran."""
        if not alerts:
            return None, "no alerts in this window"
        text = FORENSIC_PROMPT.format(t0=t0, t1=t1, events=_format_events(alerts))
        raw = self._generate([{"role": "user", "content": [{"type": "text", "text": text}]}],
                             None, self.cfg["max_new_tokens"] * 2)
        return raw.strip(), None


class QwenJudge(_HFVisionJudge):
    def __init__(self, cfg):
        from transformers import Qwen2_5_VLForConditionalGeneration
        self._load(cfg, Qwen2_5_VLForConditionalGeneration, cfg["vlm"]["model_id"])
        print(f"[vlm] Qwen2.5-VL loaded, 4-bit={self.quantized}")


class SmolVLMJudge(_HFVisionJudge):
    """The plan's documented fallback. Same prompt, same JSON contract."""

    def __init__(self, cfg):
        from transformers import AutoModelForImageTextToText
        model_id = cfg["vlm"].get("smolvlm_model_id",
                                   "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        self._load(cfg, AutoModelForImageTextToText, model_id)
        print(f"[vlm] SmolVLM2 loaded, 4-bit={self.quantized}")


class QwenDistilledJudge(_HFVisionJudge):
    """P18: same interface as QwenJudge - base model + a LoRA adapter fine-
    tuned on teacher pseudo-labels (train/distill_vlm.py). Opt-in only. The
    governing rule ('never fine-tune the VLM; prompt it', CLAUDE.md) stays
    the default; this backend is meant to be selected only AFTER Phase 18's
    mandatory A/B (eval_run.py, prompted vs. distilled on held-out clips)
    shows it wins - never as an unverified swap-in."""

    def __init__(self, cfg):
        from transformers import Qwen2_5_VLForConditionalGeneration
        adapter_path = cfg["vlm"].get("distilled_adapter_path")
        if not adapter_path:
            raise ValueError(
                "vlm.backend=qwen_distilled needs vlm.distilled_adapter_path "
                "set to train/distill_vlm.py's --out directory")
        self._load(cfg, Qwen2_5_VLForConditionalGeneration, cfg["vlm"]["model_id"])
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        print(f"[vlm] Qwen2.5-VL + distilled LoRA adapter loaded from {adapter_path}")


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
    if b == "smolvlm":
        return SmolVLMJudge(cfg)
    if b == "qwen_distilled":
        return QwenDistilledJudge(cfg)
    raise ValueError("unknown vlm backend: " + b)


def judge_event_safe(judge, video_path, ev, cfg, cap=None, novelty_fn=None):
    """#G-A: one exception boundary around frame extraction + novelty scoring
    + the judge call, shared by run.py (batch) and pipeline/stream.py (live).
    Neither call site previously had a try/except here: a VLM OOM, a corrupt
    frame, or a decode error aborted the WHOLE run instead of degrading just
    this one event. The noop judge already returns the right shape
    (`score: None` -> fuse_score's `geometric_only` path) - this makes sure a
    genuine failure produces that same shape instead of a crash.

    Returns (verdict, latency_ms).
    """
    t0 = time.time()
    try:
        frames = ([] if cfg["vlm"]["backend"] == "none" else
                  extract_frames(video_path, ev.t_start, ev.t_end,
                                 cfg["vlm"]["frames_per_event"], cap=cap))
        if novelty_fn:
            middle = extract_frames(video_path, ev.t_start, ev.t_end, 1, cap=cap)
            if middle:
                nov, err = novelty_fn(middle[0])
                if err is None:
                    ev.facts["novelty"] = round(nov, 3)
        verdict = judge.judge(ev, frames)
    except Exception as e:
        print(f"[judge] event at {ev.t_start:.1f}s ({ev.kind}) degraded to "
              f"geometric_only - {type(e).__name__}: {e}")
        verdict = {"anomalous": None, "score": None, "label": "stage_error",
                  "why": f"judge stage failed ({type(e).__name__}) - "
                         f"geometric score only"}
    return verdict, (time.time() - t0) * 1000
