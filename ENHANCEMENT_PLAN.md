# Enhancement Plan — Phases 14-19

Everything below is **additive** to `IMPLEMENTATION.md`'s Phases 1-13. Nothing
here changes the existing architecture (cascade: detector/tracker → arithmetic
events → event-triggered VLM → fuse). These are the gaps identified by
comparing against an external strategy plan for the same hackathon — items
that plan had and this repo didn't. Same format as `IMPLEMENTATION.md`: what
you do, which files, how, done when.

**Gate:** Phase 5 (Kaggle fine-tune) must be resolved — weights downloaded and
A/B'd (Phase 6) — before spending time here. None of this fixes a missing
detector; all of it is pitch-strength and robustness on top of a working
pipeline.

---

## Priority (merges into IMPLEMENTATION.md's table)

| Priority | Phase | Reason |
|---|---|---|
| High | 14 (economics), 19 (logistics) | Cheap, no code risk, directly strengthens the demo/pitch |
| Medium | 15 (citations), 16 (fallback verify) | Cheap, de-risks Q&A and Kaggle congestion |
| Low / stretch | 17 (auto zone), 18 (distillation) | Real time cost, real risk — only if 1-13 are done with hours to spare |

If Saturday morning arrives and 14-19 aren't done, **skip straight to Phase 13
dry runs.** A working demo without a headline economics number beats a broken
demo with one.

---

## Phase 14 — Economics headline metric *(local + Kaggle, ~1 h)*

**Why:** FlytBase's own product (AI-R Edge) is sold on "process locally, cut
streaming cost." Reporting "N feeds per GPU" speaks their language directly
instead of burying it in FPS logs.

**Files:** `pipeline/evaluate.py`, `run.py`, new `scripts/economics.py`

**What you do**
1. `run.py` already times each stage end-to-end (Phase 9, F3). Confirm
   `out/alerts.json` carries per-stage wall-clock (`eff_fps`, and the
   end-to-end FPS line Phase 9 added). If Phase 9 wasn't finished, do it
   first — this phase depends on real numbers, not estimates.
2. New `scripts/economics.py`:
   ```python
   import json, argparse, torch

   ap = argparse.ArgumentParser()
   ap.add_argument("--alerts", default="out/alerts.json")
   ap.add_argument("--gpu-mem-gb", type=float, default=None,
                   help="override; else read torch.cuda.max_memory_allocated")
   a = ap.parse_args()

   d = json.load(open(a.alerts, encoding="utf-8"))
   fps = d["eff_fps"]
   mem_gb = a.gpu_mem_gb or (torch.cuda.max_memory_allocated() / 1e9
                              if torch.cuda.is_available() else None)

   # A100-class card as the reference point; swap for what you actually ran on
   REFERENCE_GPU_MEM_GB = 40.0
   feeds_per_gpu = int(REFERENCE_GPU_MEM_GB // mem_gb) if mem_gb else None

   print(f"Pipeline: {fps:.1f} FPS sustained, end-to-end, single feed")
   print(f"Peak GPU memory: {mem_gb:.2f} GB" if mem_gb else "GPU memory: n/a (CPU run)")
   if feeds_per_gpu:
       print(f"Extrapolated: ~{feeds_per_gpu} concurrent drone feeds per "
             f"{REFERENCE_GPU_MEM_GB:.0f}GB GPU (memory-bound estimate)")
   print(f"VLM call rate: read from alerts.json n_frames vs len(alerts) "
         f"to report '% of frames that reached the VLM'")
   ```
3. Add the VLM-call-rate line explicitly: `len(alerts) frames judged /
   n_frames total * 100` — this is the "1-5% of frames" claim from the brief.
   Compute it in the same script from `d["alerts"]` and `d["n_frames"]`.
4. Record the numbers from a real Kaggle run (T4/P100 memory profile) once
   Phase 5/6 finish. Don't estimate GPU memory — measure it with
   `torch.cuda.max_memory_allocated()` reset via
   `torch.cuda.reset_peak_memory_stats()` at the start of `run.py`.

**Done when:** one paragraph exists with three numbers — sustained FPS,
peak GPU memory, extrapolated feeds-per-GPU — computed from a real run, not
guessed. Put it on its own slide, first, before accuracy numbers.

---

## Phase 15 — Prior-art citations *(local, ~30 min, no code)*

**Why:** "Why not just a detector?" is the obvious judge question for a VLM-
based system. Answering with a citation beats hand-waving.

**Files:** none (pitch material only) — optionally a `REFERENCES.md`

**What you do:** add one paragraph and a short reference list to your pitch
deck / `README.md`:
- **AnyAnomaly** (WACV 2026) — zero-shot, LVLM-based, no fine-tuning —
  nearest published architecture to this pipeline's Stage-3 judge.
- **FADE** (BMVC 2024) — training-free few/zero-shot VLM anomaly detection —
  supports the "prompt, don't fine-tune" governing rule as a validated
  approach, not an improvised shortcut.
- **WinCLIP / AnomalyCLIP** — the same zero-shot-embedding-bank framing your
  `pipeline/retrieve.py` novelty score already implements. Name it explicitly:
  "our normal-bank novelty score is the WinCLIP/AnomalyCLIP framing."
  Cite `Awesome-Anomaly-Detection-Foundation-Models` as the source list.
- **Holmes-VAU** (CVPR'25) — long-term anomaly *understanding*, cited to
  justify why the VLM returns free-text reasoning (`vlm_judge.py`'s
  `reason` field) rather than a closed label.
- **Open-Vocabulary VAD** (arXiv 2311.07042) — justifies why `open_vocab.py`
  exists at all: fixed class lists cannot cover a novel hazard.

**Done when:** each citation maps to one specific design decision already in
the repo (not generic name-dropping) — you should be able to say "we did X,
here's the paper that validates it" for all five.

---

## Phase 16 — Verify the CPU/edge fallback *(Kaggle + local, ~45 min)*

**Why:** repo is Kaggle-only today with no fallback if cloud GPU access is
congested. `vlm.backend: smolvlm` already exists in `config.yaml` and
`pipeline/vlm_judge.py` — it has never been run.

**Files:** `pipeline/vlm_judge.py`, `config.yaml` (no changes expected, this
is a verification phase)

**What you do**
1. On Kaggle: `python run.py --video data/sample.mp4 --set vlm.backend=smolvlm`
   and confirm it produces `out/alerts.json` with non-null verdicts. Compare
   latency against the `qwen` backend on the same clip — record both numbers.
2. Test true CPU fallback for the venue-congestion scenario: `--set
   vlm.backend=smolvlm detector.device=cpu detector.half=false` on a local
   machine (or a Kaggle CPU-only session) and confirm it still runs, just
   slower. `detector.half` must be false on CPU — `config.yaml`'s own comment
   already flags `half: true` as CPU-incompatible.
3. If `smolvlm` underperforms badly on drone footage's small/dense objects,
   note it as a known limitation rather than debugging it under time pressure
   — the fallback's job is "still produces an answer," not "matches Qwen's
   accuracy."
4. Optional hardening: cache `HuggingFaceTB/SmolVLM2-2.2B-Instruct` weights
   locally now (`huggingface-cli download`) so the fallback doesn't need
   venue wifi either.

**Done when:** you have run both backends on the same clip, have latency
numbers for both, and know — because you tested it, not because you assumed
it — that the fallback path actually produces alerts.

---

## Phase 17 — Zero-shot zone auto-calibration *(optional stretch, ~1.5 h)*

**Why:** `zones.py` requires manually clicking polygons per camera. A one-time
CLIP zero-shot scene classification removes that step for a feed you've never
seen before a live demo.

**Files:** new `pipeline/zone_classify.py`, `zones.py`, `config.yaml`

**What you do**
1. `pipeline/zone_classify.py`:
   ```python
   # Reuses the encoder fit.py already loads (SigLIP / open_clip ViT-B-32) -
   # do not add a second CLIP dependency.
   ZONE_PROMPTS = ["a driving lane", "a parking lot", "a footpath / sidewalk",
                   "a restricted / fenced area", "open ground"]

   def classify_regions(frame_rgb, grid=(4, 4), encoder=None):
       """Split the frame into a grid, classify each cell against
       ZONE_PROMPTS, return {cell_bbox: zone_label}. One-time per camera."""
       ...
   ```
2. Wire it as a *suggestion*, not an automatic override: `zones.py --auto`
   proposes polygons per grid cell above a confidence threshold; the operator
   still confirms/adjusts before saving to `config.yaml`'s
   `events.restricted_zones`. Never silently replace a hand-drawn zone.
3. Config addition:
   ```yaml
   zones:
     auto_classify: false     # KNOB: --set zones.auto_classify=true
     grid: [4, 4]
     confidence_floor: 0.3
   ```

**Done when:** `python zones.py --auto --video x.mp4` prints proposed zone
labels per grid cell, and a human still has to approve before they land in
`config.yaml`. **Cut this phase first if time is short** — manual zones.py
already works and is more precise for a demo with one known venue camera.

---

## Phase 18 — Optional stretch: distill a large VLM into the judge *(Kaggle, ~4-6 h GPU, high risk)*

**Why:** the brief explicitly invites this ("distill a larger model"). It
directly contradicts the repo's current governing rule ("never fine-tune the
VLM; prompt it") — that rule stays the *default*; this phase only replaces it
if it demonstrably wins.

**Do not start this before Phase 5, 6, and 13's dry runs are solid.** This
is the single most time-and-risk-expensive addition in this document.

**Files:** new `train/distill_vlm.py`, new `train/label_pseudo.py`,
`pipeline/vlm_judge.py` (new backend), `config.yaml`

**What you do**
1. **Feasibility check first (15 min, before committing GPU time):** decide
   the teacher model. Qwen2.5-VL-72B needs ~140GB+ VRAM even 4-bit —
   not running that locally or on a Kaggle single-GPU session. Use an API
   model instead (e.g. a hosted VLM endpoint) for teacher labeling, budget
   the API cost and rate limit for however many clips you plan to label, and
   confirm you actually have that access **before** writing the labeling
   script. If no API budget/access exists, stop here and skip this phase —
   don't discover this at the venue.
2. `train/label_pseudo.py` — reuses `pipeline/events.py`'s existing candidate
   events (the same ~8-per-video candidates your geometric stage already
   finds) as the sampling set. Do **not** label random frames; label the same
   candidate-event windows the small model will be judged on, using the same
   `pipeline/vlm_judge.py:PROMPT` template so the pseudo-labels and the
   fine-tuned model's inputs match exactly:
   ```python
   # For each CandidateEvent from a batch run with vlm.backend=none:
   #   call the teacher (API) with (frame, facts) -> {anomalous, severity, reason}
   #   write {image_path, context_facts, verdict_json} to train/pseudo_labels.jsonl
   ```
3. `train/distill_vlm.py` — Unsloth LoRA/QLoRA fine-tune of
   `Qwen/Qwen2.5-VL-3B-Instruct` (the same model id already in
   `config.yaml`'s `vlm.model_id`) on `pseudo_labels.jsonl`. Follow the
   Unsloth vision fine-tuning guide's data format exactly; do not improvise
   the chat template.
4. New backend in `pipeline/vlm_judge.py`: `qwen_distilled` — same interface
   as `QwenJudge`, loads the LoRA adapter on top of the base model.
5. **Mandatory comparison, not optional:** run both `qwen` (prompted) and
   `qwen_distilled` on the same held-out clips through `eval_run.py`. Report
   both AUC/F1/latency numbers. If distilled loses or ties, **say so and keep
   the prompted version** — same rule as Phase 6's A/B, applied here.

**Done when:** you have a real before/after table (prompted vs. distilled) on
held-out clips, not a demo of the distilled model alone. If you cannot
produce that comparison before the deadline, do not present distillation as
a claim at all — an untested LoRA adapter is a liability on stage, not a
differentiator.

---

## Phase 19 — Day-before logistics checklist *(no code, ~1 h to execute)*

**Why:** organizers explicitly ask teams to arrive with setup and model
access already tested. This is pure discipline, not engineering — but
skipping it costs the first 90 minutes of build time on-site.

**Do the night before, in this order:**
1. `git push` everything — Kaggle re-clones every session, uncommitted local
   edits do not exist there (already a rule in `IMPLEMENTATION.md` Part 0).
2. Cache every model weight locally / in a Kaggle dataset so nothing depends
   on venue wifi: `yolo11s.pt` / fine-tuned `best.pt`, `Qwen2.5-VL-3B-Instruct`,
   `SmolVLM2-2.2B-Instruct` (Phase 16), open-vocab `yolov8l-worldv2.pt` if
   `open_vocab.backend=yoloworld` is planned, OSNet re-ID weights if
   `reid.backend=osnet` is planned.
3. Confirm HF auth token and any API keys (Phase 18's teacher model, if
   pursued) work from a fresh shell — don't discover an expired token at
   9:05 AM.
4. Run all four presets (`day`, `night`, `fast`, `accurate`) once each from a
   cold shell — this is already Phase 13 step 1, just do it *before* Saturday
   morning, not as part of Saturday morning.
5. **Final check: disconnect wifi, run one preset end to end.** Anything that
   fails here is a silent network dependency you didn't know about — fix it
   before the day, not during it.

**Done when:** every item above has a checkmark from *last night*, not from
memory. If #5 fails, that's the single most valuable bug you'll find all
week — it fails silently in a live room otherwise.

---

## Summary: what changes vs. what doesn't

**Unchanged:** the cascade architecture, the governing rule ("never fine-tune
the VLM" stays default unless Phase 18 proves otherwise), all of Phases 1-13.

**Added:** an economics headline number (14), pitch citations (15), a
verified fallback path (16), optional auto zone calibration (17), an
optional and strictly gated distillation experiment (18), and a pre-flight
checklist (19).

**Nothing here should be started before Phase 5's weights exist and Phase 6's
A/B is done** — an economics number or a distillation experiment on top of an
unverified detector is a number built on sand.
