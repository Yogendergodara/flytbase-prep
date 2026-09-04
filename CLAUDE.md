# FlytBase Hackathon — flytbase-prep

Small-VLM anomaly detection on drone video. Cascade: detector + tracker turns
~18,000 frames into ~40 tracklets → arithmetic rules flag ~8 candidate events
→ only those reach the VLM judge (Qwen2.5-VL-3B) → fuse + suppress → alerts.

## Working rules for Claude in this repo
- **No local GPU. Never `pip install`, train, or run inference here.** Every
  heavy command runs on **Kaggle**, launched by the user. This repo is source
  only; write and edit code, don't execute it.
- **Don't re-read the repo to establish state.** This file is the status
  source of truth. Open a file only when you're editing it.
- **Don't build verification loops.** No dataset and no GPU locally, so a
  local run proves nothing. Static review of the diff is the check.
- **Update this file** when code changes — tables, "Fixed, unverified", and
  "Remaining steps" alike — rather than re-deriving status by re-reading
  every module.

## Repo map and status
Nothing here has ever been executed. "Written" means the code exists and
reads correctly; it does not mean it has run.

**Phases 1-4 and 6-12 of IMPLEMENTATION.md are code-complete**, and a second
review pass fixed 14 bugs, 9 performance problems and the batch-only
architecture. See IMPLEMENTATION.md "Second review pass" for the itemised
list. Highlights worth remembering:
- **Streaming mode** (`pipeline/stream.py`, `--preset live`) emits alerts
  during the pass. The batch path could not alert until the file ended, which
  contradicted "in real time" on hours-long footage.
- **4-bit NF4** for the VLM (`vlm.load_4bit`), `torch.inference_mode()` on
  every generate call, `dwell_seconds` reduced from O(n²) to O(n).
- **Real hysteresis** — `clear_threshold` and `ema_alpha` now drive a
  raise/clear state machine; before, only `raise_threshold` was read.
- **`--fit-seconds` now works** (caps `video.max_seconds`); the normal bank
  samples quiet frames instead of uniformly over the anomalies.
- Event windows are clipped to `clip_seconds` around the peak rather than
  spanning the whole tracklet.

**Phase 5 (Kaggle training) is the only remaining gate** — nothing above can
be verified against real data/weights until it runs. P13 (harden/dry-run/
pitch) is procedure, not code.

**Third pass (design-doc audit + fresh scan) fixed:**
- **G-A** no exception boundary around per-event stages — a VLM OOM or
  corrupt frame aborted the whole run. Fixed via `vlm_judge.judge_event_safe`,
  shared by `run.py` and `stream.py`; degrades ONE event to `geometric_only`.
- **G-B** suppression hysteresis was hand-rolled twice (batch vs streaming),
  consistent by inspection only. Extracted to `fuse.HysteresisSuppressor`;
  both paths now call the same object — cannot drift apart by construction.
- **G-C (partial)** streaming didn't run open-vocab at all. Now wired
  (`StreamingPipeline.open_vocab`); re-ID still batch-only — real per-window
  cost tradeoff, not yet measured, documented in the class docstring.
- **G-H** pixel speed conflated object motion with camera pan (named, never
  fixed, in every earlier pass). `pipeline/camera_motion.py` — sparse
  optical-flow background-pan estimate, opt-in via
  `events.hover_pan_px_threshold` (null = off, zero cost), discounts
  `speed_anomaly` geo_score rather than dropping the event.
- Corrected a stale claim in the design-doc artifact: **G-G** (`clip_overlap`
  unread) was already fixed two passes ago — the key was removed, not left
  dead.

**Test suite now covers all of the above** — 26 tests, including a
regression that `judge_event_safe` degrades instead of raising, that
`HysteresisSuppressor` used directly matches the `suppress()` wrapper
(proof G-B can't drift), and that `hover_pan_px_threshold` discounts
correctly with the optical-flow call mocked out.

**Fourth pass (training-focused, per explicit ask) fixed:**
- `train/make_night.py` was **not idempotent** — re-running it (interrupted
  Kaggle session, accidental re-run) could glob its own `_night.*` output
  back in as source images, re-darkening them and inflating the `--fraction`
  denominator. Now excludes `*_night.*` from the candidate pool and reports
  how many were excluded. Docstring's "Poisson-ish noise" corrected to what
  the code actually does (Gaussian).
- **D4 closed** (concurrently, by another session working this repo at the
  same time — verified correct, not re-done): `train_aerial.py` now has a
  real `PRETRAINED` HF-Hub lookup + `_resolve_pretrained()`, defended with a
  fallback to stock on any download/load failure. This was the brief's own
  recommendation ("spend 10 min finding a VisDrone-pretrained checkpoint")
  that every earlier pass flagged as open and never implemented in code.

**Fifth pass (ENHANCEMENT_PLAN.md Phases 14-19, written not run):**
- **P14** economics headline — `run.py` now resets/reads
  `torch.cuda.max_memory_allocated()` and writes `gpu_mem_gb` into
  `out/alerts.json`; new `scripts/economics.py` turns that plus
  `n_frames/wall_seconds` into sustained FPS, peak GPU memory, and an
  extrapolated feeds-per-GPU number, plus the "% of frames that reached the
  VLM" claim from the brief.
- **P15** `REFERENCES.md` — five citations (AnyAnomaly, FADE, WinCLIP/
  AnomalyCLIP, Holmes-VAU, Open-Vocabulary VAD), each mapped to one specific
  design decision already in this repo, for the "why not just a detector"
  question.
- **P16** `scripts/compare_backends.py` — shells out to `run.py` with
  different `vlm.backend` values (and a forced-CPU run) on the same clip and
  diffs latency/alerts side by side. Verifies the already-implemented
  `smolvlm` backend actually produces alerts; has never been run (needs a
  real environment).
- **P17** `pipeline/zone_classify.py` (new) + `zones.py --auto` — zero-shot
  CLIP scene classification per grid cell (reuses `fit.py`'s SigLIP/
  ViT-B-32 fallback order, no new model dependency), proposes
  `restricted_zones` polygons for cells classified as a driving lane or a
  restricted/fenced area. Suggestion only — still requires `s` to save,
  never silently overwrites a hand-drawn zone. New `config.yaml` `zones:`
  section (`auto_classify`, `grid`, `confidence_floor`).
- **P18** distillation stretch, opt-in and gated — new `train/label_pseudo.py`
  (teacher API labeling of the SAME candidate-event windows the small model
  is judged on, using `vlm_judge.PROMPT` verbatim; refuses to run without
  `TEACHER_API_BASE`/`KEY`/`MODEL` set) and `train/distill_vlm.py` (Unsloth
  LoRA/QLoRA fine-tune of `Qwen2.5-VL-3B-Instruct`). New `qwen_distilled`
  backend in `pipeline/vlm_judge.py` (base model + PEFT adapter from
  `vlm.distilled_adapter_path`). **Governing rule unchanged**: this backend
  is opt-in only and is meant to be selected AFTER a mandatory A/B against
  prompted `qwen` on held-out clips shows it wins — never presented
  unverified. Needs new deps (`unsloth`, `trl`, `peft`, `requests`) — added
  to `requirements.txt` as commented-out, Phase-18-only lines so the default
  install is unaffected.
- **P19** `PREFLIGHT_CHECKLIST.md` — the night-before checklist (weight
  caching, cold-shell preset runs, wifi-off dry run), procedure not code.

**Known-and-accepted limits** (say these out loud rather than hiding them):
SAHI is not inside tracking because it returns no track IDs (measurement via
`scripts/sahi_recall.py` only — G-E); open-vocab can only see windows
geometry already flagged, so a novel object that never forms a track is
invisible (G-D); `abandoned` gets an `owner_hint` proximity signal, never a
real association (G-F); re-ID does not run in streaming mode (G-C remainder);
Grounding DINO fallback not implemented, SmolVLM2 is (G-J); `fuse.w_novelty`
defaults to 0 until the bank is validated on real footage (G-I).

| Stage | File | Reachable from | Default |
|---|---|---|---|
| orchestrator | `run.py` | CLI (`--preset day/night/fast/accurate/live`) | — |
| night data | `train/make_night.py` | CLI | — |
| fine-tune | `train/train_aerial.py` | CLI (`--base`, `--resume`) | **never launched** |
| tracking | `pipeline/tracks.py` | run.py | on |
| events | `pipeline/events.py` | run.py, stream.py | on |
| scene fit | `fit.py` | CLI | — |
| streaming | `pipeline/stream.py` | run.py | **off** (`stream.enabled`) |
| VLM judge | `pipeline/vlm_judge.py` | run.py, stream.py, forensic.py | **off** (`vlm.backend: none`) |
| fusion | `pipeline/fuse.py` | run.py, stream.py | on |
| open-vocab | `pipeline/openvocab.py` | run.py | **off** (`open_vocab.backend: none`) |
| re-ID | `pipeline/reid.py` | run.py | **off** (`reid.backend: none`) |
| retrieval + novelty | `pipeline/retrieve.py` | `query.py`, run.py | **off** (`fuse.w_novelty: 0.0`) |
| evaluation | `pipeline/evaluate.py` | `eval_run.py` | — |
| demo UI | `demo.py` | CLI | — |
| forensic summary | `forensic.py` | CLI | — |
| zones | `zones.py` | CLI | — |
| A/B weights | `scripts/ab_weights.py` | CLI | — |
| TensorRT + FPS | `scripts/export_engine.py` | CLI | — |
| SAHI recall | `scripts/sahi_recall.py` | CLI | not in tracking (no track IDs) |
| night val YAML | `scripts/make_night_yaml.py` | CLI | — |
| F7 SAM2 / map / congestion | — | — | not started, optional |
| economics headline | `scripts/economics.py` | CLI, reads `out/alerts.json` | — |
| backend comparison | `scripts/compare_backends.py` | CLI | — |
| zone auto-classify | `pipeline/zone_classify.py` | `zones.py --auto` | **off** (manual is default) |
| distillation labeling | `train/label_pseudo.py` | CLI | refuses without teacher API env vars |
| distillation fine-tune | `train/distill_vlm.py` | CLI | — |
| distilled VLM backend | `pipeline/vlm_judge.py` (`QwenDistilledJudge`) | run.py, stream.py | **off** (`vlm.backend: qwen_distilled`, needs `vlm.distilled_adapter_path`) |

**Everything is reachable; much of it is off by default.** A bare
`python run.py --video x.mp4` runs tracking → events → geometric fusion only.
The presets turn the rest on.

No trained weights exist (`weights/aerial_night/weights/best.pt` absent).

## Fixed, unverified
Found by static review, fixed, never run. Re-check these first if the first
Kaggle run fails.
- `out/` is now created by `run.py` and `fit.py` before they write into it.
- `NoopOpenVocab.detect` took a required `prompts` arg that `run.py` never
  passed — `TypeError` on the first candidate event with `backend: none`.
- `retrieve.search` divided source frame numbers by the *sampled* fps.
  `fit.py` now records `src_fps` in `scene_fit.json` and retrieval reads it.
- `requirements.txt` was missing `torchreid`; `sahi` / `qwen-vl-utils` are
  commented out until something actually imports them.

## Kaggle execution notes
- `train/train_aerial.py --preset kaggle` → `yolo11s` @1024, 40 epochs, no
  freeze. `--preset fast` (`yolo11n` @768, 12 ep) is the fallback.
- VisDrone auto-downloads on the first `train()` call via `VisDrone.yaml`.
- Check epoch-1 wall-clock: `epoch1 × epochs × 1.15` must fit the session
  limit. If it doesn't, kill and drop a preset immediately.
- Run `train/make_night.py` on the VisDrone train split **before** training,
  or the night robustness is not in the weights.
- Download `weights/aerial_night/weights/best.pt` before the session ends —
  Kaggle output is lost otherwise. `last.pt` is the rescue if you stop early.

## Remaining steps, in order
1. First Kaggle run: `make_night.py`, then T3. Keep `best.pt`.
2. **F0 — A/B the weights.** Fine-tuned vs stock, day and night, VisDrone val.
   Keep whichever wins; report both numbers either way.
3. Wire F1 → F2 → F3 (re-ID, retrieval, TensorRT export + sustained FPS).
4. **F4 — demo surface.** Alert timeline, click-to-clip, one-sentence "why".
5. **F5 — eval sweep.** Day and night AUC separately, threshold sweep,
   FP/hour, p50/p95 latency, time-to-detection.
6. F6 and/or F7 — only if the above is finished.
7. **F8 — presets** (`day`, `night`, `fast`, `accurate`) as one-line commands.
8. **F9/F10 — dry runs** on unseen clips, timed. Non-negotiable.
9. **F11 — five slides**: problem, architecture, numbers, limitations, pitch.

## Governing rules
- Train only what is domain-general (the aerial detector). Scene-specific
  baselines are *fitted* at the event in under 5 min by `fit.py` — arithmetic,
  not gradient descent. **Never fine-tune the VLM; prompt it.**
- Report day and night metrics separately, even when night is worse.
- Split datasets by video, never by frame. `x.jpg` and `x_night.jpg` belong on
  the same side of the split.
- A `None` score or baseline is a deliberate refusal, never coerced to `0.0`.
  `_speed_stats` under 20 samples, `min_track_frames`, and single-class AUC all
  refuse on purpose — keep it that way.
