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

**Phase 5 (Kaggle training) has run and finished.** VisDrone-only `yolo11s`
@1024, early-stopped at epoch 33/40 (patience=8, no improvement in last 8),
best weights from **epoch 25**. Val (548 VisDrone val images, 38,759
instances): **P 0.598, R 0.464, mAP50 0.491, mAP50-95 0.291**. Per-class
mAP50-95 ranges from 0.609 (car) down to 0.116 (awning-tricycle) — small/rare
classes are notably weaker than car/bus. `best.pt` saved to
`weights/aerial_night/weights/best.pt` on Kaggle — **must still be downloaded
locally**, Kaggle output does not persist. Whether `make_night.py` ran before
this training pass has not been confirmed — the run log doesn't show it, and
the checkpoint name (`aerial_night`) is a directory name, not evidence night
augmentation was in the data. **F0 (A/B fine-tuned vs stock on a night clip)
is the mandatory next step before trusting this checkpoint** — the training
log itself said so. P13 (harden/dry-run/pitch) is procedure, not code.

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

**Test suite now covers all of the above** — 37 tests passing, including a
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
| fine-tune | `train/train_aerial.py` | CLI (`--base`, `--resume`) | **ran, done** — mAP50-95 0.291 @ epoch 25, `best.pt` not yet downloaded locally |
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
| Pool 1 dataset build | `train/build_aerial_dataset.py` | CLI | written, never run |
| Pool 3 dataset build | `train/build_scene_classifier_dataset.py` | CLI | written, never run |
| Pool 2 corpus build | `train/build_vlm_text_corpus.py` | CLI | written, never run |
| Kaggle run sheet | `KAGGLE_RUN.md` | — | the ordered build+train+A/B commands |

**Everything is reachable; much of it is off by default.** A bare
`python run.py --video x.mp4` runs tracking → events → geometric fusion only.
The presets turn the rest on.

Trained weights exist on Kaggle (`weights/aerial_night/weights/best.pt`,
epoch 25, mAP50-95 0.291) but have **not yet been downloaded** into this
repo's `weights/` — treat the local path as still absent until confirmed
downloaded.

## Dataset build scripts (DATASET_PLAN.md Phase D2-D4, written not run)
`train/build_aerial_dataset.py` (Pool 1: VisDrone + ~15k-frame stratified
UIT-ADrone video sample, class-remapped, VisDrone val untouched for a clean
A/B), `train/build_scene_classifier_dataset.py` (Pool 3: FloodNet + FASDD_UAV
+ capped D-Fire -> `datasets/scene_hazard/{train,val}/{fire,smoke,flood,normal}/`),
`train/build_vlm_text_corpus.py` (Pool 2: small few-shot corpus for
`vlm_judge.PROMPT`). All three assume documented-but-unverified schemas for
datasets this repo has never downloaded. Two corrections grounded in actual
knowledge of these public datasets (not generic guesses): FloodNet's Track-1
classification folders (`Flooded`/`Non-Flooded`) are used directly when
present, only falling back to guessing segmentation-mask pixel values if
they're absent; FASDD is a detection dataset (bbox annotations), not
folder-per-class, and is now handled with the same YOLO-label logic as
D-Fire (`--fasdd-fire-id`/`--fasdd-smoke-id` to correct the class order).
D-Fire's earlier fake "scene grouping" was removed — it's a compiled set of
independent stock photos, not video frames, so a plain per-image split is
honest where the grouping heuristic wasn't. UIT-ADrone's COCO json fields
and the four VLM-text sources' schemas remain unverified guesses; each
refuses or prints a loud warning on a mismatch rather than silently
mis-labeling data — the printed per-source/per-class counts on the first
real run are the actual check. No new dependencies (uses numpy/Pillow
already in requirements.txt).

**Bugs found by smoke-testing these three scripts against synthetic data
(the scripts now have that coverage; the datasets themselves are still
undownloaded):**
- **Labels would never have been found.** UIT labels were written to
  `<out>/uit_labels/` while images stayed in the source tree. Ultralytics
  pairs them by swapping the last `/images/` for `/labels/`, so all 40k
  sampled frames would have trained as empty background — no crash, just a
  quietly ruined run. Frames are now symlinked into `<out>/images/` with
  labels at `<out>/labels/`; a test asserts the derived label path resolves.
- **`motorbike` → bicycle.** Synonyms matched in insertion order, so "bike"
  caught "motorbike" and "tricycle" caught "awning-tricycle". Now exact-match
  first, then longest key first.
- **Every video collapsed into one group.** `Path(file_name).stem` threw away
  the directory, so `clipA/000001.jpg` grouped on `""`. Now prefers the
  parent dir — split-by-video was silently not happening.
- **val took an entire source.** `split_scenes` split on scene *count* with a
  `max(1, …)` floor, so a source with one big scene sent 100% of its images
  to val and none to train. Now splits on image-count fraction and always
  leaves train at least one scene.
- Out-of-frame COCO boxes (coords >1.0, rejected by Ultralytics) now clipped;
  non-recursive globs missed `train/val/test`-nested images; identically
  named files from different sources overwrote each other; a re-run merged
  into the previous build; an O(n²) membership scan over A2Seek's 42k rows.

`train_aerial.py` gained `--epochs`/`--batch`
overrides — needed because the combined dataset is ~5.4x more images than
VisDrone alone, so the "kaggle" preset's 40 epochs would run far longer than
one Kaggle session. `DATASET_PLAN.md`'s Pool-1-v2 command had a real bug
(`--resume` there silently ignores `--data`/`--base`) — fixed in the doc.

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
1. ~~First Kaggle run~~ — **done.** `best.pt` from epoch 25 (mAP50-95 0.291)
   exists on Kaggle; download it into `weights/aerial_night/weights/` before
   the next Kaggle session recycles it.
2. **F0 — A/B the weights.** Fine-tuned vs stock, day and night, VisDrone val.
   Keep whichever wins; report both numbers either way. (The training run
   itself flagged this as the required next step.)
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
