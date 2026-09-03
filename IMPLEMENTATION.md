# Implementation Plan — start to end

How to build this repo from where it is now to a demo-ready system. Every
phase says: **what you do**, **which files**, **how**, and **done when**.

- Runtime is **Kaggle**. Nothing heavy runs locally. See [Part 0](#part-0--how-to-work).
- Current state and the real gaps: [Part 1](#part-1--where-we-actually-are).
- File-by-file reference: [Part 2](#part-2--file-by-file-reference).
- The phases: [Part 3](#part-3--the-phases).

---

## Part 0 — How to work

**Local machine (no GPU):** edit code only. `python -c "import ast,sys;
ast.parse(open(f).read())"` is the only check available — there is no torch,
no ultralytics, no dataset.

**Kaggle (the runtime):** one notebook, GPU accelerator on (P100 or T4).

```python
# Cell 1 - get the code in
!git clone https://github.com/Yogendergodara/flytbase-prep.git
%cd flytbase-prep

# Cell 2 - install (Kaggle already has torch + opencv + numpy + sklearn)
!pip install -q ultralytics transformers accelerate open_clip_torch

# Cell 3 - confirm the GPU before anything else
!nvidia-smi
```

**Rules of the runtime**
- Kaggle sessions die. Anything you want to keep must be written to
  `/kaggle/working/` and downloaded before the session ends.
- GPU quota is weekly, not per-session. Do not burn it on runs you have not
  extrapolated first.
- Re-clone every session. Uncommitted local edits do not exist on Kaggle —
  **commit and push before you start a session.**

---

## Part 1 — Where we actually are

> **Every claim below is static review. Not one line of this repo has ever
> executed.** "Fixed" means the code now reads correctly, not that it ran.

### Wired and reachable
`tracks.py` → `events.py` → `openvocab.py` → `vlm_judge.py` → `fuse.py` is the
batch chain. `stream.py` runs the same cascade on a sliding window.
`reid.py`, `retrieve.py` and `evaluate.py` are now called by `run.py`,
`query.py`/`run.py`, and `eval_run.py` respectively.

**Wired but off by default** — a default run exercises none of these:
`reid.backend: none`, `fuse.w_novelty: 0.0`, `open_vocab.backend: none`,
`vlm.backend: none`, `stream.enabled: false`, `detector.sahi.enabled: false`.

### The original five gaps — all closed
| # | Gap | Closed by |
|---|---|---|
| G1 | `run.py` never read `out/scene_fit.json` | fit loads before tracking; `speed_stats` + `density` passed into `detect_events` |
| G2 | normal bank unused as an anomaly score | `retrieve.novelty()` / `score_frame_novelty()`, blended in `fuse.py` via `w_novelty` |
| G3 | no density / abandoned events | `density_anomaly` (scene-level, `track_id=-1`) and `abandoned` in `events.py` |
| G4 | no CLAHE / night path | `night.*` config, L-channel CLAHE in `tracks.py`, `--preset night` sets `night.clahe=true` |
| G5 | no `--preset` | `--preset {day,night,fast,accurate,live}` |

### Second review pass — also fixed
**Bugs:** `--fit-seconds` was dead (now caps `video.max_seconds`) ·
hysteresis did not exist (`clear_threshold` + `ema_alpha` now drive a real
raise/clear state machine) · events carried the whole tracklet lifetime (now
clipped to `clip_seconds` around the peak) · loiter/abandoned double-fired
(abandoned wins, non-person only) · VLM latency was measured and discarded
(now in `alerts.json`, read by `eval_run.py`) · `extract_frames(k=1)` returned
`t_start` not the middle · `smolvlm` backend raised `ValueError` (now
implemented) · `link_identities` used chained assignment (now union-find,
same-class only) · `identity` was never `-1` so "linked" was indistinguishable
from "unlinked" · `sweep()` returned empty rows on single-class labels ·
`half: true` crashed on CPU.

**Performance:** no quantization (now `vlm.load_4bit`, NF4 double-quant) ·
no `inference_mode` on the Qwen calls · `dwell_seconds` was O(n²) and called
twice per tracklet (now O(n) via monotonic deques) · `score_frame_novelty`
reloaded SigLIP per event (now cached, GPU) · `crop_at` opened a
`VideoCapture` per crop (now one per run) · three captures per event in the
judge loop (now one) · open-vocab looped a noop when disabled.

**Architecture:** batch-only, no alerts until the file ended → `pipeline/stream.py`
+ `--preset live`, with tracklet retirement to bound RAM over hours.

**Tooling added:** `zones.py` (click polygons into config), `query.py`,
`demo.py`, `eval_run.py`, `forensic.py`, `scripts/ab_weights.py`,
`scripts/export_engine.py`, `scripts/sahi_recall.py`,
`scripts/make_night_yaml.py`; `train_aerial.py` gained `--base`, `--resume`,
`--seed`, `save_period`.

### Still open — known and deliberate
| Item | Status |
|---|---|
| **Phase 5 — the fine-tune has never run** | The only hard gate. No weights exist. |
| **SAHI is not inside tracking** | SAHI returns no track IDs and Ultralytics' tracker is coupled to `model.track()`. It ships as `scripts/sahi_recall.py` to back the recall claim with a number. Wiring it into tracking would mean hand-rolling association. |
| **Open-vocab can only see geometry-flagged windows** | Structural: a bag never becomes a track (not in `detector.classes`), so it never becomes a candidate. Name this limitation rather than hiding it. |
| **No owner association for `abandoned`** | It means "object stopped moving", not "someone left it". |
| **F7** (SAM2 / map grounding / congestion) | Optional by the plan. Congestion is nearly free given `density_anomaly`. |
| **Grounding DINO fallback** | Not implemented. |
| **Ground-truth labels for `eval_run.py`** | Must be hand-built per video: `{"anomalous_ranges": [[t0,t1]]}`. |
| **Phase 13** — dry runs, cold-shell preset check, slides | Procedure, not code. |

---

## Part 2 — File-by-file reference

### `config.yaml`
Single source of every knob. Sections: `video`, `detector`, `events`,
`open_vocab`, `reid`, `retrieve`, `vlm`, `fuse`. Overridable at the CLI with
`--set detector.imgsz=1280 vlm.backend=qwen`.

### `run.py`
Orchestrator. `apply_overrides(cfg, pairs)` walks dotted keys and
`yaml.safe_load`s the value. `main()` runs the six stages and prints one line
each (`[1-2]` tracking, `[3]` candidates, `[3b]` open-vocab, `[4]` VLM,
`[5]` suppression, `[6]` write). Emits `out/alerts.json` = `{config, eff_fps,
n_frames, alerts}`.

### `fit.py`
Measures, never trains. `fit_embeddings(video, n, out_npy)` samples evenly
spaced frames, encodes with SigLIP (falls back to ViT-B-32/openai), L2
normalises, saves `out/normal_bank.npy`, and returns
`{encoder, n, frame_idx, src_fps}`. `main()` also computes per-class speed
stats and a density baseline, then writes `out/scene_fit.json`.
**Refuses:** density is `None` under 20 samples.

### `pipeline/tracks.py`
`Tracklet` dataclass — `track_id, cls, t[], cx[], cy[], w[], h[], conf[]`,
with `n()`, `duration()`, `speeds_px_s()` (returns `None` under 2 points,
never `0.0`), and `dwell_seconds(radius_px)` (longest run inside a radius of
its own running centroid). `run_tracking(cfg, on_frame=None)` runs
`model.track(stream=True, persist=True, vid_stride=stride)` and returns
`(tracks, n_frames, eff_fps)`.

### `pipeline/events.py`
Arithmetic only, no model. `CandidateEvent(kind, track_id, cls, t_start,
t_end, geo_score, facts)`. `detect_events(tracks, cfg, class_speed_stats=None)`
emits `loiter`, `speed_anomaly`, `zone_intrusion`, then filters by
`events.candidate_floor`. `_speed_stats(tracks)` returns `{cls: (mean, std)}`
with `(None, None)` under 20 samples. `_point_in_poly` is a ray-cast test.

### `pipeline/openvocab.py`
`build_open_vocab(cfg)` → `NoopOpenVocab` or `YoloWorldOpenVocab`
(`model.set_classes(prompts)` at init, re-set if prompts change). `detect()`
samples `frames_per_event` frames inside the window and returns
`{hits: [{t, prompt, conf, xyxy}], reason}`. Hits land in `ev.facts` and
therefore reach the VLM prompt.

### `pipeline/vlm_judge.py`
`PROMPT` hands the model the *measured* facts and asks for strict JSON.
`extract_frames(video, t0, t1, k)` seeks by ms and converts BGR→RGB.
`build_judge(cfg)` → `NoopJudge` (returns `score: None`) or `QwenJudge`.
`_parse` regex-extracts the JSON object and clamps `score` to 0–1; an
unparseable reply becomes `score: None`, never `0.0`.

### `pipeline/fuse.py`
`fuse_score(ev, verdict, cfg)` → `(score, mode)`; `mode` is
`"geometric_only"` when the VLM gave no opinion. `suppress(scored, cfg)`
applies the raise threshold, `min_event_seconds`, and a per-`track_id`
cooldown, and returns alert dicts.

### `pipeline/reid.py`
`OSNetEmbedder` via `torchreid.utils.FeatureExtractor`. `crop_at(cap, ...)`
takes an **already-open** capture; `tracklet_embedding(cap, ...)` averages `k`
crops and returns `None` below `min_crops`.
`link_identities(tracks, video, cfg)` → `(identity, linked)` — union-find,
same-class only, never links tracklets that overlap in time. `linked` is the
set that actually merged, so `run.py` can set `identity = -1` for the rest.

### `pipeline/retrieve.py`
`load_bank()` → `(embs, bank_meta, err)`. `_load_encoder` caches the model
module-level and moves it to GPU. `search(query, ..., src_fps, top_k)` returns
ranked frames with `approx_seconds`. `novelty(emb, bank, k)` and
`score_frame_novelty(frame_rgb)` are the anomaly-score half of the same index
— the WinCLIP/AnomalyCLIP framing, zero-shot rather than trained.
Driven by `query.py` and by `run.py` when `fuse.w_novelty > 0`.

### `pipeline/evaluate.py`
`frame_scores(alerts, n_frames, fps)` paints alert scores onto a frame
timeline. `report(...)` → frame AUC, P/R/F1, FP frames, FP/hour, p50/p95
latency; **refuses** to report AUC on single-class labels. `sweep(...)` walks
thresholds and refuses as a whole on single-class labels rather than
returning rows containing only a threshold. Driven by `eval_run.py`.

### `pipeline/stream.py`
`StreamingPipeline` runs the same cascade on a sliding window so alerts are
emitted *during* the pass. `on_frame` evaluates every `stream.window_seconds`;
`finalize` handles the tail. `(track_id, kind)` memoisation means the VLM is
never re-paid for a known event; hysteresis state persists across windows;
`_retire` drops tracklets idle beyond `stream.retire_after`, which is what
bounds memory over hours of footage.

### `train/make_night.py`
`to_night(img, rng)` = gamma 1.8–3.0 down, gain 0.35–0.6, blue-ward
desaturation, Gaussian noise σ 4–12, 50% chance of a 3×3 blur. Writes
`<stem>_night.<ext>` beside the original and copies the label verbatim
(geometry unchanged). Skips images with no label.

### `train/train_aerial.py`
`PRESETS = {overnight: (yolo11s, 1280, 60ep, b8, freeze0), laptop: (yolo11n,
960, 25ep, b8, freeze10), kaggle: (yolo11s, 1024, 40ep, b12, freeze0), fast:
(yolo11n, 768, 12ep, b16, freeze10)}`. Aug is aerial/low-light aimed:
`hsv_v=0.65`, `mosaic=1.0`, `close_mosaic=8`, `flipud=0.2`, `cos_lr=True`.
Output → `weights/<name>/weights/best.pt`.

---

## Part 3 — The phases

### Phase 1 — Make the scaffold honest *(local, ~30 min)* ✅ DONE

Fix G1 first; it is the smallest change with the biggest effect on the story.

**Files:** `run.py`, `pipeline/events.py`

**What to write** — in `run.py`, load the fit if it exists and pass it down:

```python
import os
fit = None
if os.path.exists("out/scene_fit.json"):
    fit = json.load(open("out/scene_fit.json", encoding="utf-8"))
    print(f"[0]   using scene fit from {fit['video']} "
          f"({fit['sampled_frames']} frames, {fit['wall_seconds']}s)")

speed_stats = None
if fit:
    speed_stats = {int(c): (v["mean"], v["std"]) if v else (None, None)
                   for c, v in fit["speed_by_class"].items()}
cands = detect_events(tracks, cfg, speed_stats)
```

`detect_events` already accepts the argument — nothing in `events.py` needs
to change for this. Print which path was taken; on Saturday you want to
*see* that the fit was used.

**Done when:** `run.py` prints a `[0]` line naming the fit file, and running
without `out/scene_fit.json` still works (falls back to per-video stats).

---

### Phase 2 — Density and abandoned events *(local, ~45 min)* ✅ DONE

Fixes G3. Two new event kinds, both pure arithmetic.

**Files:** `pipeline/events.py`, `config.yaml`

**Config to add** under `events:`

```yaml
  density_z_threshold: 2.0    # KNOB: crowding vs the fitted density baseline
  abandoned_radius_px: 25     # a bag that stops moving entirely
```

**What to write** in `events.py` — `detect_events` needs the fitted density,
so give it one more optional argument:

```python
def detect_events(tracks, cfg, class_speed_stats=None, density=None):
```

Then, after the per-tracklet loop, add a frame-level density pass: rebuild
the per-timestamp object count the same way `fit.py` does, z-score each
frame against `density["mean"]`/`density["std"]`, and emit a
`density_anomaly` event for runs above `density_z_threshold`. Emit it with
`track_id = -1` — it is a scene event, not an object event, and `fuse.suppress`
keys its cooldown on `track_id`.

For `abandoned`: a tracklet of class bag/suitcase (or any non-person class)
whose `dwell_seconds(abandoned_radius_px) >= events.abandoned_seconds` and
whose owner-person track has left. Simplest honest version: dwell above the
threshold with a *tighter* radius than loiter, and say in the demo that
owner-association is not implemented.

**Done when:** both kinds appear in `out/alerts.json` on a clip that has
them, and neither fires when `density` is `None` (no fit → no claim).

---

### Phase 3 — The night path *(local, ~1 h)* ✅ DONE (both options implemented)

Fixes G4. This is the stated differentiator and it currently does not exist.

**Files:** `config.yaml`, `pipeline/tracks.py`

**Config to add** (top level):

```yaml
night:
  enabled: false           # KNOB: --set night.enabled=true
  clahe_clip: 2.0          # KNOB: more contrast AND more amplified noise
  clahe_grid: 8
  conf: 0.15               # KNOB: separate, LOWER than detector.conf
```

**What to write** in `tracks.py` — `model.track(source=path, ...)` hands the
video to Ultralytics, which means you cannot preprocess frames inside that
call. Two options, pick one and say which in Q&A:

1. **Lower `conf` only** (10 minutes): in `run_tracking`, when
   `cfg["night"]["enabled"]`, override `conf=cfg["night"]["conf"]`. Honest,
   cheap, no CLAHE.
2. **CLAHE properly** (1 hour): read frames yourself with `cv2.VideoCapture`,
   apply CLAHE on the L channel of LAB, and feed the array list to
   `model.track(source=frames, ...)`. Costs the streaming behaviour.

Start with option 1 so the night preset exists, then upgrade if time allows.
CLAHE goes on the L channel, never on BGR directly:

```python
def clahe_bgr(frame, clip, grid):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```

**Done when:** T9's four numbers exist — detections counted day/night ×
before/after your night path, written down.

---

### Phase 4 — Presets *(local, ~30 min)* ✅ DONE

Fixes G5. Saturday morning must type one line, not edit YAML.

**Files:** `run.py`, `fit.py`

```python
PRESETS = {
    "day":      {"detector.conf": 0.25, "night.enabled": False,
                 "vlm.backend": "qwen"},
    "night":    {"detector.conf": 0.15, "night.enabled": True,
                 "vlm.backend": "qwen"},
    "fast":     {"video.target_fps": 2, "detector.imgsz": 960,
                 "vlm.backend": "none", "open_vocab.backend": "none"},
    "accurate": {"video.target_fps": 5, "detector.imgsz": 1280,
                 "vlm.backend": "qwen", "open_vocab.backend": "yoloworld",
                 "vlm.frames_per_event": 8},
}
```

Add `ap.add_argument("--preset", choices=list(PRESETS))` and apply the preset
**before** `--set`, so an explicit `--set` always wins. Reuse
`apply_overrides` by turning the dict into `k=v` strings — one code path for
both, so a preset can never drift from what `--set` does.

**Done when:** all four presets run from a cold shell with one line each.

---

### Phase 5 — Kaggle: night data + the fine-tune *(Kaggle, ~5-6 h GPU)*

**Commit and push Phases 1-4 first.**

```python
# VisDrone downloads on the first train call. Trigger it, then darken.
from ultralytics import YOLO
YOLO("yolo11n.pt").train(data="VisDrone.yaml", epochs=0)  # fetch only

!python train/make_night.py \
    --images /kaggle/working/datasets/VisDrone/images/train \
    --labels /kaggle/working/datasets/VisDrone/labels/train \
    --fraction 0.4
```

**Look at ten of the output images before training.** Too dark is a real
failure mode — if you cannot see the objects, neither can the detector; lower
the gamma range in `to_night` and rerun.

```python
!python train/train_aerial.py --preset kaggle --name aerial_night
```

**The extrapolation check:** epoch-1 wall-clock × 40 × 1.15 must fit inside
the session limit. If it doesn't, kill it and relaunch on `--preset fast`.
A run you stop early still leaves a usable `last.pt`.

**Download before the session dies:**
`weights/aerial_night/weights/best.pt` and `last.pt`.

**Done when:** `best.pt` is on your local disk and committed somewhere you
can reach on Saturday.

---

### Phase 6 — F0: A/B the weights *(Kaggle, ~30 min)* ✅ SCRIPT DONE, run on Kaggle

Never trust a checkpoint you have not measured against stock.

```python
!yolo val model=weights/aerial_night/weights/best.pt data=VisDrone.yaml imgsz=1280
!yolo val model=yolo11s.pt                          data=VisDrone.yaml imgsz=1280
```

Then repeat both on a **night** subset (point a small YAML at only the
`_night.jpg` files). Four numbers, one table.

**Done when:** you have the table and a one-sentence verdict. **If the
fine-tune lost, keep stock and say so on the slide** — that is a legitimate
result, not a wasted night.

---

### Phase 7 — F1: wire re-ID *(~1.5 h)* ✅ DONE

**Files:** `run.py`, `pipeline/events.py`

Check `botsort.yaml` first — if BoT-SORT's own re-ID closes your ID switches,
you have saved a model and a phase. Set
`--set detector.tracker=botsort.yaml`, count switches by eye over 30 s, and
only proceed if it is still bad.

Otherwise, in `run.py` after tracking:

```python
if cfg["reid"]["backend"] != "none":
    from pipeline.reid import link_identities
    identity = link_identities(tracks, cfg["video"]["path"], cfg)
    merged = len(set(identity.values()))
    print(f"[2b]  re-id: {len(tracks)} tracklets -> {merged} identities")
    for tid, tr in tracks.items():
        tr.identity = identity[tid]
```

`Tracklet` is a plain dataclass, so add `identity: int = -1` to it rather
than setting an undeclared attribute. Then carry `identity` into
`CandidateEvent.facts` so the VLM prompt and the demo can both quote it.

**Done when:** one identity is matched across an occlusion gap, and you can
state both failure directions of `cosine_threshold` — too low merges
strangers, too high splits one person into three.

---

### Phase 8 — F2: retrieval + the normal-bank score *(~1.5 h)* ✅ DONE

Fixes G2 and delivers the level-3 query.

**Files:** new `query.py`, `pipeline/events.py`

**The CLI** (`query.py`):

```python
import argparse, json
from pipeline.retrieve import search

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--top-k", type=int, default=5)
a = ap.parse_args()
r = search(a.query, top_k=a.top_k)
if r["reason"]:
    print("refused:", r["reason"])
for h in r["results"]:
    print(f"  #{h['rank']}  frame {h['frame_index']:>7}  "
          f"{h.get('approx_seconds', '?')}s  sim={h['similarity']}")
```

**The anomaly score** — the missing half of the dual-use index. Add to
`retrieve.py`:

```python
def novelty(frame_emb, bank, k=5):
    """1 - mean cosine to the k nearest normal frames. High = unlike normal."""
    sims = bank @ frame_emb
    return float(1.0 - sims[np.argsort(-sims)[:k]].mean())
```

Score each candidate event's middle frame and put the result in
`ev.facts["novelty"]`, then give it weight in `fuse.py`. This is what makes
the normal bank zero-shot rather than trained — the same framing as WinCLIP
and AnomalyCLIP.

**Done when:** `python query.py "person carrying a bag near the gate"`
returns the right clip, and `novelty` appears in the alert facts.

---

### Phase 9 — F3: real-time and the FPS number *(~1 h)* ✅ DONE

```python
!yolo export model=weights/aerial_night/weights/best.pt format=engine half=True
```

Then measure **your pipeline**, not the detector. `run.py` already prints
frames/s for the tracking stage; add an end-to-end wall-clock line covering
every stage, and report that.

**Done when:** you have one sentence — "X FPS sustained, end to end, on this
machine" — and you know which stage is the bottleneck.

---

### Phase 10 — F4: the demo surface *(~1.5 h)* ✅ DONE

The judges see this. Build the *smallest* thing that reads clearly.

**File:** new `demo.py` — a single HTML file written from `out/alerts.json`
is enough and cannot break on stage:

- A timeline of alerts, ordered by `t_start`.
- Each row: time, kind, score, and the one-sentence `why` from the VLM.
- Click a row → play the evidence clip from `t_start` to `t_end`.
- Show the measured facts beside each verdict, so "why" is auditable.

**Done when:** someone who has not seen it can say what the system found and
why, within thirty seconds. If they cannot, rebuild the screen — not the
model.

---

### Phase 11 — F5: the evaluation sweep *(~1.5 h)* ✅ DONE (needs a labels file per video)

**File:** new `eval_run.py`

Load CUHK Avenue frame labels, run the pipeline, then:

```python
from pipeline.evaluate import frame_scores, report, sweep
scores = frame_scores(alerts, n_frames, fps)
print(report(scores, labels, cfg["fuse"]["raise_threshold"], fps, latencies))
for row in sweep(scores, labels):
    print(row)
```

**Report day and night separately.** Two runs, two tables, both on the
slide. `report` refuses single-class labels on purpose — that refusal is a
feature, not a bug to work around.

Add **time-to-detection**: seconds between the label's first anomalous frame
and your first alert in that window. For a live system this matters more than
raw accuracy — an alert four minutes late is a report, not an alarm.

**Done when:** day AUC, night AUC, threshold sweep, FP/hour, p50/p95 latency,
and time-to-detection all exist as numbers.

---

### Phase 12 — F6/F7: forensics, then one upgrade *(~2 h, optional)* ✅ F6 DONE, F7 skipped

**F6 forensic summary** — new function in `vlm_judge.py`: take *all* alerts
in a five-minute window, hand the VLM their timestamps and facts (no images,
or one frame each), and ask for one paragraph citing timestamps. Read it
aloud and check every claim against the footage; a VLM will invent a detail
if the prompt lets it.

**F7 — pick exactly one:** SAM 2 masking, map grounding, or congestion
analysis (congestion is nearly free once Phase 2's density event exists).
**If it is not working by the deadline, delete it.** A half-wired upgrade is
worse than none.

---

### Phase 13 — F8-F11: harden, rehearse, pitch *(~3 h, non-negotiable)*

1. Verify all four presets run from a cold shell.
2. **Dry run one:** an unseen clip, start to finish, nothing touched. Time it
   — that number is your honest estimate for the arena's first thirty minutes.
3. **Dry run two:** an unseen *night* clip in twenty minutes, changing
   `conf`, a prompt, and a threshold live while narrating each one. This is
   the skill the arena actually tests.
4. **Final dry run with wifi off.** The only way to prove nothing silently
   reaches for a download.
5. Five slides: problem, architecture, the numbers, the honest limitations,
   the one sentence. Say the two-minute pitch out loud, timed, once.

---

## Order of work, and what to drop

| Priority | Phases | If time runs out |
|---|---|---|
| Must | 1, 4, 5, 6, 13 | Nothing here is droppable. Phase 5 gates everything. |
| High | 2, 3, 11 | Phase 3 can shrink to the `conf`-only version. |
| Medium | 7, 8, 9, 10 | Phase 10 beats Phase 9 if you can only do one — judges see the screen. |
| Optional | 12 | Drop first, without regret. |

**The single sentence to be able to say:** everything domain-general was
trained before arrival; everything scene-specific is fitted in five minutes
because it is arithmetic, not gradient descent; and the class list is text,
so what counts as an anomaly can change while they watch.
