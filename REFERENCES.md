# References — prior art behind this design

P15. Each entry maps to one specific decision already in this repo, not a
generic name-drop. Use these when a judge asks "why not just a detector?" or
"why is this VLM call justified?"

## AnyAnomaly (WACV 2026)
Zero-shot, customizable video anomaly detection with an LVLM; no fine-tuning
required. github.com/SkiddieAhn/Paper-AnyAnomaly

**Maps to:** the Stage-4 judge (`pipeline/vlm_judge.py`) - the nearest
published architecture to "detector/tracker feeds a prompted VLM judge,"
which is this repo's own governing rule ("never fine-tune the VLM; prompt
it," `CLAUDE.md`).

## FADE (BMVC 2024)
Training-free few/zero-shot anomaly detection using a large VLM, no
fine-tuning or auxiliary training data. github.com/BMVC-FADE/BMVC-FADE

**Maps to:** validates "prompt, don't fine-tune" as a deliberate, published
approach rather than a shortcut taken under time pressure. If Phase 18
(distillation) is attempted and loses the A/B, FADE is the citation for why
the fallback is still a legitimate architecture, not a consolation prize.

## WinCLIP / AnomalyCLIP
Zero-shot anomaly detection via CLIP-style embedding similarity to a "normal"
reference set, no training. Curated list:
github.com/mala-lab/Awesome-Anomaly-Detection-Foundation-Models

**Maps to:** `pipeline/retrieve.py`'s `novelty()` function - "1 minus mean
cosine similarity to the k nearest normal-bank frames" is exactly this
framing, built on the same SigLIP embeddings `fit.py` already computes for
retrieval. Say explicitly: "our novelty score is the WinCLIP/AnomalyCLIP
framing, zero-shot, no training."

## Holmes-VAU (CVPR'25)
Long-term video anomaly *understanding* at multiple granularities using
MLLMs - explanation, not just a binary flag.

**Maps to:** `PROMPT` in `pipeline/vlm_judge.py` asks for a `why` field
(free-text reasoning), and `summarize()` (F6, `forensic.py`) produces a
paragraph across multiple events. Both exist because a flag without a reason
is not actionable for an operator.

## Open-Vocabulary Video Anomaly Detection (arXiv 2311.07042)
Argues anomaly detection must be open-set, not classifier-based, because the
anomaly class list cannot be enumerated in advance.

**Maps to:** `pipeline/openvocab.py` (YOLO-World, text-prompted, no
retraining to add a class) and the VLM's free-text `why` field - both exist
because a closed class list cannot name "smoke where there shouldn't be."
Also the honest limitation to state alongside it (G-D, `CLAUDE.md`): a novel
object that never forms a track is still invisible to this pipeline, because
open-vocab only samples windows geometry already flagged.

## Drone-Anomaly dataset
37 train / 22 test aerial anomaly video sequences, 7 scenes, 87,488 frames.
github.com search: "Drone-Anomaly ANDT baseline"

**Maps to:** if Phase 6's A/B or Phase 11's eval sweep need a second
benchmark beyond VisDrone, this is the closest public dataset to what this
pipeline is actually judged on - purpose-built aerial anomaly footage, not
generic aerial detection.
