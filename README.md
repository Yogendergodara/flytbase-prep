# FlytBase prep - video anomaly pipeline skeleton

    pip install -r requirements.txt
    # drop any surveillance/drone mp4 into data/
    python run.py --video data/sample.mp4                     # CPU, no VLM
    python run.py --video data/sample.mp4 --set vlm.backend=qwen

## Why it is shaped like this
A VLM on every frame is ~1-3 s/frame. 10 min of 30fps video = 18,000 frames.
So: sample to 3fps (1,800), let YOLO+tracker reduce that to ~40 tracklets, let
arithmetic reduce that to ~8 candidate events, and spend the VLM only there.
Cost drops ~2000x and the VLM's job becomes the one thing it is good at:
judging whether a flagged thing is actually unusual, and saying why.

## Every knob, in one place
`config.yaml`. Each line marked KNOB is a thing you will be asked about.

## The refusals that are deliberate
- `min_track_frames` - a 2-frame blip gets no verdict.
- `_speed_stats` returns `(None, None)` under 20 samples: no baseline, no claim.
- `verdict["score"] is None` -> `fuse_score` returns geometric only, and labels
  it `geometric_only`. It never silently becomes 0.0.
- `evaluate.report` refuses to print an AUC when labels are single-class.
