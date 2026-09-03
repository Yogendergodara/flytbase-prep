"""#27: fail fast and clearly, before spending minutes on tracking/VLM calls
only to crash on a bad config. Not exhaustive schema validation - just the
mistakes that are cheap to make under time pressure (a typo'd path, a
two-point "polygon", a threshold outside its own valid range) and expensive
to discover forty minutes into a run.
"""
import os
from pathlib import Path


class ConfigError(ValueError):
    pass


def _is_remote(path):
    return isinstance(path, str) and path.startswith(("rtsp://", "http://", "https://"))


def validate(cfg, out_path=None):
    errors = []

    path = cfg.get("video", {}).get("path")
    if not path:
        errors.append("video.path is empty")
    elif not _is_remote(path) and not os.path.exists(path):
        errors.append(f"video.path '{path}' does not exist")

    weights = cfg.get("detector", {}).get("weights", "")
    # bare Ultralytics model names auto-download; anything path-shaped that
    # doesn't exist yet is very likely a typo, not a pending download
    if ("/" in weights or "\\" in weights) and not os.path.exists(weights):
        errors.append(f"detector.weights '{weights}' looks like a path but "
                      f"does not exist")

    floor = cfg.get("events", {}).get("candidate_floor")
    if floor is not None and not (0.0 <= floor <= 1.0):
        errors.append(f"events.candidate_floor={floor} must be in [0, 1]")

    for i, poly in enumerate(cfg.get("events", {}).get("restricted_zones") or []):
        if len(poly) < 3:
            errors.append(f"events.restricted_zones[{i}] has {len(poly)} points - "
                          f"a polygon needs at least 3 (use zones.py to draw one)")

    f = cfg.get("fuse", {})
    if "raise_threshold" in f and "clear_threshold" in f:
        if f["clear_threshold"] >= f["raise_threshold"]:
            errors.append(f"fuse.clear_threshold ({f['clear_threshold']}) must be "
                          f"< fuse.raise_threshold ({f['raise_threshold']}) or "
                          f"hysteresis never closes an alert")

    if out_path:
        out_dir = Path(out_path).parent or Path(".")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            probe = out_dir / ".write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as e:
            errors.append(f"output directory '{out_dir}' is not writable: {e}")

    if errors:
        raise ConfigError("Config validation failed:\n  - " + "\n  - ".join(errors))
