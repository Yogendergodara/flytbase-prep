"""#29 (partial): "which code, which weights, which config produced this
alert" and "where is the audit trail". Not a metrics endpoint or distributed
tracing - there is no running service here for either of those to attach to;
this is the honest, scoped version for an offline CLI: a provenance block in
every output file, and one append-only JSONL line per alert.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path


def get_provenance(cfg):
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, timeout=3).stdout.strip() or None
    except Exception:
        git_hash = None   # not a git repo, or git unavailable - not fatal
    return {
        "git_commit": git_hash,
        "config_hash": cfg_hash,
        "detector_weights": cfg.get("detector", {}).get("weights"),
        "vlm_model_id": cfg.get("vlm", {}).get("model_id")
                        if cfg.get("vlm", {}).get("backend") != "none" else None,
        "generated_at": time.time(),
    }


def append_audit_log(alerts, path="out/audit.log.jsonl"):
    """One JSON line per alert, append-only. This - plus alerts.json itself -
    IS the audit trail for an offline tool with no running service to expose
    one from."""
    if not alerts:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for al in alerts:
            f.write(json.dumps({"logged_at": time.time(), **al}) + "\n")
