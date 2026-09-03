"""#30 (partial): retention. `out/` accumulates alerts.json, scene_fit.json,
normal_bank.npy, audit logs, demo.html - all of it derived from real footage
of real people. This deletes anything older than --days, so "how long do we
keep this" has an actual answer instead of "forever, by accident".

    python scripts/cleanup_out.py --days 30           # delete, after listing
    python scripts/cleanup_out.py --days 30 --dry-run  # list only
"""
import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out")
    ap.add_argument("--days", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cutoff = time.time() - a.days * 86400
    root = Path(a.dir)
    if not root.exists():
        print(f"{root} does not exist - nothing to clean")
        return

    old = [p for p in root.rglob("*") if p.is_file() and p.stat().st_mtime < cutoff]
    if not old:
        print(f"nothing older than {a.days} days in {root}")
        return

    for p in old:
        age_days = (time.time() - p.stat().st_mtime) / 86400
        print(f"{'[dry-run] would delete' if a.dry_run else '[delete]'} "
              f"{p} (age {age_days:.1f}d)")
        if not a.dry_run:
            p.unlink()

    print(f"{'would remove' if a.dry_run else 'removed'} {len(old)} file(s)")


if __name__ == "__main__":
    main()
