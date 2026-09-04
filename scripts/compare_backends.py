"""P16: verify the CPU/edge fallback actually works, on the same clip, back
to back, instead of trusting that `vlm.backend=smolvlm` runs just because it
parses. Run this ON KAGGLE (or wherever the real GPU/CPU environment is) -
it shells out to run.py, it does not import the pipeline directly, so it
exercises the exact same code path as the real CLI.

    python scripts/compare_backends.py --video data/sample.mp4
    python scripts/compare_backends.py --video data/sample.mp4 --backends qwen smolvlm --cpu-fallback
"""
import argparse
import json
import subprocess
import sys
import time


def run_backend(video, backend, out_path, extra_overrides, python_exe):
    cmd = [python_exe, "run.py", "--video", video,
           "--set", f"vlm.backend={backend}", *extra_overrides,
           "--out", out_path]
    print(f"\n=== running: {' '.join(cmd)} ===")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return {"backend": backend, "ok": False, "wall_seconds": round(wall, 1)}
    d = json.load(open(out_path, encoding="utf-8"))
    lat = d.get("vlm_latency_ms", [])
    return {
        "backend": backend,
        "ok": True,
        "wall_seconds": round(wall, 1),
        "n_alerts": len(d.get("alerts", [])),
        "n_candidates": len(d.get("candidates", [])),
        "vlm_latency_mean_ms": round(sum(lat) / len(lat), 1) if lat else None,
        "gpu_mem_gb": d.get("gpu_mem_gb"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--backends", nargs="+", default=["qwen", "smolvlm"])
    ap.add_argument("--cpu-fallback", action="store_true",
                    help="also test the last backend forced onto CPU "
                         "(detector.device=cpu detector.half=false) - "
                         "the venue-congestion scenario, not just a backend swap")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    results = []
    for b in a.backends:
        results.append(run_backend(a.video, b, f"out/compare_{b}.json", [], a.python))

    if a.cpu_fallback:
        b = a.backends[-1]
        results.append(run_backend(
            a.video, b, f"out/compare_{b}_cpu.json",
            ["detector.device=cpu", "detector.half=false"], a.python))
        results[-1]["backend"] = f"{b} (forced CPU)"

    print("\n=== comparison ===")
    header = f"{'backend':<22}{'ok':<6}{'wall_s':<9}{'alerts':<8}{'vlm_ms_mean':<13}{'gpu_gb':<8}"
    print(header)
    for r in results:
        print(f"{r['backend']:<22}{str(r['ok']):<6}{r['wall_seconds']:<9}"
              f"{r.get('n_alerts', '-'):<8}{str(r.get('vlm_latency_mean_ms')):<13}"
              f"{str(r.get('gpu_mem_gb')):<8}")

    if not all(r["ok"] for r in results):
        print("\nAt least one backend failed to run end-to-end - fix this "
              "before relying on it as a fallback. A fallback that has never "
              "produced alerts.json is not a fallback.")


if __name__ == "__main__":
    main()
