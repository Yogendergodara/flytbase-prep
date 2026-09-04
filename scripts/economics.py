"""P14: the economics headline - sustained FPS, peak GPU memory, extrapolated
feeds-per-GPU, and the VLM-call-rate ("N% of frames reach the VLM") claim
from the brief. Reads a completed out/alerts.json - run.py must have run
first (this script measures, it does not run the pipeline).

    python scripts/economics.py --alerts out/alerts.json
    python scripts/economics.py --alerts out/alerts.json --reference-gpu-mem-gb 16
"""
import argparse
import json


def compute(d, reference_gpu_mem_gb):
    n_frames = d["n_frames"]
    wall = d["wall_seconds"]
    fps = n_frames / wall if wall else None

    mem_gb = d.get("gpu_mem_gb")   # None on a CPU run - never coerced to 0
    feeds_per_gpu = (int(reference_gpu_mem_gb // mem_gb)
                     if mem_gb else None)

    # candidates = every event that actually reached the VLM (pre-suppression,
    # see run.py._write_out docstring) - this is the "1-5% of frames" claim,
    # measured, not asserted
    n_candidates = len(d.get("candidates", []))
    frames_per_event = d.get("config", {}).get("vlm", {}).get("frames_per_event", 0)
    vlm_touched_frames = n_candidates * frames_per_event
    vlm_frame_pct = (100.0 * vlm_touched_frames / n_frames) if n_frames else None

    return {
        "sustained_fps": round(fps, 2) if fps else None,
        "peak_gpu_mem_gb": mem_gb,
        "reference_gpu_mem_gb": reference_gpu_mem_gb,
        "feeds_per_gpu": feeds_per_gpu,
        "n_frames_total": n_frames,
        "n_events_judged_by_vlm": n_candidates,
        "vlm_touched_frames_approx": vlm_touched_frames,
        "vlm_frame_pct_approx": round(vlm_frame_pct, 2) if vlm_frame_pct is not None else None,
    }


def report(m):
    lines = []
    if m["sustained_fps"] is not None:
        lines.append(f"Pipeline: {m['sustained_fps']} FPS sustained, "
                      f"end-to-end, single feed, this machine")
    else:
        lines.append("Pipeline: FPS unavailable (wall_seconds was 0)")

    if m["peak_gpu_mem_gb"] is not None:
        lines.append(f"Peak GPU memory: {m['peak_gpu_mem_gb']} GB")
        if m["feeds_per_gpu"]:
            lines.append(f"Extrapolated: ~{m['feeds_per_gpu']} concurrent drone "
                         f"feeds per {m['reference_gpu_mem_gb']:.0f}GB GPU "
                         f"(memory-bound estimate, not measured concurrently)")
    else:
        lines.append("GPU memory: n/a (CPU run, or vlm.backend=none)")

    if m["vlm_frame_pct_approx"] is not None:
        lines.append(f"VLM call rate: {m['n_events_judged_by_vlm']} events judged "
                     f"(~{m['vlm_touched_frames_approx']} frames, "
                     f"~{m['vlm_frame_pct_approx']}% of {m['n_frames_total']} "
                     f"sampled frames) - the rest never reach a model")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="out/alerts.json")
    ap.add_argument("--reference-gpu-mem-gb", type=float, default=16.0,
                    help="the GPU class you're extrapolating feeds-per-GPU "
                         "against - e.g. 16 for a T4, 40 for an A100")
    ap.add_argument("--json", action="store_true", help="print raw numbers, not prose")
    a = ap.parse_args()

    d = json.load(open(a.alerts, encoding="utf-8"))
    m = compute(d, a.reference_gpu_mem_gb)
    if a.json:
        print(json.dumps(m, indent=2))
    else:
        print(report(m))


if __name__ == "__main__":
    main()
