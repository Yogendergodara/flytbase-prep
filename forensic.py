"""F6: forensic reasoning - one paragraph summarising every alert inside a
window, citing timestamps. Text-only, reuses the already-computed alerts and
verdicts; no extra frames, no extra model.

    python forensic.py --alerts out/alerts.json --window 0 300
"""
import argparse, json, yaml
from pipeline.vlm_judge import build_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="out/alerts.json")
    ap.add_argument("--window", nargs=2, type=float, default=None,
                     help="t0 t1 in seconds; default is the whole alert set")
    a = ap.parse_args()

    data = json.load(open(a.alerts, encoding="utf-8"))
    cfg = data["config"]
    alerts = data["alerts"]

    if a.window:
        t0, t1 = a.window
        alerts = [x for x in alerts if t0 <= x["t_start"] <= t1]
    else:
        t0 = min((x["t_start"] for x in alerts), default=0.0)
        t1 = max((x["t_end"] for x in alerts), default=0.0)

    judge = build_judge(cfg)
    summary, err = judge.summarize(alerts, t0, t1)
    if err:
        print("refused:", err)
        return
    print(f"[forensic] window {t0:.1f}s-{t1:.1f}s, {len(alerts)} events:\n")
    print(summary)
    print("\n[forensic] verify every claim above against the footage before "
          "using it - a VLM will invent a detail if the prompt lets it.")


if __name__ == "__main__":
    main()
