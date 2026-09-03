"""F0: A/B the fine-tune against stock, day and night. Run on Kaggle/wherever
the weights and VisDrone val split live - not locally.

Decided by measured mAP, not by sentiment: keep stock if it wins, and say so
on the slide regardless of which one wins.

    python scripts/ab_weights.py --tuned weights/aerial_night/weights/best.pt \\
        --stock yolo11s.pt --data VisDrone.yaml --night-data VisDroneNight.yaml
"""
import argparse


def _val(weights, data, imgsz):
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data, imgsz=imgsz, verbose=False)
    return {"map50": round(float(m.box.map50), 4), "map5095": round(float(m.box.map), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--stock", default="yolo11s.pt")
    ap.add_argument("--data", required=True, help="day/full VisDrone-format YAML")
    ap.add_argument("--night-data", default=None,
                     help="YAML pointed at only the _night.* split, if you built one")
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()

    rows = [("tuned", "day", a.tuned, a.data), ("stock", "day", a.stock, a.data)]
    if a.night_data:
        rows += [("tuned", "night", a.tuned, a.night_data),
                 ("stock", "night", a.stock, a.night_data)]

    results = {}
    for name, cond, weights, data in rows:
        r = _val(weights, data, a.imgsz)
        results[(name, cond)] = r
        print(f"[{name:5s}/{cond:5s}] mAP50={r['map50']}  mAP50-95={r['map5095']}")

    print()
    for cond in ({c for _, c in results}):
        t = results.get(("tuned", cond))
        s = results.get(("stock", cond))
        if t and s:
            winner = "tuned" if t["map5095"] >= s["map5095"] else "stock"
            print(f"[verdict/{cond}] {winner} wins on mAP50-95 "
                  f"(tuned={t['map5095']} vs stock={s['map5095']})")


if __name__ == "__main__":
    main()
