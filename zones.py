"""Draw restricted-zone polygons by clicking on a still, and write them
straight into config.yaml.

The 09:00 runbook allots three minutes to "draw two or three zone polygons on
a still - one minute of work, disproportionate payoff". Without this you would
be hand-typing pixel coordinates under time pressure, which is exactly the
kind of thing that eats twenty minutes on the day.

    python zones.py --video theirs.mp4              # click, then it saves
    python zones.py --video theirs.mp4 --at 45       # grab the frame at 45s
    python zones.py --video theirs.mp4 --auto        # P17: pre-seed polygons
                                                      # from zero-shot scene
                                                      # classification, still
                                                      # requires 's' to save

  left click   add a point
  n            finish this polygon, start the next
  u            undo the last point
  s            save all polygons into config.yaml and quit
  q / ESC      quit without saving
"""
import argparse

import cv2
import yaml


def _auto_seed(frame_bgr, grid, confidence_floor):
    """P17: zero-shot classify a grid over the still, turn cells labelled
    'driving lane' or 'restricted/fenced' into candidate polygons. A
    SUGGESTION only - returned polygons still need 's' to actually save,
    identically to a hand-clicked one."""
    from pipeline.zone_classify import classify_regions
    frame_rgb = frame_bgr[:, :, ::-1]
    regions = classify_regions(frame_rgb, grid=tuple(grid),
                               confidence_floor=confidence_floor)
    polys = []
    for r in regions:
        if not r["intrusion_worthy"]:
            continue
        x0, y0, x1, y1 = r["bbox"]
        polys.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        print(f"[zones] auto: {r['label']} (conf={r['confidence']}) "
              f"-> proposed polygon {r['bbox']}")
    if not polys:
        print("[zones] auto: nothing above confidence_floor - draw manually")
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--at", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--auto", action="store_true",
                    help="P17: propose polygons via zero-shot scene "
                         "classification before the manual click loop. Also "
                         "triggered by config.yaml's zones.auto_classify=true "
                         "without passing this flag")
    ap.add_argument("--grid", nargs=2, type=int, default=None,
                    help="default: config.yaml zones.grid, else [4, 4]")
    ap.add_argument("--confidence-floor", type=float, default=None,
                    help="default: config.yaml zones.confidence_floor, else 0.3")
    a = ap.parse_args()

    # P17 fix: config.yaml's zones.* knobs were dead - CLI flags had their own
    # hardcoded defaults and never read the file. --auto (CLI) OR
    # zones.auto_classify (config) triggers auto-seeding; explicit CLI flags
    # win over config, which wins over the hardcoded [4, 4] / 0.3 fallback.
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    zcfg = cfg.get("zones", {})
    auto = a.auto or bool(zcfg.get("auto_classify"))
    grid = a.grid if a.grid is not None else zcfg.get("grid", [4, 4])
    confidence_floor = (a.confidence_floor if a.confidence_floor is not None
                        else zcfg.get("confidence_floor", 0.3))

    cap = cv2.VideoCapture(a.video)
    if a.at:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.at * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"could not read a frame from {a.video}")
        return

    polys, current = [], []
    if auto:
        polys = _auto_seed(frame, grid, confidence_floor)
        print(f"[zones] {len(polys)} auto-proposed polygon(s) loaded - "
              f"press 'u'/redraw to adjust, 's' to save, 'q' to discard all")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([int(x), int(y)])

    win = "zones - click points | n next | u undo | s save | q quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = frame.copy()
        for poly in polys:
            cv2.polylines(canvas, [_np(poly)], True, (0, 220, 200), 2)
        if current:
            cv2.polylines(canvas, [_np(current)], False, (0, 160, 255), 2)
            for px, py in current:
                cv2.circle(canvas, (px, py), 4, (0, 160, 255), -1)
        cv2.putText(canvas, f"{len(polys)} saved, {len(current)} points",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("quit without saving")
            break
        if key == ord("u") and current:
            current.pop()
        if key == ord("n") and len(current) >= 3:
            polys.append(current.copy())
            current.clear()
        if key == ord("s"):
            if len(current) >= 3:
                polys.append(current.copy())
            if not polys:
                print("no polygons to save")
                break
            cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
            cfg["events"]["restricted_zones"] = polys
            with open(a.config, "w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=None)
            print(f"[zones] wrote {len(polys)} polygon(s) into {a.config}")
            for i, p in enumerate(polys):
                print(f"  zone {i}: {len(p)} points")
            break

    cv2.destroyAllWindows()


def _np(points):
    import numpy as np
    return np.asarray(points, dtype=np.int32)


if __name__ == "__main__":
    main()
