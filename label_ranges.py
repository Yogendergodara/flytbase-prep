"""#24: mark anomalous time ranges on a video, interactively, in the exact
JSON shape eval_run.py expects. Ground truth does not exist until a person
looks at the footage and says so - this makes that fast rather than
hand-typing timestamps while scrubbing a video player separately.

    python label_ranges.py --video theirs.mp4 --out labels/theirs.json

  space        play / pause
  a / d        step back / forward one frame (while paused)
  [            mark the start of an anomalous range at the current time
  ]            close it - mark the end
  u            undo the last completed range
  s            save and quit
  q / ESC      quit without saving
"""
import argparse
import json

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        print(f"could not read frame count from {a.video}")
        return

    ranges, open_start = [], None
    playing = False
    idx = 0

    win = "label ranges - space play/pause | [ start ] end | u undo | s save | q quit"
    cv2.namedWindow(win)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1)
            playing = False
            continue

        t = idx / fps
        overlay = frame.copy()
        status = f"t={t:7.2f}s  frame {idx}/{total}  ranges={len(ranges)}"
        colour = (255, 255, 255)
        if open_start is not None:
            status += f"  [OPEN range from {open_start:.2f}s]"
            colour = (0, 255, 255)
        cv2.putText(overlay, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.imshow(win, overlay)

        key = cv2.waitKey(1 if playing else 0) & 0xFF
        if key in (ord("q"), 27):
            print("quit without saving")
            break
        if key == ord(" "):
            playing = not playing
        if key == ord("a") and not playing:
            idx = max(0, idx - 1)
        if key == ord("d") and not playing:
            idx = min(total - 1, idx + 1)
        if key == ord("["):
            open_start = t
        if key == ord("]") and open_start is not None:
            if t > open_start:
                ranges.append([round(open_start, 2), round(t, 2)])
            else:
                print("range end must be after its start - ignored")
            open_start = None
        if key == ord("u") and ranges:
            dropped = ranges.pop()
            print(f"undid range {dropped}")
        if key == ord("s"):
            json.dump({"anomalous_ranges": ranges, "video": a.video},
                      open(a.out, "w", encoding="utf-8"), indent=2)
            print(f"[label] wrote {len(ranges)} range(s) to {a.out}")
            break
        if playing:
            idx = min(total - 1, idx + 1)
            if idx == total - 1:
                playing = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
