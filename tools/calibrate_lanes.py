"""Interactive per-camera ROI and counting-line calibration utility.

Run from the project root: python tools/calibrate_lanes.py
Left-click polygon vertices, press ENTER to finish the polygon, then left-click
both ends of the counting line. Press D to toggle crossing direction, R to reset,
S to save this lane, or Q to quit without saving the current lane.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "videos"
CONFIG_PATH = ROOT / "config" / "lanes.json"
LANES = [f"lane{i}" for i in range(1, 5)]


def main() -> None:
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        config = {}
    for lane in LANES:
        video = VIDEO_DIR / f"{lane}.mp4"
        if not video.is_file():
            print(f"Skipping {lane}: video not found: {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"Skipping {lane}: cannot read first frame")
            continue
        h, w = frame.shape[:2]
        points: list[tuple[int, int]] = []
        line: list[tuple[int, int]] = []
        direction = "down"
        polygon_done = False
        window = f"ROI Calibration - {lane}"
        def mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                if not polygon_done:
                    points.append((x, y))
                elif len(line) < 2:
                    line.append((x, y))
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(1280, w), min(800, h))
        cv2.setMouseCallback(window, mouse)
        print(f"\n{lane}: click 3+ polygon corners, ENTER to finish ROI, then click 2 line endpoints.")
        print("D toggles crossing direction; R resets; S saves; Q quits.")
        while True:
            display = frame.copy()
            for point in points:
                cv2.circle(display, point, 6, (0, 255, 255), -1)
            if len(points) >= 2:
                cv2.polylines(display, [np.array(points, dtype=np.int32)], polygon_done, (255, 180, 0), 2)
            for point in line:
                cv2.circle(display, point, 7, (0, 255, 0), -1)
            if len(line) == 2:
                cv2.line(display, line[0], line[1], (0, 255, 0), 3)
            cv2.putText(display, f"{lane}: ROI points={len(points)} | line points={len(line)} | direction={direction}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and len(points) >= 3:
                polygon_done = True
            elif key in (ord('d'), ord('D')):
                direction = "up" if direction == "down" else "down"
            elif key in (ord('r'), ord('R')):
                points.clear(); line.clear(); polygon_done = False; direction = "down"
            elif key in (ord('s'), ord('S')):
                if len(points) < 3 or len(line) != 2:
                    print("Need at least 3 ROI points and exactly 2 counting-line points before saving.")
                    continue
                config[lane] = {
                    "name": lane.replace("lane", "Lane "), "video": f"{lane}.mp4",
                    "roi": [[round(x/w, 6), round(y/h, 6)] for x,y in points],
                    "counting_line": [[round(x/w, 6), round(y/h, 6)] for x,y in line],
                    "count_direction": direction, "enabled": True,
                }
                CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
                CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
                print(f"Saved {lane} calibration to {CONFIG_PATH}")
                break
            elif key in (ord('q'), ord('Q'), 27):
                print(f"Skipped {lane}")
                break
        cv2.destroyWindow(window)
    cv2.destroyAllWindows()
    print("Calibration finished. Review config/lanes.json before starting the dashboard.")

if __name__ == "__main__":
    main()
