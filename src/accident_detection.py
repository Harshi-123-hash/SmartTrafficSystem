from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import imageio_ffmpeg
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
    import imageio_ffmpeg

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.ambulance_dispatch import dispatch_ambulance
except Exception:
    dispatch_ambulance = None


# =====================================================
# 1. PATHS & PERFORMANCE CONFIGURATION
# =====================================================

VIDEO_PATH = Path(
    os.getenv("INPUT_VIDEO")
    or (BASE_DIR / "data" / "videos" / "accident_test.mp4")
)

OUTPUT_PATH = Path(
    os.getenv("OUTPUT_VIDEO")
    or (BASE_DIR / "results" / "videos" / "accident_detected_output_web.mp4")
)

PHOTOS_DIR = BASE_DIR / "results" / "accident_frames"

# COCO classes: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

DEFAULT_LATITUDE = 16.5080
DEFAULT_LONGITUDE = 80.6400

# Fast inference settings
YOLO_IMGSZ = 480
USE_HALF = torch.cuda.is_available()
FRAME_STRIDE = 2                     # Process every 2nd frame (~2x faster)

# =====================================================
# 2. 4-STEP ACCIDENT DETECTION THRESHOLDS
# =====================================================

HISTORY_FRAMES = 24
CONFIRMATION_FRAMES = 8              # Confirm across 8 original frames (~0.27s-0.5s)

# Normalized speed = center displacement per step / vehicle body diagonal
MOVING_SPEED = 0.020                 # Vehicle is actively moving
FAST_SPEED = 0.042                   # Vehicle is moving at road speed
ZERO_SPEED = 0.020                   # Vehicle is abruptly stopped / wrecked

# Crash Pattern Thresholds
OVERLAP_IOU_THRESHOLD = 0.14         # Physical overlap during impact
SPIN_ANGLE_DEG = 50.0                # Violent swerve / spin-out angle on impact
SIDEWAYS_ASPECT_MIN = 1.45           # Wide sideways / fallen-over profile (width / height)
AREA_COLLAPSE_RATIO = 0.42           # >= 42% sudden bounding-box area collapse (rollover / bike fall)

# Colors (BGR)
RED = (0, 0, 255)
GREEN = (0, 190, 0)
WHITE = (255, 255, 255)


def _resolve_yolo_model() -> Path | str:
    for name in ("yolov8n.pt", "yolo26n.pt", "yolo11n.pt"):
        candidate = BASE_DIR / name
        if candidate.is_file():
            return candidate
    return "yolov8n.pt"


def calculate_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = ax1 if ax1 > bx1 else bx1
    iy1 = ay1 if ay1 > by1 else by1
    ix2 = ax2 if ax2 < bx2 else bx2
    iy2 = ay2 if ay2 < by2 else by2

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, area_a + area_b - inter)


def _angle_diff_deg(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 2.0 or mag2 < 2.0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def _filter_duplicate_boxes(
    xyxy: np.ndarray, cls_ids: np.ndarray, confs: np.ndarray, track_ids: np.ndarray
) -> list[tuple[tuple[float, float, float, float], int, float, int]]:
    """Remove duplicate inner boxes on the same vehicle so a vehicle never collides with itself."""
    order = np.argsort(-confs)
    kept: list[tuple[tuple[float, float, float, float], int, float, int]] = []

    for idx in order:
        box = (float(xyxy[idx][0]), float(xyxy[idx][1]), float(xyxy[idx][2]), float(xyxy[idx][3]))
        cid = int(cls_ids[idx])
        conf = float(confs[idx])
        tid = int(track_ids[idx])

        duplicate = False
        b_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        for kbox, _, _, _ in kept:
            ix1 = box[0] if box[0] > kbox[0] else kbox[0]
            iy1 = box[1] if box[1] > kbox[1] else kbox[1]
            ix2 = box[2] if box[2] < kbox[2] else kbox[2]
            iy2 = box[3] if box[3] < kbox[3] else kbox[3]
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                k_area = max(1.0, (kbox[2] - kbox[0]) * (kbox[3] - kbox[1]))
                if (inter / min(b_area, k_area)) > 0.60:
                    duplicate = True
                    break

        if not duplicate:
            kept.append((box, cid, conf, tid))

    return kept


def _reencode_browser_mp4(raw_path: Path, final_path: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_bin:
        if raw_path != final_path:
            raw_path.replace(final_path)
        return

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(raw_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "24",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(final_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and final_path.is_file() and final_path.stat().st_size > 0:
        if raw_path != final_path and raw_path.exists():
            raw_path.unlink(missing_ok=True)
    elif raw_path != final_path and raw_path.exists():
        raw_path.replace(final_path)


def draw_status_overlay(frame: np.ndarray, status_text: str, is_accident: bool) -> np.ndarray:
    bg_color = RED if is_accident else GREEN
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)
    pad_x, pad_y = 14, 10
    x1, y1 = 16, 16
    x2 = x1 + tw + pad_x * 2
    y2 = y1 + th + baseline + pad_y * 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)
    cv2.putText(
        frame,
        status_text,
        (x1 + pad_x, y1 + th + pad_y),
        font,
        font_scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_visualizations(
    frame: np.ndarray,
    vehicles: dict[int, dict],
    involved_ids: set[int],
    accident_confirmed: bool,
) -> np.ndarray:
    """Highlight ONLY the vehicles involved in the crash in RED; keep all other vehicles GREEN."""
    h, w = frame.shape[:2]

    if accident_confirmed:
        cv2.rectangle(frame, (3, 3), (w - 4, h - 4), RED, 6)

    for tid, v in vehicles.items():
        x1, y1, x2, y2 = map(int, v["box"])

        if accident_confirmed and tid in involved_ids:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), RED, -1)
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 4)
            cv2.putText(
                frame,
                f"ACCIDENT ID:{tid}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                RED,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 70), 2)
            cv2.putText(
                frame,
                f"ID:{tid}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 70),
                1,
                cv2.LINE_AA,
            )

    return frame


def main() -> None:
    if not VIDEO_PATH.is_file():
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.unlink(missing_ok=True)

    raw_output_path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_raw.mp4")
    raw_output_path.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0

    effective_fps = fps / FRAME_STRIDE
    confirm_steps_needed = max(3, int(round(CONFIRMATION_FRAMES / FRAME_STRIDE)))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diag = math.hypot(width, height)

    model = YOLO(str(_resolve_yolo_model()))

    centers_hist: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))
    speeds_hist: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))
    aspects_hist: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))
    areas_hist: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))
    was_moving: defaultdict[int, bool] = defaultdict(bool)
    first_seen_step: dict[int, int] = {}

    occluded_moving_vehicles: dict[int, dict] = {}
    prev_vehicles: dict[int, dict] = {}
    collision_pair_ttl: dict[tuple[int, int], int] = {}

    crash_streak: defaultdict[int, int] = defaultdict(int)
    confirmed_involved_ids: set[int] = set()
    # Maps lost_wreck_id -> (cx, cy, diag, step_lost)
    wreck_positions: dict[int, tuple[float, float, float, int]] = {}

    annotated_frames_buffer: list[np.ndarray] = []
    accident_confirmed = False
    ambulance_dispatched = False
    snapshot_saved = False
    frame_number = 0
    step_number = 0
    latest_reason = ""

    print("Starting 4-step accident detection...")
    print("Input video:", VIDEO_PATH)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            if FRAME_STRIDE > 1 and (frame_number % FRAME_STRIDE != 1):
                continue

            step_number += 1

            # -------------------------------------------------
            # STEP 1: Detect & Track Vehicles with Persistent ID
            # -------------------------------------------------
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.22,
                iou=0.45,
                imgsz=YOLO_IMGSZ,
                half=USE_HALF,
                classes=VEHICLE_CLASS_IDS,
                verbose=False,
            )

            current_vehicles: dict[int, dict] = {}
            boxes = results[0].boxes if results else None

            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)
                conf_arr = boxes.conf.cpu().numpy()
                ids_arr = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else np.arange(1, len(xyxy_arr) + 1, dtype=int)
                )

                clean_dets = _filter_duplicate_boxes(xyxy_arr, cls_arr, conf_arr, ids_arr)

                for box, cid, conf, tid in clean_dets:
                    x1, y1, x2, y2 = box
                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    area = bw * bh
                    diag = max(20.0, math.hypot(bw, bh))
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    aspect = bw / bh

                    is_brand_new = tid not in first_seen_step
                    if is_brand_new:
                        first_seen_step[tid] = step_number

                    c_hist = centers_hist[tid]
                    s_hist = speeds_hist[tid]
                    a_hist = aspects_hist[tid]
                    ar_hist = areas_hist[tid]

                    # -------------------------------------------------
                    # STEP 2: Measure Movement (Speed & Direction Angle)
                    # -------------------------------------------------
                    if len(c_hist) >= 2:
                        ox, oy = c_hist[-2]
                        dx, dy = cx - ox, cy - oy
                        inst_speed = (math.hypot(dx, dy) / 2.0) / diag
                        direction_deg = math.degrees(math.atan2(dy, dx))
                    elif len(c_hist) == 1:
                        ox, oy = c_hist[-1]
                        dx, dy = cx - ox, cy - oy
                        inst_speed = math.hypot(dx, dy) / diag
                        direction_deg = math.degrees(math.atan2(dy, dx))
                    else:
                        inst_speed = 0.0
                        direction_deg = 0.0

                    in_collision = any(tid in pair for pair in collision_pair_ttl)
                    touching_edge = (
                        x1 <= width * 0.015
                        or y1 <= height * 0.015
                        or x2 >= width * 0.985
                        or y2 >= height * 0.985
                    )
                    smooth_exit = touching_edge and inst_speed > MOVING_SPEED and not in_collision and tid not in confirmed_involved_ids

                    c_hist.append((cx, cy))
                    if not touching_edge:
                        s_hist.append(inst_speed)
                        a_hist.append(aspect)
                        ar_hist.append(area)
                        if inst_speed >= MOVING_SPEED:
                            was_moving[tid] = True

                    # Re-associate ONLY if a brand-new ID spawns ALREADY STOPPED directly on top of a lost stationary wreck
                    if is_brand_new and not was_moving[tid] and inst_speed <= ZERO_SPEED:
                        for wid, (wx, wy, wdiag, wstep) in list(wreck_positions.items()):
                            if wid not in current_vehicles and (step_number - wstep) <= 6:
                                if math.hypot(cx - wx, cy - wy) <= wdiag * 0.30:
                                    confirmed_involved_ids.add(tid)
                                    del wreck_positions[wid]
                                    break

                    current_vehicles[tid] = {
                        "box": box,
                        "center": (cx, cy),
                        "diag": diag,
                        "area": area,
                        "speed": inst_speed,
                        "direction": direction_deg,
                        "aspect": aspect,
                        "class_id": cid,
                        "smooth_exit": smooth_exit,
                    }

            # Record any moving vehicle that vanished behind/into another vehicle (occlusion crash)
            for old_tid, old_v in prev_vehicles.items():
                if old_tid not in current_vehicles and was_moving[old_tid] and not old_v["smooth_exit"]:
                    for cur_tid, cur_v in current_vehicles.items():
                        if calculate_iou(old_v["box"], cur_v["box"]) >= 0.08 or (
                            math.hypot(
                                old_v["center"][0] - cur_v["center"][0],
                                old_v["center"][1] - cur_v["center"][1],
                            )
                            <= (old_v["diag"] + cur_v["diag"]) * 0.85
                        ):
                            occluded_moving_vehicles[old_tid] = {
                                "center": old_v["center"],
                                "diag": old_v["diag"],
                                "occluder_id": cur_tid,
                                "step": step_number,
                            }
                            break

            max_occ_steps = int(round(4.0 * effective_fps))
            for otid in list(occluded_moving_vehicles.keys()):
                if step_number - occluded_moving_vehicles[otid]["step"] > max_occ_steps:
                    del occluded_moving_vehicles[otid]

            for pair in list(collision_pair_ttl.keys()):
                collision_pair_ttl[pair] -= 1
                if collision_pair_ttl[pair] <= 0:
                    del collision_pair_ttl[pair]

            # -------------------------------------------------
            # STEP 3: Check for Crash Pattern
            # -------------------------------------------------
            frame_suspect_ids: set[int] = set()
            active_ids = list(current_vehicles.keys())

            # (A) Multi-Vehicle Overlap
            for i in range(len(active_ids)):
                id_a = active_ids[i]
                va = current_vehicles[id_a]

                for j in range(i + 1, len(active_ids)):
                    id_b = active_ids[j]
                    vb = current_vehicles[id_b]

                    if (id_a in confirmed_involved_ids) ^ (id_b in confirmed_involved_ids):
                        continue

                    reach = (va["diag"] + vb["diag"]) * 0.80
                    if abs(va["center"][0] - vb["center"][0]) > reach or abs(va["center"][1] - vb["center"][1]) > reach:
                        continue

                    iou = calculate_iou(va["box"], vb["box"])
                    if iou >= OVERLAP_IOU_THRESHOLD:
                        sa = list(speeds_hist[id_a])
                        sb = list(speeds_hist[id_b])
                        if (sa and max(sa) >= MOVING_SPEED) or (sb and max(sb) >= MOVING_SPEED) or was_moving[id_a] or was_moving[id_b]:
                            pair = (min(id_a, id_b), max(id_a, id_b))
                            collision_pair_ttl[pair] = confirm_steps_needed + 10

            # Evaluate post-overlap crash behavior ONLY on the colliding pair
            for (id_a, id_b) in list(collision_pair_ttl.keys()):
                va = current_vehicles.get(id_a)
                vb = current_vehicles.get(id_b)
                if va is None and vb is None:
                    continue

                sa = list(speeds_hist[id_a])
                sb = list(speeds_hist[id_b])
                prior_a = max(sa) if sa else (MOVING_SPEED if was_moving[id_a] else 0.0)
                prior_b = max(sb) if sb else (MOVING_SPEED if was_moving[id_b] else 0.0)

                now_a = va["speed"] if va is not None else 0.0
                now_b = vb["speed"] if vb is not None else 0.0

                if (prior_a >= MOVING_SPEED or prior_b >= MOVING_SPEED):
                    if va is not None and now_a <= ZERO_SPEED * 1.4:
                        frame_suspect_ids.add(id_a)
                        latest_reason = f"Collision + Sudden Stop (ID:{id_a} & ID:{id_b})"
                    if vb is not None and now_b <= ZERO_SPEED * 1.4:
                        frame_suspect_ids.add(id_b)
                        latest_reason = f"Collision + Sudden Stop (ID:{id_a} & ID:{id_b})"

            # (B) Occluded Crash Reappearance (e.g., silver sedan hit by bus -> reappears stopped as ID:18)
            for tid, v in current_vehicles.items():
                if v["smooth_exit"]:
                    continue

                is_stopped_wreck = (
                    v["speed"] <= ZERO_SPEED * 1.6
                    and not was_moving[tid]
                    and first_seen_step.get(tid, 1) > 3
                )
                if is_stopped_wreck:
                    for otid, occ in list(occluded_moving_vehicles.items()):
                        if tid == occ["occluder_id"]:
                            continue
                        if first_seen_step[tid] >= occ["step"]:
                            dist = math.hypot(v["center"][0] - occ["center"][0], v["center"][1] - occ["center"][1])
                            if dist <= frame_diag * 0.50:
                                frame_suspect_ids.add(tid)
                                # If collision was already confirmed when the bus hit the car, immediately mark the revealed wreck RED
                                if accident_confirmed:
                                    confirmed_involved_ids.add(tid)
                                latest_reason = f"Occluded Impact + Wrecked Stop (ID:{tid})"
                                break

                # (C) Single-Vehicle Crash / Rollover / Bike Fall-Over / Violent Swerve-to-Stop
                sh = list(speeds_hist[tid])
                ch = list(centers_hist[tid])
                ah = list(aspects_hist[tid])
                arh = list(areas_hist[tid])

                if len(ah) >= 4 and len(arh) >= 4 and was_moving[tid]:
                    base_aspect = float(np.median(ah[:2]))
                    curr_aspect = v["aspect"]
                    base_area = float(np.median(arh[:2]))
                    curr_area = v["area"]

                    aspect_jump = (curr_aspect - base_aspect) / max(0.2, base_aspect)
                    area_drop = (base_area - curr_area) / max(1.0, base_area)

                    if (curr_aspect >= SIDEWAYS_ASPECT_MIN and aspect_jump >= 0.40) or (
                        area_drop >= AREA_COLLAPSE_RATIO and v["speed"] <= ZERO_SPEED * 1.4
                    ):
                        frame_suspect_ids.add(tid)
                        latest_reason = f"Vehicle Rollover / Fall-Over (ID:{tid})"

                if len(ch) >= 5 and len(sh) >= 5 and was_moving[tid]:
                    deflection = _angle_diff_deg(
                        (ch[-3][0] - ch[-5][0], ch[-3][1] - ch[-5][1]),
                        (ch[-1][0] - ch[-3][0], ch[-1][1] - ch[-3][1]),
                    )
                    if max(sh[:-1]) >= FAST_SPEED and deflection >= SPIN_ANGLE_DEG and v["speed"] <= ZERO_SPEED * 1.3:
                        frame_suspect_ids.add(tid)
                        latest_reason = f"Spin-Out + Impact Stop (ID:{tid}, {deflection:.0f}deg)"

            # -------------------------------------------------
            # STEP 4: Confirm and Display (Red box ONLY on crashed vehicle(s))
            # -------------------------------------------------
            for tid in list(crash_streak.keys()):
                if tid not in frame_suspect_ids:
                    crash_streak[tid] = max(0, crash_streak[tid] - 1)
                    if crash_streak[tid] == 0:
                        del crash_streak[tid]

            for tid in frame_suspect_ids:
                crash_streak[tid] += 1
                if crash_streak[tid] >= confirm_steps_needed:
                    confirmed_involved_ids.add(tid)
                    v = current_vehicles[tid]
                    wreck_positions[tid] = (v["center"][0], v["center"][1], v["diag"], step_number)
                    if not accident_confirmed:
                        accident_confirmed = True
                        print(
                            f"ACCIDENT DETECTED at frame {frame_number} "
                            f"(time {frame_number / fps:.2f}s) | {latest_reason}"
                        )

                        if not ambulance_dispatched and dispatch_ambulance is not None:
                            ambulance_dispatched = True
                            try:
                                disp = dispatch_ambulance(
                                    DEFAULT_LATITUDE,
                                    DEFAULT_LONGITUDE,
                                    traffic_level="HIGH",
                                )
                                print(f"Ambulance dispatched: {disp}")
                            except Exception as exc:
                                print(f"Ambulance dispatch warning: {exc}")

            for tid in list(confirmed_involved_ids):
                if tid in current_vehicles:
                    v = current_vehicles[tid]
                    wreck_positions[tid] = (v["center"][0], v["center"][1], v["diag"], step_number)

            if accident_confirmed:
                frame = draw_visualizations(frame, current_vehicles, confirmed_involved_ids, True)
                frame = draw_status_overlay(frame, "ACCIDENT DETECTED", is_accident=True)
                if not snapshot_saved:
                    snap_file = PHOTOS_DIR / f"accident_frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(snap_file), frame)
                    snapshot_saved = True
            else:
                frame = draw_visualizations(frame, current_vehicles, set(), False)
                frame = draw_status_overlay(frame, "No Accident", is_accident=False)

            annotated_frames_buffer.append(frame)
            prev_vehicles = current_vehicles

    finally:
        cap.release()

    if accident_confirmed and annotated_frames_buffer:
        writer = cv2.VideoWriter(
            str(raw_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            effective_fps,
            (width, height),
        )
        if writer.isOpened():
            for out_frame in annotated_frames_buffer:
                writer.write(out_frame)
            writer.release()
            _reencode_browser_mp4(raw_output_path, OUTPUT_PATH)
    else:
        raw_output_path.unlink(missing_ok=True)
        OUTPUT_PATH.unlink(missing_ok=True)

    print("Frames processed:", frame_number)
    print("Accident detected:", accident_confirmed)


if __name__ == "__main__":
    main()