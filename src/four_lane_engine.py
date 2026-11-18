"""Four-camera traffic pipeline: Strict Interior ROI + Heavy-Vehicle Box Lock + Priority Scheduling (Ambulance -> High -> Medium -> Low)."""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

LOG = logging.getLogger("itms.four_lane")


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, here.parent.parent, Path.cwd()):
        if (parent / "app.py").is_file():
            return parent
    return here.parent.parent


BASE_DIR = _find_project_root()
VIDEO_DIR = BASE_DIR / "data" / "videos"
MODEL_PATH = BASE_DIR / "yolov8n.pt"
FALLBACK_MODEL_PATH = BASE_DIR / "yolo26n.pt"
AMBULANCE_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
FALLBACK_AMBULANCE_PATH = BASE_DIR / "models" / "ambulance_best.pt"
CONFIG_DIR = BASE_DIR / "config"
ROIS_PATH = CONFIG_DIR / "rois.json"
LEGACY_CONFIG_PATH = CONFIG_DIR / "lanes.json"

LANES = ["lane1", "lane2", "lane3", "lane4"]

# COCO vehicle classes: 2=Car, 3=Motorcycle (Bike), 5=Bus, 7=Truck
COCO_VEHICLES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
COCO_VEHICLE_IDS = list(COCO_VEHICLES.keys())
VEHICLE_CLASS_NAMES = tuple(COCO_VEHICLES.values())

# Weighted Occupancy Score per vehicle class
OCCUPANCY_WEIGHTS = {
    "Motorcycle": 1,
    "Car": 2,
    "Bus": 4,
    "Truck": 4,
}

# Distinct BGR colours for Car, Truck, Motorcycle, Bus
VEHICLE_COLORS = {
    "Car": (0, 220, 70),         # Green
    "Truck": (0, 140, 255),      # Orange
    "Motorcycle": (255, 220, 0), # Cyan / Sky Blue
    "Bus": (220, 50, 220),       # Magenta / Purple
}
AMBULANCE_RED = (0, 0, 255)

QUANTIZE_MODE = "fp16" if torch.cuda.is_available() else None
YOLO_INPUT_SIZE = 640
CONFIDENCE = 0.22
MIN_HITS_TO_COUNT = 3                # Vehicle must be tracked across >= 3 frames before counting as passed

# High-confidence ambulance detection
AMBULANCE_IMAGE_SIZE = 640
AMBULANCE_CONFIDENCE = 0.65          # High confidence required to detect ambulance
AMBULANCE_STREAK_REQUIRED = 3        # Must hold >= 0.65 across 3 consecutive checks to confirm
AMBULANCE_BOX_PERSIST_SEC = 0.5
AMBULANCE_INFER_EVERY_N_FRAMES = 3
YELLOW_DURATION = 3.0                # 3 video seconds -> round(3.0 * lane_fps) frames
AMBULANCE_CLEAR_SECONDS = 3.0        # 3 video seconds -> round(3.0 * lane_fps) frames
INFERENCE_EVERY_N_FRAMES = 2
JPEG_QUALITY = 75
ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

_AMBULANCE_MODEL = None
_AMBULANCE_CLASS_IDS: set[int] = set()
_AMBULANCE_LOCK = threading.Lock()
_AMBULANCE_LOAD_ERROR = None


def _resolve_main_model_path() -> Path | str:
    for candidate in (
        MODEL_PATH,
        Path.cwd() / "yolov8n.pt",
        FALLBACK_MODEL_PATH,
        Path.cwd() / "yolo26n.pt",
    ):
        if candidate.is_file():
            return candidate
    return "yolov8n.pt"


def _extract_ambulance_class_ids(model: YOLO) -> set[int]:
    """Identify ambulance class IDs from trained weights."""
    names = getattr(model, "names", {}) or {}
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}

    ambulance_ids: set[int] = set()
    for class_id, class_name in names.items():
        normalized_name = str(class_name).lower().strip()
        if (
            "ambulance" in normalized_name
            or "storbow" in normalized_name
            or "emergency vehicle" in normalized_name
            or "amb" in normalized_name
        ):
            ambulance_ids.add(int(class_id))

    if not ambulance_ids and len(names) == 1:
        ambulance_ids.add(int(next(iter(names.keys()))))

    return ambulance_ids


def _get_ambulance_model():
    global _AMBULANCE_MODEL, _AMBULANCE_CLASS_IDS, _AMBULANCE_LOAD_ERROR
    with _AMBULANCE_LOCK:
        if _AMBULANCE_MODEL is not None:
            return _AMBULANCE_MODEL
        if _AMBULANCE_LOAD_ERROR:
            return None

        amb_path = None
        for cand in (
            AMBULANCE_MODEL_PATH,
            Path.cwd() / "runs" / "detect" / "train" / "weights" / "best.pt",
            FALLBACK_AMBULANCE_PATH,
            Path.cwd() / "models" / "ambulance_best.pt",
        ):
            if cand.is_file():
                amb_path = cand
                break

        if amb_path is None:
            _AMBULANCE_LOAD_ERROR = f"Ambulance weights not found: {AMBULANCE_MODEL_PATH}"
            return None
        try:
            loaded = YOLO(str(amb_path))
            cls_ids = _extract_ambulance_class_ids(loaded)
            if not cls_ids:
                _AMBULANCE_LOAD_ERROR = f"Ambulance class not found in {amb_path} (names={loaded.names})"
                return None
            _AMBULANCE_MODEL = loaded
            _AMBULANCE_CLASS_IDS = cls_ids
            return _AMBULANCE_MODEL
        except Exception as exc:
            _AMBULANCE_LOAD_ERROR = str(exc)
            return None


def _default_config() -> dict[str, Any]:
    return {
        f"lane{i}": {
            "name": f"Lane {i}",
            "video": f"lane{i}.mp4",
            "roi": None,
            "counting_line": None,
            "count_direction": "down",
            "enabled": True,
        }
        for i in range(1, 5)
    }


def save_rois(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config, indent=2)
    ROIS_PATH.write_text(payload, encoding="utf-8")
    LEGACY_CONFIG_PATH.write_text(payload, encoding="utf-8")


def load_config() -> dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    target = ROIS_PATH if ROIS_PATH.exists() else (LEGACY_CONFIG_PATH if LEGACY_CONFIG_PATH.exists() else ROIS_PATH)
    if not target.exists():
        save_rois(_default_config())
        target = ROIS_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid ROI configuration {target}: {exc}") from exc

    config = _default_config()
    for lane_id in LANES:
        entry = raw.get(lane_id)
        if isinstance(entry, list):
            config[lane_id]["roi"] = entry
        elif isinstance(entry, dict):
            config[lane_id].update(entry)
            if not config[lane_id].get("roi"):
                for alt_key in ("points", "polygon", "coordinates"):
                    if isinstance(entry.get(alt_key), list) and len(entry[alt_key]) >= 3:
                        config[lane_id]["roi"] = entry[alt_key]
                        break
    return config


class LaneWorker:
    def __init__(
        self,
        lane_id: str,
        config: dict[str, Any],
        model_path: Path | str,
        video_override: str | Path | None = None,
    ):
        self.lane_id = lane_id
        self.config = dict(config)
        # Always preserve ROI and counting_line from config/rois.json for both default and uploaded videos
        if video_override is not None:
            self.video_path = Path(video_override)
        else:
            self.video_path = self._resolve_video_path(config.get("video", f"{lane_id}.mp4"))
        self.model_path = model_path
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.jpeg: bytes | None = None

        # Cached pixel-space ROI polygon & counting line
        self._cached_roi_key: tuple[int, int, str] | None = None
        self._cached_roi_poly: np.ndarray | None = None
        self._cached_line_key: tuple[int, int, str] | None = None
        self._cached_line_coords: tuple[float, float, float, float, float] | None = None

        self.video_fps: float = 25.0
        self.total_frames: int = 0
        self.video_duration: float = 0.0
        self.current_frame_no: int = 0

        self.status: dict[str, Any] = {
            "lane": lane_id,
            "running": False,
            "completed": False,
            "video_eof": False,
            "paused": True,
            "error": None,
            "unique_total": 0,
            "crossed": 0,
            "vehicles_passed": 0,
            "waiting": 0,
            "occupancy_score": 0,
            "yellow_max_score": 0,
            "yellow_max_vehicles": 0,
            "decided_green_seconds": 0,
            "density": "PENDING",
            "locked_density": None,
            "ambulance": False,
            "ambulance_confidence": 0.0,
            "ambulance_priority": False,
            "ambulance_count": 0,
            "accident": False,
            "fps": 0.0,
            "video_fps": 25.0,
            "video_duration": 0.0,
            "frame": 0,
            "total_frames": 0,
            "signal": "RED",
            "green_remaining": 0,
            "remaining_seconds": 0,
            "remaining_frames": 0,
            "timer": 0,
            "roi_calibrated": bool(self.config.get("roi")),
            "classes": {name: 0 for name in VEHICLE_CLASS_NAMES},
        }

        self.counted_ids: set[int] = set()
        self.counted_class_by_id: dict[int, str] = {}

        # Fixed single-class lock per vehicle (never downgrades Truck/Bus -> Car)
        self.locked_track_class: dict[int, str] = {}
        self.locked_track_conf: dict[int, float] = {}
        self.track_hits: defaultdict[int, int] = defaultdict(int)

        self.last_seen: dict[int, int] = {}
        self.current_tracks: dict[int, tuple[int, int, str]] = {}
        self.previous_centers: dict[int, tuple[int, int]] = {}
        self.crossed_ids: set[int] = set()

        self._last_frame_time = time.monotonic()
        self._last_inference_frame = -999
        self._last_boxes: list = []

        # Per-lane video-frame-synced timer state
        self._signal = "RED"
        self._paused_after_green = False
        self._turn_completed = False
        self._has_entered_green = False
        self._current_occupancy_score = 0
        self._yellow_max_score = 0
        self._yellow_max_vehicles = 0
        self._locked_density: str | None = None
        self._decided_green_seconds = 0
        self._phase_frames_remaining = 0
        self._phase_expired = False

        # Initial frame scan metrics for priority ordering (Ambulance -> High -> Medium -> Low)
        self._initial_scan_done = threading.Event()
        self._initial_occupancy_score: int = 0
        self._initial_density_level: str = "LOW"
        self._initial_ambulance_detected: bool = False

        # High-confidence ambulance state
        self._ambulance_present = False
        self._ambulance_priority_active = False
        self._ambulance_streak = 0
        self._ambulance_recent_confs: deque[float] = deque(maxlen=AMBULANCE_STREAK_REQUIRED)
        self._ambulance_prev_center: tuple[int, int] | None = None
        self._ambulance_missing_frames = 0
        self._ambulance_events = 0
        self._ambulance_event_active = False
        self._last_ambulance_bbox: tuple[int, int, int, int] | None = None
        self._last_ambulance_conf: float = 0.0
        self._last_ambulance_frame: int = -999
        self._ambulance_detected_now: bool = False

    @staticmethod
    def _resolve_video_path(filename: str) -> Path:
        raw_path = Path(filename)
        if raw_path.is_file():
            return raw_path
        for base in (VIDEO_DIR, Path.cwd() / "data" / "videos"):
            cand = base / filename
            if cand.is_file():
                return cand
        return VIDEO_DIR / filename

    def _get_roi_polygon(self, width: int, height: int) -> np.ndarray | None:
        roi = self.config.get("roi")
        if not roi or len(roi) < 3:
            return None
        key = (width, height, repr(roi))
        if self._cached_roi_key != key:
            pts = []
            for px, py in roi:
                fx, fy = float(px), float(py)
                # Support both normalized [0..1] coordinates and pixel coordinates
                x_px = int(fx * width) if fx <= 1.0 else int(fx)
                y_px = int(fy * height) if fy <= 1.0 else int(fy)
                pts.append([x_px, y_px])
            self._cached_roi_poly = np.array(pts, dtype=np.int32)
            self._cached_roi_key = key
        return self._cached_roi_poly

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self._initial_scan_done.clear()
        self.thread = threading.Thread(target=self._run, name=f"{self.lane_id}-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._sync_timer_fields()
            snap = dict(self.status)
            snap["classes"] = dict(self.status["classes"])
            return snap

    def get_jpeg(self) -> bytes | None:
        with self.lock:
            return self.jpeg

    def _frames_to_seconds(self, frames_left: int) -> int:
        if frames_left <= 0:
            return 0
        fps = self.video_fps if self.video_fps > 0.0 else 25.0
        return int(math.ceil(frames_left / fps))

    def _sync_timer_fields(self) -> None:
        sec = self._frames_to_seconds(self._phase_frames_remaining)
        self.status["remaining_frames"] = max(0, self._phase_frames_remaining)
        self.status["timer"] = sec
        self.status["green_remaining"] = sec
        self.status["remaining_seconds"] = sec

    @staticmethod
    def _score_to_density(score: int) -> tuple[str, int]:
        if score < 10:
            return "LOW", 10
        elif score < 25:
            return "MEDIUM", 20
        else:
            return "HIGH", 30

    def _classify_from_occupancy_score(self) -> tuple[str, int]:
        return self._score_to_density(self._yellow_max_score)

    def get_priority_key(self) -> tuple[int, int]:
        """
        Scheduling priority tuple (higher sorts first):
          1) Ambulance present or initially detected -> rank 4
          2) HIGH traffic density (score >= 25) -> rank 3
          3) MEDIUM traffic density (10 <= score < 25) -> rank 2
          4) LOW traffic density (score < 10) -> rank 1
        Tie-breaker: exact occupancy score.
        """
        with self.lock:
            has_amb = self._ambulance_present or self._initial_ambulance_detected
            score = max(self._yellow_max_score, self._current_occupancy_score, self._initial_occupancy_score)
            level, _ = self._score_to_density(score)
            if has_amb:
                return (4, score)
            if level == "HIGH":
                return (3, score)
            if level == "MEDIUM":
                return (2, score)
            return (1, score)

    def begin_yellow_phase(self, resume_frames: int | None = None) -> None:
        with self.lock:
            latest_cfg = load_config().get(self.lane_id, {})
            if latest_cfg.get("roi"):
                self.config["roi"] = latest_cfg["roi"]
                self._cached_roi_key = None
                self.status["roi_calibrated"] = True
            if latest_cfg.get("counting_line"):
                self.config["counting_line"] = latest_cfg["counting_line"]
                self._cached_line_key = None
            if latest_cfg.get("count_direction"):
                self.config["count_direction"] = latest_cfg["count_direction"]

            fps = self.video_fps if self.video_fps > 0.0 else 25.0
            target_frames = int(resume_frames) if resume_frames is not None else max(1, int(round(YELLOW_DURATION * fps)))
            self._paused_after_green = False
            self._signal = "YELLOW"
            if resume_frames is None:
                self._current_occupancy_score = self._initial_occupancy_score
                self._yellow_max_score = self._initial_occupancy_score
                self._yellow_max_vehicles = 0
                self._locked_density = None
                self._decided_green_seconds = 0
            self._phase_frames_remaining = max(1, target_frames)
            self._phase_expired = False
            self._last_boxes = []
            self._last_inference_frame = -999

            self.status["signal"] = "YELLOW"
            self.status["paused"] = False
            self.status["locked_density"] = self._locked_density
            self.status["density"] = "AMBULANCE PRIORITY" if self._ambulance_present else "ANALYSING"
            self.status["occupancy_score"] = self._current_occupancy_score
            self.status["yellow_max_score"] = self._yellow_max_score
            self.status["yellow_max_vehicles"] = self._yellow_max_vehicles
            self.status["decided_green_seconds"] = self._decided_green_seconds
            self._sync_timer_fields()

    def decide_and_begin_green(self) -> tuple[str, int]:
        with self.lock:
            level, green_sec = self._classify_from_occupancy_score()
            fps = self.video_fps if self.video_fps > 0.0 else 25.0
            green_frames = max(1, int(round(green_sec * fps)))

            self._has_entered_green = True
            self._locked_density = level
            self._decided_green_seconds = green_sec
            self._phase_frames_remaining = green_frames
            self._phase_expired = False
            self._signal = "GREEN"
            self._paused_after_green = False

            self.status["signal"] = "GREEN"
            self.status["paused"] = False
            self.status["locked_density"] = level
            self.status["density"] = "AMBULANCE PRIORITY" if self._ambulance_present else level
            self.status["decided_green_seconds"] = green_sec
            self._sync_timer_fields()
            return level, green_sec

    def resume_green_phase(self, remaining_frames: int) -> None:
        with self.lock:
            self._paused_after_green = False
            self._ambulance_priority_active = False
            self._signal = "GREEN"
            self._phase_frames_remaining = max(1, int(remaining_frames))
            self._phase_expired = False
            self.status["signal"] = "GREEN"
            self.status["paused"] = False
            self.status["ambulance_priority"] = False
            if self._locked_density:
                self.status["density"] = self._locked_density
                self.status["locked_density"] = self._locked_density
            self._sync_timer_fields()

    def set_ambulance_priority_green(self) -> None:
        with self.lock:
            self._paused_after_green = False
            self._ambulance_priority_active = True
            self._signal = "GREEN"
            self._phase_frames_remaining = 0
            self._phase_expired = False
            self.status["signal"] = "GREEN"
            self.status["paused"] = False
            self.status["ambulance"] = True
            self.status["ambulance_priority"] = True
            self.status["density"] = "AMBULANCE PRIORITY"
            self._sync_timer_fields()

    def set_red_paused(self) -> None:
        with self.lock:
            self._signal = "RED"
            self._ambulance_priority_active = False
            self._phase_frames_remaining = 0
            self._phase_expired = False
            self._last_boxes = []
            self.status["signal"] = "RED"
            self.status["paused"] = True
            self.status["ambulance_priority"] = False
            if self._ambulance_present:
                self.status["density"] = "AMBULANCE PRIORITY"
            elif self._locked_density:
                self.status["density"] = self._locked_density
                self.status["locked_density"] = self._locked_density
            elif not self._turn_completed:
                self.status["density"] = self._initial_density_level
            self._sync_timer_fields()

    def pause_temporarily(self) -> None:
        with self.lock:
            self._signal = "RED"
            self._ambulance_priority_active = False
            self._paused_after_green = True
            self._phase_frames_remaining = 0
            self._phase_expired = False
            self._last_boxes = []
            self.status["signal"] = "RED"
            self.status["paused"] = True
            self.status["ambulance_priority"] = False
            if self._ambulance_present:
                self.status["density"] = "AMBULANCE PRIORITY"
            elif self._locked_density:
                self.status["density"] = self._locked_density
                self.status["locked_density"] = self._locked_density
            self._sync_timer_fields()

    def pause_after_green(self) -> None:
        with self.lock:
            if not self._locked_density:
                lvl, sec = self._classify_from_occupancy_score()
                self._locked_density = lvl
                if not self._decided_green_seconds:
                    self._decided_green_seconds = sec

            self._signal = "RED"
            self._ambulance_priority_active = False
            self._paused_after_green = True
            self._turn_completed = True
            self._phase_frames_remaining = 0
            self._phase_expired = True
            self._last_boxes = []

            self.status["signal"] = "RED"
            self.status["paused"] = True
            self.status["completed"] = True
            self.status["video_eof"] = True
            self.status["ambulance_priority"] = False
            self.status["locked_density"] = self._locked_density
            self.status["density"] = self._locked_density
            self.status["decided_green_seconds"] = self._decided_green_seconds
            self._sync_timer_fields()

    def _box_in_roi(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> bool:
        """Strict interior ROI check using cached polygon from rois.json and vehicle road-contact anchor."""
        polygon = self._get_roi_polygon(width, height)
        if polygon is None:
            return True
        cx = (x1 + x2) >> 1
        contact_y = y1 + int((y2 - y1) * 0.92)
        return cv2.pointPolygonTest(polygon, (cx, contact_y), False) >= 0

    def _crossed_line(self, track_id: int, x: int, y: int, width: int, height: int) -> bool:
        line = self.config.get("counting_line")
        previous = self.previous_centers.get(track_id)
        self.previous_centers[track_id] = (x, y)
        if not line or len(line) != 2:
            return True

        lkey = (width, height, repr(line))
        if self._cached_line_key != lkey:
            (x1n, y1n), (x2n, y2n) = line
            fx1, fy1, fx2, fy2 = float(x1n), float(y1n), float(x2n), float(y2n)
            x1 = fx1 * width if fx1 <= 1.0 else fx1
            y1 = fy1 * height if fy1 <= 1.0 else fy1
            x2 = fx2 * width if fx2 <= 1.0 else fx2
            y2 = fy2 * height if fy2 <= 1.0 else fy2
            self._cached_line_coords = (x1, y1, x2, y2, math.hypot(x2 - x1, y2 - y1))
            self._cached_line_key = lkey

        x1, y1, x2, y2, line_len = self._cached_line_coords
        if line_len > 1.0:
            dist = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / line_len
            if dist <= 18.0:
                return True

        if previous is None:
            return False

        px, py = previous
        old_side = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        new_side = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        direction = str(self.config.get("count_direction", "down")).lower()
        return (old_side <= 0 < new_side) if direction == "down" else (old_side >= 0 > new_side)

    def _resolve_vehicle_class(self, track_id: int, raw_label: str, score: float) -> str:
        """Lock a SINGLE fixed class per vehicle (never downgrades Truck/Bus -> Car)."""
        existing = self.locked_track_class.get(track_id)
        if existing is None:
            self.locked_track_class[track_id] = raw_label
            self.locked_track_conf[track_id] = score
            return raw_label

        if existing in ("Truck", "Bus") and raw_label == "Car":
            return existing

        if existing == "Car" and raw_label in ("Truck", "Bus") and score >= 0.35:
            self.locked_track_class[track_id] = raw_label
            self.locked_track_conf[track_id] = score
            return raw_label

        if score > self.locked_track_conf[track_id]:
            self.locked_track_class[track_id] = raw_label
            self.locked_track_conf[track_id] = score

        return self.locked_track_class[track_id]

    @staticmethod
    def _suppress_overlapping_boxes(
        xyxy: np.ndarray, ids: np.ndarray | list[int], cls: np.ndarray, conf: np.ndarray
    ) -> list[tuple[np.ndarray, int, int, float]]:
        """Eliminate duplicate boxes on the same physical vehicle."""
        n = len(xyxy)
        order = sorted(
            range(n),
            key=lambda i: (
                1 if (COCO_VEHICLES.get(int(cls[i])) in ("Bus", "Truck") and float(conf[i]) >= 0.30) else 0,
                float(conf[i]),
            ),
            reverse=True,
        )
        kept: list[tuple[np.ndarray, int, int, float]] = []

        for idx in order:
            cid = int(cls[idx])
            label = COCO_VEHICLES.get(cid)
            if not label:
                continue
            box = xyxy[idx]
            bx1, by1, bx2, by2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            b_area = max(1.0, (bx2 - bx1) * (by2 - by1))

            duplicate = False
            for kbox, _, kcid, _ in kept:
                ix1 = bx1 if bx1 > kbox[0] else float(kbox[0])
                iy1 = by1 if by1 > kbox[1] else float(kbox[1])
                ix2 = bx2 if bx2 < kbox[2] else float(kbox[2])
                iy2 = by2 if by2 < kbox[3] else float(kbox[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter = (ix2 - ix1) * (iy2 - iy1)
                k_area = max(1.0, float((kbox[2] - kbox[0]) * (kbox[3] - kbox[1])))
                iou = inter / max(1.0, b_area + k_area - inter)
                containment = inter / (b_area if b_area < k_area else k_area)
                klabel = COCO_VEHICLES.get(kcid)

                if iou > 0.40 or (klabel in ("Bus", "Truck") and label == "Car" and containment > 0.55):
                    duplicate = True
                    break

            if not duplicate:
                kept.append((box, int(ids[idx]), cid, float(conf[idx])))
        return kept

    def _detect_high_conf_ambulance(self, frame: np.ndarray, orig_w: int, orig_h: int) -> dict[str, Any] | None:
        """
        Detect ambulance inside this lane's ROI strictly at high confidence (>= AMBULANCE_CONFIDENCE = 0.65).
        """
        amb_model = _get_ambulance_model()
        if amb_model is None or not _AMBULANCE_CLASS_IDS:
            return None
        try:
            with _AMBULANCE_LOCK:
                results = amb_model.predict(
                    source=frame,
                    conf=AMBULANCE_CONFIDENCE,
                    imgsz=AMBULANCE_IMAGE_SIZE,
                    iou=0.45,
                    quantize=QUANTIZE_MODE,
                    verbose=False,
                )

            best_detection: dict[str, Any] | None = None
            highest_confidence = 0.0

            boxes = results[0].boxes if results else None
            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy().astype(int)
                cls_arr = boxes.cls.cpu().numpy().astype(int)
                conf_arr = boxes.conf.cpu().numpy()

                for abox, cid, confidence in zip(xyxy_arr, cls_arr, conf_arr):
                    cid_int = int(cid)
                    conf_val = float(confidence)
                    if cid_int not in _AMBULANCE_CLASS_IDS or conf_val < AMBULANCE_CONFIDENCE:
                        continue
                    ax1, ay1, ax2, ay2 = int(abox[0]), int(abox[1]), int(abox[2]), int(abox[3])
                    if not self._box_in_roi(ax1, ay1, ax2, ay2, orig_w, orig_h):
                        continue
                    if conf_val > highest_confidence:
                        highest_confidence = conf_val
                        best_detection = {
                            "bbox": (ax1, ay1, ax2, ay2),
                            "confidence": conf_val,
                            "class_id": cid_int,
                        }
            return best_detection
        except Exception as exc:
            LOG.warning("Ambulance check failed on %s: %s", self.lane_id, exc)
            return None

    def _update_ambulance_state(
        self,
        amb_det: dict[str, Any] | None,
        frame_no: int,
        orig_w: int,
        orig_h: int,
        amb_clear_frames: int,
        stride: int,
    ) -> None:
        """
        Confirm ambulance ONLY when high-confidence (>= 0.65) detection is consistent across
        AMBULANCE_STREAK_REQUIRED consecutive checks on the same vehicle.
        """
        max_jump_px = max(120.0, math.hypot(orig_w, orig_h) * 0.22)

        with self.lock:
            if amb_det is not None:
                ax1, ay1, ax2, ay2 = amb_det["bbox"]
                acx, acy = (ax1 + ax2) >> 1, (ay1 + ay2) >> 1

                if self._ambulance_prev_center is not None:
                    jump = math.hypot(acx - self._ambulance_prev_center[0], acy - self._ambulance_prev_center[1])
                    if jump > max_jump_px and not self._ambulance_present:
                        self._ambulance_streak = 0
                        self._ambulance_recent_confs.clear()

                self._ambulance_prev_center = (acx, acy)
                self._ambulance_detected_now = True
                self._ambulance_streak += 1
                self._ambulance_recent_confs.append(amb_det["confidence"])
                self._ambulance_missing_frames = 0
                self._last_ambulance_bbox = amb_det["bbox"]
                self._last_ambulance_conf = amb_det["confidence"]
                self._last_ambulance_frame = frame_no

                if self._ambulance_streak >= AMBULANCE_STREAK_REQUIRED:
                    self._ambulance_present = True
                    if not self._ambulance_event_active:
                        self._ambulance_events += 1
                        self._ambulance_event_active = True
            else:
                self._ambulance_detected_now = False
                self._ambulance_streak = 0
                self._ambulance_recent_confs.clear()
                if not self._ambulance_present:
                    self._ambulance_prev_center = None
                    self._last_ambulance_bbox = None
                    self._last_ambulance_conf = 0.0
                else:
                    self._ambulance_missing_frames += stride
                    if self._ambulance_missing_frames >= amb_clear_frames:
                        self._ambulance_present = False
                        self._ambulance_event_active = False
                        self._ambulance_missing_frames = 0
                        self._ambulance_prev_center = None
                        self._last_ambulance_bbox = None
                        self._last_ambulance_conf = 0.0

    @staticmethod
    def _draw_ambulance_box(
        frame: np.ndarray, bbox: tuple[int, int, int, int], confidence: float, detected_now: bool = True
    ) -> None:
        """Draw high-visibility red ambulance bounding box & AMBULANCE PRIORITY banner."""
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), AMBULANCE_RED, 4)

        label = f"AMBULANCE PRIORITY {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.72
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(0, y1 - text_height - 15)

        cv2.rectangle(
            frame,
            (x1, label_y),
            (min(width - 1, x1 + text_width + 12), label_y + text_height + baseline + 10),
            AMBULANCE_RED,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 6, label_y + text_height + 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_overlays(
        self,
        frame: np.ndarray,
        frame_no: int,
        total_frames: int,
        active: list,
        occ_score: int,
        show_amb_priority: bool,
        amb_bbox: tuple[int, int, int, int] | None = None,
        amb_conf: float = 0.0,
        amb_now: bool = False,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        annotated = frame.copy()

        polygon = self._get_roi_polygon(w, h)
        if polygon is not None:
            cv2.polylines(annotated, [polygon], True, (255, 180, 0), 2)

        for _, x1, y1, x2, y2, label, score in active:
            box_color = VEHICLE_COLORS.get(label, (0, 220, 70))
            pts = OCCUPANCY_WEIGHTS.get(label, 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                annotated,
                f"{label} (+{pts}) {score:.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
            )

        if amb_bbox is not None and amb_conf >= AMBULANCE_CONFIDENCE:
            self._draw_ambulance_box(annotated, amb_bbox, amb_conf, detected_now=amb_now)

        if show_amb_priority or (amb_bbox is not None and amb_conf >= AMBULANCE_CONFIDENCE):
            cv2.putText(
                annotated,
                "AMBULANCE PRIORITY",
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                AMBULANCE_RED,
                2,
            )

        cv2.putText(
            annotated,
            f"{self.lane_id.upper()} | Score: {occ_score} | Frame: {frame_no}/{total_frames or '?'}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        return annotated

    def _run(self) -> None:
        cap = None
        try:
            if not self.video_path.is_file():
                raise FileNotFoundError(f"Video not found: {self.video_path}")
            model = YOLO(str(self.model_path))
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV cannot open {self.video_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1.0 or fps > 120.0:
                fps = 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_duration = (total_frames / fps) if total_frames > 0 else 0.0
            frame_delay = 1.0 / fps

            with self.lock:
                self.video_fps = fps
                self.total_frames = total_frames
                self.video_duration = video_duration
                self.status.update(
                    running=True,
                    error=None,
                    total_frames=total_frames,
                    video_fps=round(fps, 2),
                    video_duration=round(video_duration, 2),
                )
                if self._signal == "YELLOW" and self._phase_frames_remaining > 0:
                    self._phase_frames_remaining = max(1, int(round(YELLOW_DURATION * self.video_fps)))
                    self._sync_timer_fields()

            amb_clear_frames = max(1, int(round(AMBULANCE_CLEAR_SECONDS * fps)))
            amb_persist_frames = max(1, int(round(AMBULANCE_BOX_PERSIST_SEC * fps)))
            stale_frame_limit = max(15, int(fps * 1.5))

            # Initial Frame 1 Priority Scan (checks Ambulance + initial traffic density inside ROI before scheduling)
            frame_no = 0
            first_frame = None
            ok_first, first_frame = cap.read()
            if ok_first:
                frame_no = 1
                fh, fw = first_frame.shape[:2]

                # 1. Check ambulance on initial frame
                first_amb = self._detect_high_conf_ambulance(first_frame, fw, fh)

                # 2. Check initial vehicle density inside ROI on initial frame
                init_res = model.predict(
                    source=first_frame,
                    conf=CONFIDENCE,
                    iou=0.45,
                    imgsz=YOLO_INPUT_SIZE,
                    classes=COCO_VEHICLE_IDS,
                    quantize=QUANTIZE_MODE,
                    verbose=False,
                )
                init_score = 0
                init_boxes = init_res[0].boxes if init_res else None
                if init_boxes is not None and len(init_boxes) > 0:
                    ixyxy = init_boxes.xyxy.cpu().numpy().astype(int)
                    icls = init_boxes.cls.cpu().numpy().astype(int)
                    iconf = init_boxes.conf.cpu().numpy()
                    iids = np.arange(1, len(ixyxy) + 1, dtype=int)
                    for box, _, class_id, _ in self._suppress_overlapping_boxes(ixyxy, iids, icls, iconf):
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        if not self._box_in_roi(x1, y1, x2, y2, fw, fh):
                            continue
                        lbl = COCO_VEHICLES.get(class_id)
                        if lbl:
                            init_score += OCCUPANCY_WEIGHTS.get(lbl, 1)

                init_level, _ = self._score_to_density(init_score)

                with self.lock:
                    self.current_frame_no = 1
                    self._initial_occupancy_score = init_score
                    self._initial_density_level = init_level
                    self._initial_ambulance_detected = first_amb is not None
                    self._current_occupancy_score = init_score
                    self._yellow_max_score = init_score
                    if first_amb is not None:
                        self._ambulance_present = True
                        self._ambulance_streak = AMBULANCE_STREAK_REQUIRED
                        self._ambulance_events = 1
                        self._ambulance_event_active = True
                        self._last_ambulance_bbox = first_amb["bbox"]
                        self._last_ambulance_conf = first_amb["confidence"]
                        self._last_ambulance_frame = 1
                        self._ambulance_detected_now = True

                    self.status.update(
                        frame=1,
                        occupancy_score=init_score,
                        yellow_max_score=init_score,
                        density="AMBULANCE PRIORITY" if first_amb is not None else init_level,
                        ambulance=first_amb is not None,
                        ambulance_confidence=round(first_amb["confidence"], 2) if first_amb else 0.0,
                        ambulance_count=1 if first_amb is not None else 0,
                    )

                preview = self._draw_overlays(
                    first_frame,
                    1,
                    total_frames,
                    [],
                    init_score,
                    first_amb is not None,
                    amb_bbox=first_amb["bbox"] if first_amb else None,
                    amb_conf=first_amb["confidence"] if first_amb else 0.0,
                    amb_now=first_amb is not None,
                )
                ok_enc, enc_first = cv2.imencode(".jpg", preview, ENCODE_PARAMS)
                if ok_enc:
                    with self.lock:
                        self.jpeg = enc_first.tobytes()

            self._initial_scan_done.set()

            red_bg_tick = 0

            while not self.stop_event.is_set():
                loop_start = time.monotonic()
                with self.lock:
                    current_signal = self._signal
                    is_paused = self._paused_after_green
                    is_done = self._turn_completed

                if is_done:
                    self.stop_event.wait(0.05)
                    continue

                if is_paused or current_signal == "RED":
                    red_bg_tick += 1
                    if first_frame is not None and not self._turn_completed and red_bg_tick % 12 == 0:
                        fh, fw = first_frame.shape[:2]
                        bg_amb = self._detect_high_conf_ambulance(first_frame, fw, fh)
                        self._update_ambulance_state(bg_amb, frame_no, fw, fh, amb_clear_frames, 1)
                        with self.lock:
                            self.status.update(
                                ambulance=self._ambulance_present,
                                ambulance_confidence=round(self._last_ambulance_conf if self._ambulance_present else 0.0, 2),
                                ambulance_count=self._ambulance_events,
                            )
                            if self._ambulance_present:
                                self.status["density"] = "AMBULANCE PRIORITY"
                    self.stop_event.wait(0.06)
                    continue

                ok, frame = cap.read()
                if not ok:
                    self.pause_after_green()
                    continue

                frame_no += 1
                with self.lock:
                    self.current_frame_no = frame_no

                orig_h, orig_w = frame.shape[:2]

                # High-confidence ambulance detection inside ROI
                if frame_no % AMBULANCE_INFER_EVERY_N_FRAMES == 0:
                    amb_det = self._detect_high_conf_ambulance(frame, orig_w, orig_h)
                    self._update_ambulance_state(
                        amb_det, frame_no, orig_w, orig_h, amb_clear_frames, AMBULANCE_INFER_EVERY_N_FRAMES
                    )

                # Vehicle detection inside strict ROI
                infer_stride = 1 if current_signal == "YELLOW" else INFERENCE_EVERY_N_FRAMES
                if frame_no - self._last_inference_frame >= infer_stride:
                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=CONFIDENCE,
                        iou=0.45,
                        imgsz=YOLO_INPUT_SIZE,
                        classes=COCO_VEHICLE_IDS,
                        quantize=QUANTIZE_MODE,
                        verbose=False,
                    )

                    detections = []
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy().astype(int)
                        cls = boxes.cls.cpu().numpy().astype(int)
                        conf = boxes.conf.cpu().numpy()
                        ids = (
                            boxes.id.cpu().numpy().astype(int)
                            if boxes.id is not None
                            else np.arange(1, len(xyxy) + 1, dtype=int)
                        )

                        filtered = self._suppress_overlapping_boxes(xyxy, ids, cls, conf)

                        frame_occupancy_score = 0
                        vehicles_in_this_frame = 0

                        for box, tid, class_id, score in filtered:
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                            if not self._box_in_roi(x1, y1, x2, y2, orig_w, orig_h):
                                continue
                            raw_label = COCO_VEHICLES.get(class_id)
                            if not raw_label:
                                continue

                            cx, cy = (x1 + x2) >> 1, y2
                            fixed_label = self._resolve_vehicle_class(tid, raw_label, score)
                            weight = OCCUPANCY_WEIGHTS.get(fixed_label, 1)

                            frame_occupancy_score += weight
                            vehicles_in_this_frame += 1

                            self.track_hits[tid] += 1
                            self.last_seen[tid] = frame_no
                            self.current_tracks[tid] = (cx, cy, fixed_label)

                            if (
                                self.track_hits[tid] >= MIN_HITS_TO_COUNT
                                and self._crossed_line(tid, cx, cy, orig_w, orig_h)
                                and tid not in self.crossed_ids
                            ):
                                self.crossed_ids.add(tid)
                                self.counted_ids.add(tid)
                                self.counted_class_by_id[tid] = fixed_label

                            detections.append((tid, x1, y1, x2, y2, fixed_label, score))

                        with self.lock:
                            self._current_occupancy_score = frame_occupancy_score
                            if current_signal == "YELLOW":
                                if frame_occupancy_score > self._yellow_max_score:
                                    self._yellow_max_score = frame_occupancy_score
                                if vehicles_in_this_frame > self._yellow_max_vehicles:
                                    self._yellow_max_vehicles = vehicles_in_this_frame

                    self._last_boxes = detections
                    self._last_inference_frame = frame_no

                # Video-frame-synced timer decrement
                with self.lock:
                    if self._signal in ("YELLOW", "GREEN") and not self._ambulance_priority_active:
                        if self._phase_frames_remaining > 0:
                            self._phase_frames_remaining -= 1
                        if self._phase_frames_remaining <= 0:
                            self._phase_frames_remaining = 0
                            self._phase_expired = True
                        self._sync_timer_fields()

                # Bounded memory cleanup for stale tracks
                if frame_no % 15 == 0 and self.last_seen:
                    stale = [tid for tid, last in self.last_seen.items() if frame_no - last > stale_frame_limit]
                    for tid in stale:
                        self.last_seen.pop(tid, None)
                        self.current_tracks.pop(tid, None)
                        self.previous_centers.pop(tid, None)
                        self.track_hits.pop(tid, None)
                        if tid not in self.crossed_ids:
                            self.locked_track_class.pop(tid, None)
                            self.locked_track_conf.pop(tid, None)

                cumulative_class_counts = Counter(self.counted_class_by_id.values())

                with self.lock:
                    active = self._last_boxes
                    amb_flag = self._ambulance_present
                    amb_prio = self._ambulance_priority_active
                    occ_score = self._current_occupancy_score

                    show_amb_box = (
                        amb_flag
                        and self._last_ambulance_bbox is not None
                        and (frame_no - self._last_ambulance_frame) <= amb_persist_frames
                    )
                    amb_bbox = self._last_ambulance_bbox if show_amb_box else None
                    amb_conf = self._last_ambulance_conf if show_amb_box else 0.0
                    amb_now = self._ambulance_detected_now

                    if amb_prio or amb_flag:
                        display_density = "AMBULANCE PRIORITY"
                    elif self._signal == "YELLOW":
                        display_density = "ANALYSING"
                    else:
                        display_density = self._locked_density or "PENDING"

                annotated = self._draw_overlays(
                    frame,
                    frame_no,
                    total_frames,
                    active,
                    occ_score,
                    amb_prio or amb_flag,
                    amb_bbox=amb_bbox,
                    amb_conf=amb_conf,
                    amb_now=amb_now,
                )
                total_crossed = len(self.crossed_ids)

                ok, encoded = cv2.imencode(".jpg", annotated, ENCODE_PARAMS)
                if ok:
                    now_t = time.monotonic()
                    inst_fps = round(1.0 / max(0.001, now_t - self._last_frame_time), 1)
                    self._last_frame_time = now_t
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                        self.status.update(
                            running=True,
                            frame=frame_no,
                            total_frames=total_frames,
                            unique_total=total_crossed,
                            crossed=total_crossed,
                            vehicles_passed=total_crossed,
                            waiting=len(active),
                            occupancy_score=occ_score,
                            yellow_max_score=self._yellow_max_score,
                            yellow_max_vehicles=self._yellow_max_vehicles,
                            density=display_density,
                            locked_density=self._locked_density,
                            ambulance=amb_flag,
                            ambulance_confidence=round(amb_conf, 2),
                            ambulance_priority=amb_prio or amb_flag,
                            ambulance_count=self._ambulance_events,
                            classes={name: int(cumulative_class_counts.get(name, 0)) for name in VEHICLE_CLASS_NAMES},
                            fps=inst_fps,
                            roi_calibrated=bool(self.config.get("roi")),
                        )

                elapsed = time.monotonic() - loop_start
                remaining_delay = frame_delay - elapsed
                if remaining_delay > 0.0:
                    self.stop_event.wait(remaining_delay)

        except Exception as exc:
            LOG.exception("Worker %s failed", self.lane_id)
            with self.lock:
                self.status.update(running=False, error=str(exc))
        finally:
            self._initial_scan_done.set()
            if cap is not None:
                cap.release()
            with self.lock:
                self.status["running"] = False


class FourLaneSystem:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = load_config()
        self.custom_video_sources: dict[str, str] = {}
        self.workers: dict[str, LaneWorker] = {}
        self.running = False
        self.all_completed = False
        self.controller_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.active_lane: str | None = None
        self.upcoming_yellow_lane: str | None = None
        self.signal_phase: str = "RED"
        self.accident_blocked: set[str] = set()
        self._lane_order = list(LANES)
        self._lane_idx = 0

        self._ambulance_lane: str | None = None
        self._preempted_lane: str | None = None
        self._preempted_phase: str | None = None
        self._preempted_remaining_frames: int = 0
        self._preempted_upcoming_lane: str | None = None
        self._preempted_upcoming_frames: int = 0

    def set_video_sources(self, sources: dict[str, str]) -> None:
        """Set custom uploaded video paths for lane1..lane4 while keeping rois.json calibration."""
        with self.lock:
            self.custom_video_sources = {
                lid: str(sources[lid])
                for lid in LANES
                if lid in sources and sources[lid]
            }

    def _select_next_priority_lane(self, exclude: set[str] | None = None) -> str | None:
        """
        Select the highest-priority pending lane among the 4 lanes:
          1. Ambulance lane (if any lane has ambulance)
          2. HIGH traffic density lane
          3. MEDIUM traffic density lane
          4. LOW traffic density lane
        """
        excluded = exclude or set()
        candidates: list[tuple[tuple[int, int], int, str]] = []
        for idx, lid in enumerate(self._lane_order):
            if lid in excluded:
                continue
            w = self.workers.get(lid)
            if w and not w._turn_completed:
                candidates.append((w.get_priority_key(), -idx, lid))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        chosen_lid = candidates[0][2]
        self._lane_idx = self._lane_order.index(chosen_lid)
        return chosen_lid

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.running:
                return self.snapshot()

            model_path = _resolve_main_model_path()
            _get_ambulance_model()

            for old_w in self.workers.values():
                old_w.stop()

            self.config = load_config()
            self.stop_event.clear()
            self.workers.clear()
            self.all_completed = False
            self.upcoming_yellow_lane = None
            self._ambulance_lane = None
            self._preempted_lane = None
            self._preempted_phase = None
            self._preempted_remaining_frames = 0
            self._preempted_upcoming_lane = None
            self._preempted_upcoming_frames = 0

            for lane_id in self._lane_order:
                override_path = self.custom_video_sources.get(lane_id)
                worker = LaneWorker(
                    lane_id,
                    self.config[lane_id],
                    model_path,
                    video_override=override_path,
                )
                self.workers[lane_id] = worker
                worker.start()

            for worker in self.workers.values():
                worker._initial_scan_done.wait(timeout=3.0)

            self.running = True
            first_lane = self._select_next_priority_lane() or self._lane_order[0]
            self._enter_yellow(first_lane)

            self.controller_thread = threading.Thread(
                target=self._control_loop, name="four-lane-controller", daemon=True
            )
            self.controller_thread.start()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self.running = False
            self.stop_event.set()
            workers = list(self.workers.values())

        for w in workers:
            w.stop()

        if self.controller_thread and self.controller_thread.is_alive():
            self.controller_thread.join(timeout=3)

        with self.lock:
            self.active_lane = None
            self.upcoming_yellow_lane = None
            self.signal_phase = "RED"
            return self.snapshot()

    def _enter_yellow(self, lane_id: str, resume_frames: int | None = None) -> None:
        self.active_lane = lane_id
        self.upcoming_yellow_lane = None
        self.signal_phase = "YELLOW"
        for lid, w in self.workers.items():
            if lid == lane_id:
                w.begin_yellow_phase(resume_frames=resume_frames)
            elif not w._turn_completed and not w._paused_after_green:
                w.set_red_paused()

    def _enter_green(self, lane_id: str) -> None:
        w = self.workers[lane_id]
        w.decide_and_begin_green()
        self.active_lane = lane_id
        self.upcoming_yellow_lane = None
        self.signal_phase = "GREEN"
        for lid, other in self.workers.items():
            if lid != lane_id and not other._turn_completed and not other._paused_after_green:
                other.set_red_paused()

    def _peek_next_priority_lane(self) -> str | None:
        exclude = {self.active_lane} if self.active_lane else set()
        return self._select_next_priority_lane(exclude=exclude)

    def _complete_all(self) -> None:
        self.running = False
        self.all_completed = True
        self.active_lane = None
        self.upcoming_yellow_lane = None
        self._ambulance_lane = None
        self.signal_phase = "COMPLETED"
        self.stop_event.set()
        for w in self.workers.values():
            w.pause_after_green()
            w.stop_event.set()

    def _control_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                if not self.running:
                    break

                if all(w._turn_completed for w in self.workers.values()):
                    self._complete_all()
                    break

                # 1st Priority: Ambulance override check across all 4 lanes
                if self.signal_phase != "AMBULANCE_PRIORITY":
                    amb_lane = None
                    for lid in self._lane_order:
                        if not self.workers[lid]._turn_completed and self.workers[lid]._ambulance_present:
                            amb_lane = lid
                            break

                    if amb_lane is not None:
                        self._preempted_lane = self.active_lane
                        self._preempted_phase = self.signal_phase
                        if (
                            self._preempted_lane
                            and self._preempted_lane in self.workers
                            and self._preempted_lane != amb_lane
                        ):
                            pw = self.workers[self._preempted_lane]
                            with pw.lock:
                                self._preempted_remaining_frames = max(1, pw._phase_frames_remaining)
                            pw.pause_temporarily()

                        if self.upcoming_yellow_lane and self.upcoming_yellow_lane in self.workers:
                            uw = self.workers[self.upcoming_yellow_lane]
                            with uw.lock:
                                self._preempted_upcoming_lane = self.upcoming_yellow_lane
                                self._preempted_upcoming_frames = max(1, uw._phase_frames_remaining)
                            if self.upcoming_yellow_lane != amb_lane:
                                uw.set_red_paused()
                        else:
                            self._preempted_upcoming_lane = None
                            self._preempted_upcoming_frames = 0

                        self.upcoming_yellow_lane = None
                        self._ambulance_lane = amb_lane
                        self.active_lane = amb_lane
                        self.signal_phase = "AMBULANCE_PRIORITY"

                        for lid, w in self.workers.items():
                            if lid == amb_lane:
                                w.set_ambulance_priority_green()
                            elif not w._turn_completed and not w._paused_after_green:
                                w.set_red_paused()

                        self.stop_event.wait(0.05)
                        continue

                if self.signal_phase == "AMBULANCE_PRIORITY":
                    amb_w = self.workers.get(self._ambulance_lane or "")
                    if amb_w is None or not amb_w._ambulance_present or amb_w._turn_completed:
                        if amb_w and not amb_w._turn_completed:
                            amb_w.set_red_paused()

                        resume_lane = self._preempted_lane
                        resume_phase = self._preempted_phase
                        resume_frames = self._preempted_remaining_frames
                        up_lane = self._preempted_upcoming_lane
                        up_frames = self._preempted_upcoming_frames

                        self._ambulance_lane = None
                        self._preempted_lane = None
                        self._preempted_phase = None
                        self._preempted_remaining_frames = 0
                        self._preempted_upcoming_lane = None
                        self._preempted_upcoming_frames = 0

                        if resume_lane and not self.workers[resume_lane]._turn_completed:
                            if resume_phase == "YELLOW":
                                self._enter_yellow(resume_lane, resume_frames=resume_frames)
                            else:
                                self.active_lane = resume_lane
                                self.signal_phase = "GREEN"
                                self.workers[resume_lane].resume_green_phase(resume_frames)
                                if up_lane and not self.workers[up_lane]._turn_completed:
                                    self.upcoming_yellow_lane = up_lane
                                    self.workers[up_lane].begin_yellow_phase(resume_frames=up_frames)
                        else:
                            nxt = self._select_next_priority_lane()
                            if nxt is None:
                                self._complete_all()
                                break
                            self._enter_yellow(nxt)

                    self.stop_event.wait(0.05)
                    continue

                if self.active_lane and self.active_lane in self.workers:
                    cur_w = self.workers[self.active_lane]
                    with cur_w.lock:
                        done = cur_w._turn_completed
                        expired = cur_w._phase_expired
                        rem_frames = cur_w._phase_frames_remaining
                        cur_fps = cur_w.video_fps if cur_w.video_fps > 0.0 else 25.0

                    if self.signal_phase == "GREEN" and not done and self.upcoming_yellow_lane is None:
                        lookahead_frames = max(1, int(round(YELLOW_DURATION * cur_fps)))
                        if 0 < rem_frames <= lookahead_frames:
                            next_lid = self._peek_next_priority_lane()
                            if next_lid is not None and next_lid != self.active_lane:
                                self.upcoming_yellow_lane = next_lid
                                self.workers[next_lid].begin_yellow_phase()

                    if done or expired:
                        if self.signal_phase == "YELLOW" and not done:
                            self._enter_green(self.active_lane)
                        else:
                            cur_w.pause_after_green()

                            if self.upcoming_yellow_lane and not self.workers[self.upcoming_yellow_lane]._turn_completed:
                                next_lid = self.upcoming_yellow_lane
                                self._lane_idx = self._lane_order.index(next_lid)
                                self.upcoming_yellow_lane = None
                                self._enter_green(next_lid)
                            else:
                                self.upcoming_yellow_lane = None
                                nxt = self._select_next_priority_lane()
                                if nxt is None:
                                    self._complete_all()
                                    break
                                self._enter_yellow(nxt)

            self.stop_event.wait(0.04)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            lanes_snapshot = {}
            total_unique = 0
            for lid in self._lane_order:
                worker = self.workers.get(lid)
                if worker:
                    snap = worker.snapshot()
                else:
                    snap = {
                        "lane": lid,
                        "running": False,
                        "completed": False,
                        "video_eof": False,
                        "paused": True,
                        "unique_total": 0,
                        "crossed": 0,
                        "vehicles_passed": 0,
                        "waiting": 0,
                        "occupancy_score": 0,
                        "yellow_max_score": 0,
                        "yellow_max_vehicles": 0,
                        "decided_green_seconds": 0,
                        "density": "PENDING",
                        "ambulance": False,
                        "ambulance_confidence": 0.0,
                        "ambulance_priority": False,
                        "ambulance_count": 0,
                        "signal": "RED",
                        "timer": 0,
                        "remaining_seconds": 0,
                        "remaining_frames": 0,
                        "classes": {name: 0 for name in VEHICLE_CLASS_NAMES},
                    }
                lanes_snapshot[lid] = snap
                total_unique += int(snap.get("vehicles_passed", 0))

            return {
                "running": self.running,
                "all_completed": self.all_completed,
                "active_lane": self.active_lane,
                "upcoming_yellow_lane": self.upcoming_yellow_lane,
                "signal_phase": self.signal_phase,
                "ambulance_priority": self.signal_phase == "AMBULANCE_PRIORITY",
                "ambulance_lane": self._ambulance_lane,
                "total_unique_vehicles": total_unique,
                "total_vehicles": total_unique,
                "lanes": lanes_snapshot,
            }

    def get_frame(self, lane_id: str) -> bytes | None:
        with self.lock:
            worker = self.workers.get(lane_id)
        if not worker:
            return None
        return worker.get_jpeg()


four_lane_system = FourLaneSystem()
system = four_lane_system