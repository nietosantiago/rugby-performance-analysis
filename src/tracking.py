"""
tracking.py — Player tracking (DeepSORT) and ball tracking (Kalman filter).

PlayerTracker wraps deep-sort-realtime to assign persistent IDs.
BallTracker uses an OpenCV Kalman filter to smooth / predict ball position.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _HAS_DEEPSORT = True
except ImportError:
    _HAS_DEEPSORT = False

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models import Detection, TrackedObject


# ───────────────────────────────────────────────
# Player Tracker (DeepSORT with centroid fallback)
# ───────────────────────────────────────────────

class PlayerTracker:
    """Multi-object tracker for players.

    Uses DeepSORT when available; otherwise falls back to a simple
    centroid-based tracker (scipy distance matrix).
    """

    VELOCITY_WINDOW = 5  # frames for velocity averaging

    def __init__(
        self,
        max_age: int = config.DEEPSORT_MAX_AGE,
        nn_budget: int = config.DEEPSORT_NN_BUDGET,
    ):
        self.use_deepsort = _HAS_DEEPSORT
        if self.use_deepsort:
            self.tracker = DeepSort(
                max_age=max_age,
                nn_budget=nn_budget,
                nms_max_overlap=0.7,
            )
        else:
            # Fallback: centroid tracker (import from legacy code)
            from collections import OrderedDict
            from scipy.spatial import distance as dist_module
            self._dist_module = dist_module
            self._next_id = 1
            self._objects: dict[int, np.ndarray] = OrderedDict()
            self._bboxes: dict[int, list[int]] = OrderedDict()
            self._disappeared: dict[int, int] = OrderedDict()
            self._max_disappeared = max_age
            self._max_distance = 100

        self._velocity_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.VELOCITY_WINDOW)
        )

    # ── public API ─────────────────────────────

    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[TrackedObject]:
        if self.use_deepsort:
            return self._update_deepsort(detections, frame)
        return self._update_centroid(detections, frame)

    # ── DeepSORT path ──────────────────────────

    def _update_deepsort(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[TrackedObject]:
        raw_dets = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            w, h = x2 - x1, y2 - y1
            raw_dets.append(([x1, y1, w, h], d.confidence, d.class_name))

        tracks = self.tracker.update_tracks(raw_dets, frame=frame)
        tracked: list[TrackedObject] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = int(track.track_id)
            ltwh = track.to_ltwh()
            x1 = int(ltwh[0])
            y1 = int(ltwh[1])
            x2 = int(ltwh[0] + ltwh[2])
            y2 = int(ltwh[1] + ltwh[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            self._velocity_history[tid].append((cx, cy))
            vel = self._compute_velocity(self._velocity_history[tid])
            speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)

            tracked.append(
                TrackedObject(
                    track_id=tid,
                    bbox=[x1, y1, x2, y2],
                    centroid=(cx, cy),
                    velocity=vel,
                    speed=speed,
                )
            )
        return tracked

    # ── Centroid fallback path ─────────────────

    def _update_centroid(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[TrackedObject]:
        from collections import OrderedDict

        if len(detections) == 0:
            for oid in list(self._disappeared.keys()):
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self._max_disappeared:
                    self._objects.pop(oid, None)
                    self._bboxes.pop(oid, None)
                    self._disappeared.pop(oid, None)
            return self._build_tracked_list()

        input_centroids = []
        input_bboxes = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            input_centroids.append(np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]))
            input_bboxes.append(d.bbox)

        if len(self._objects) == 0:
            for i, c in enumerate(input_centroids):
                self._register(c, input_bboxes[i])
        else:
            obj_ids = list(self._objects.keys())
            obj_cents = np.array(list(self._objects.values()))
            inp_cents = np.array(input_centroids)
            D = self._dist_module.cdist(obj_cents, inp_cents)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self._max_distance:
                    continue
                oid = obj_ids[row]
                self._objects[oid] = inp_cents[col]
                self._bboxes[oid] = input_bboxes[col]
                self._disappeared[oid] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])) - used_rows:
                oid = obj_ids[row]
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self._max_disappeared:
                    self._objects.pop(oid, None)
                    self._bboxes.pop(oid, None)
                    self._disappeared.pop(oid, None)

            for col in set(range(D.shape[1])) - used_cols:
                self._register(inp_cents[col], input_bboxes[col])

        return self._build_tracked_list()

    def _register(self, centroid: np.ndarray, bbox: list[int]) -> int:
        oid = self._next_id
        self._objects[oid] = centroid
        self._bboxes[oid] = bbox
        self._disappeared[oid] = 0
        self._next_id += 1
        return oid

    def _build_tracked_list(self) -> list[TrackedObject]:
        tracked: list[TrackedObject] = []
        for oid, cent in self._objects.items():
            cx, cy = float(cent[0]), float(cent[1])
            self._velocity_history[oid].append((cx, cy))
            vel = self._compute_velocity(self._velocity_history[oid])
            speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            bbox = self._bboxes.get(oid, [0, 0, 0, 0])
            tracked.append(
                TrackedObject(
                    track_id=oid,
                    bbox=bbox,
                    centroid=(cx, cy),
                    velocity=vel,
                    speed=speed,
                )
            )
        return tracked

    # ── shared helpers ─────────────────────────

    @staticmethod
    def _compute_velocity(history: deque) -> Tuple[float, float]:
        if len(history) < 2:
            return (0.0, 0.0)
        dx = history[-1][0] - history[0][0]
        dy = history[-1][1] - history[0][1]
        n = len(history) - 1
        return (dx / n, dy / n)


# ───────────────────────────────────────────────
# Ball Tracker (Kalman filter)
# ───────────────────────────────────────────────

class BallTracker:
    """Tracks the rugby ball using an OpenCV Kalman filter.

    When a detection is available the filter is *corrected*; when the
    ball is not detected the filter *predicts* for up to
    ``KALMAN_MAX_PREDICTION_FRAMES`` consecutive frames.
    """

    def __init__(
        self,
        max_prediction_frames: int = config.KALMAN_MAX_PREDICTION_FRAMES,
    ):
        self.max_pred = max_prediction_frames
        self.kf = cv2.KalmanFilter(4, 2)  # state (x,y,vx,vy), measurement (x,y)

        # Transition matrix
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32,
        )
        # Measurement matrix
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        self.initialized = False
        self.frames_without = 0
        self.last_position: Optional[Tuple[float, float]] = None
        self.ball_speed: float = 0.0

    def update(self, detection: Optional[Detection]) -> Optional[Tuple[float, float]]:
        if detection is not None:
            x1, y1, x2, y2 = detection.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if not self.initialized:
                self.kf.statePre = np.array(
                    [[cx], [cy], [0], [0]], dtype=np.float32
                )
                self.kf.statePost = np.array(
                    [[cx], [cy], [0], [0]], dtype=np.float32
                )
                self.initialized = True

            measurement = np.array([[cx], [cy]], dtype=np.float32)
            self.kf.correct(measurement)
            self.frames_without = 0
        else:
            self.frames_without += 1

        if self.initialized and self.frames_without < self.max_pred:
            predicted = self.kf.predict()
            pos = (float(predicted[0][0]), float(predicted[1][0]))

            if self.last_position is not None:
                dx = pos[0] - self.last_position[0]
                dy = pos[1] - self.last_position[1]
                self.ball_speed = math.sqrt(dx ** 2 + dy ** 2)

            self.last_position = pos
            return pos

        return None
