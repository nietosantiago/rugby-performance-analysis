"""
detection.py — Player detection, ball detection, and team classification.

Uses YOLO for object detection and HSV colour clustering for team assignment.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models import Detection


# ───────────────────────────────────────────────
# Player Detector
# ───────────────────────────────────────────────

class PlayerDetector:
    """Detects people in a video frame using YOLO."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_NAME,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        min_area: int = config.MIN_BBOX_AREA,
    ):
        self.model = YOLO(model_path)
        self.conf = confidence
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(frame, classes=[0], conf=self.conf, verbose=False)
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if not hasattr(boxes, "xyxy") or len(boxes.xyxy) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                area = (x2 - x1) * (y2 - y1)
                if area >= self.min_area:
                    detections.append(
                        Detection(
                            bbox=[x1, y1, x2, y2],
                            confidence=conf,
                            class_name="person",
                        )
                    )
        return detections


# ───────────────────────────────────────────────
# Ball Detector
# ───────────────────────────────────────────────

class BallDetector:
    """Detects a rugby ball using YOLO (COCO class 32 — sports ball)."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_NAME,
        confidence: float = config.BALL_CONFIDENCE_THRESHOLD,
    ):
        self.model = YOLO(model_path)
        self.conf = confidence

    def detect(self, frame: np.ndarray):
        results = self.model(
            frame,
            classes=[config.BALL_YOLO_CLASS],
            conf=self.conf,
            verbose=False,
        )
        best: Detection | None = None
        best_conf = 0.0

        for result in results:
            boxes = result.boxes
            if not hasattr(boxes, "xyxy") or len(boxes.xyxy) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                if conf > best_conf:
                    best_conf = conf
                    best = Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=conf,
                        class_name="ball",
                    )
        return best


# ───────────────────────────────────────────────
# Team Classifier (HSV + KMeans)
# ───────────────────────────────────────────────

class TeamClassifier:
    """Classifies detected players into Team A / Team B using jersey colour in HSV space."""

    def __init__(
        self,
        team_a_label: str = config.TEAM_A_LABEL,
        team_b_label: str = config.TEAM_B_LABEL,
        calibration_samples: int = config.TEAM_CALIBRATION_SAMPLES,
    ):
        self.team_a = team_a_label
        self.team_b = team_b_label
        self.calibration_samples = calibration_samples

        self.calibrated = False
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.color_samples: list[np.ndarray] = []
        self.hsv_centers: np.ndarray | None = None

    # ── helpers ────────────────────────────────

    @staticmethod
    def _crop_jersey(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        """Crop the upper-torso region of a bounding box (10–50 % height, 30–70 % width)."""
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        cy1 = max(0, y1 + int(h * 0.10))
        cy2 = min(frame.shape[0], y1 + int(h * 0.50))
        cx1 = max(0, x1 + int(w * 0.30))
        cx2 = min(frame.shape[1], x1 + int(w * 0.70))
        if cy2 <= cy1 or cx2 <= cx1:
            return None
        return frame[cy1:cy2, cx1:cx2]

    def extract_jersey_hsv(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        """Return the mean HSV colour of the jersey, filtering out pitch-green pixels."""
        crop = self._crop_jersey(frame, bbox)
        if crop is None or crop.size == 0:
            return np.array([0.0, 0.0, 0.0])

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Mask green pixels (grass)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        green_mask = (h_channel >= 35) & (h_channel <= 85) & (s_channel > 40)
        jersey_mask = ~green_mask

        jersey_pixels = hsv[jersey_mask]
        if len(jersey_pixels) < 5:
            # Not enough non-green pixels → fallback to full crop mean
            return np.mean(hsv.reshape(-1, 3), axis=0)
        return np.mean(jersey_pixels, axis=0)

    # ── calibration ────────────────────────────

    def add_sample(self, hsv_color: np.ndarray) -> None:
        """Accumulate an HSV sample for calibration."""
        if self.calibrated:
            return
        if np.all(hsv_color == 0):
            return
        self.color_samples.append(hsv_color)
        if len(self.color_samples) >= self.calibration_samples:
            self._calibrate()

    def _calibrate(self) -> None:
        samples = np.array(self.color_samples)
        self.kmeans.fit(samples)
        self.hsv_centers = self.kmeans.cluster_centers_
        self.calibrated = True

    # ── classification ─────────────────────────

    def classify(self, hsv_color: np.ndarray) -> str:
        if not self.calibrated:
            return "Unknown"
        cluster = self.kmeans.predict([hsv_color])[0]
        return self.team_a if cluster == 0 else self.team_b
