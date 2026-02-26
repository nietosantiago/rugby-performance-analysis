# ──────────────────────────────────────────────
# Configuration — Rugby Video Analysis System
# ──────────────────────────────────────────────

import os

# ── General ───────────────────────────────────
PROCESS_SAMPLE_RATE_FPS = 2          # Frames to process per second of video

# ── Field Coordinates ─────────────────────────
FIELD_LENGTH = 100.0                 # Normalised x range
FIELD_WIDTH  = 100.0                 # Normalised y range (0-100)

# ── YOLO Detection ────────────────────────────
YOLO_MODEL_NAME        = "yolov8s.pt"
CONFIDENCE_THRESHOLD   = 0.50
MIN_BBOX_AREA          = 1000       # px², ignore noise/spectators

# ── Ball Detection ────────────────────────────
BALL_YOLO_CLASS            = 32     # "sports ball" in COCO
BALL_CONFIDENCE_THRESHOLD  = 0.30

# ── Kalman Filter (ball) ─────────────────────
KALMAN_MAX_PREDICTION_FRAMES = 15

# ── DeepSORT ──────────────────────────────────
DEEPSORT_MAX_AGE   = 30
DEEPSORT_NN_BUDGET = 100

# ── Team Identification ──────────────────────
TEAM_A_LABEL = "Team_A"
TEAM_B_LABEL = "Team_B"
TEAM_CALIBRATION_SAMPLES = 50       # HSV samples before calibration

# ── Event Detection ──────────────────────────
PROXIMITY_THRESHOLD_PX = 80         # px, players considered "close"
TACKLE_COOLDOWN        = 30         # frames
CARRY_METERS_THRESHOLD = 3.0        # field-metres
RUCK_CLUSTER_RADIUS_PX = 100        # px
RUCK_MIN_PLAYERS       = 3
KICK_SPEED_MULTIPLIER  = 3.0
KICK_COOLDOWN          = 60         # frames
LINEOUT_LATERAL_THRESHOLD = 10      # % of field width
LINEOUT_MIN_PLAYERS    = 4
LINEOUT_COOLDOWN       = 90         # frames
MIN_KICK_SPEED         = 25.0       # px/frame

# ── Performance ──────────────────────────────
FRAME_SKIP = None                   # None → auto from fps

# ── Output Paths ─────────────────────────────
OUTPUT_DIR    = "output"
PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR   = os.path.join("reports", "figures")
