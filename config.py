# Configuration parameters for the Rugby Video Analysis System

# General
PROCESS_SAMPLE_RATE_FPS = 2  # Number of frames to process per second of video

# Field Coordinates
# We map the video to a standard rugby field 100m x 70m
FIELD_WIDTH_METERS = 70.0
FIELD_LENGTH_METERS = 100.0

# YOLO Detection
YOLO_MODEL_NAME = "yolov8n.pt"  # Use nano model for performance
CONFIDENCE_THRESHOLD = 0.50
MIN_BBOX_AREA = 1000  # Minimum bounding box area in pixels to ignore noise/spectators

# Heuristics for Event Detection
PROXIMITY_THRESHOLD_PX = 80  # Distance in pixels to be considered "close" (for tackles, carries, etc.)
RUCK_MIN_PLAYERS = 4  # Minimum number of players clustered to identify a ruck
RUCK_PROXIMITY_PX = 60

# Team Identification
# We use color clustering (K-Means k=2) for upper body colors.
TEAM_A_LABEL = "Team_A"
TEAM_B_LABEL = "Team_B"

# Outputs
OUTPUT_DIR = "output"
