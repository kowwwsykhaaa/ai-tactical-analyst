"""
Central config: paths, model choices, pitch dimensions.
Edit these before running the pipeline.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Paths ---
VIDEO_PATH = str(ROOT_DIR / "data" / "match_clip.mp4")
OUTPUT_DIR = str(ROOT_DIR / "outputs")
TRACKS_CSV = str(ROOT_DIR / "outputs" / "tracks.csv")
PITCH_TRACKS_CSV = str(ROOT_DIR / "outputs" / "pitch_tracks.csv")

# --- Detection model ---
YOLO_MODEL_PATH = str(ROOT_DIR / "models" / "football-player-detection.pt")

# Class IDs from the detection model (confirmed from Roboflow
# "football-players-detection-3zvbc" v20 data.yaml)
CLASS_BALL = 0
CLASS_GOALKEEPER = 1
CLASS_PLAYER = 2
CLASS_REFEREE = 3

# --- Pitch geometry (standard FIFA pitch, meters) ---
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

PITCH_REFERENCE_POINTS = [
    (0, 0),
    (PITCH_LENGTH, 0),
    (PITCH_LENGTH, PITCH_WIDTH),
    (0, PITCH_WIDTH),
]

# --- Team assignment ---
N_TEAMS = 2

# --- Pitch control model ---
PLAYER_MAX_SPEED_MPS = 8.0
GRID_RESOLUTION = 1.0

# --- Formation detection ---
FORMATION_WINDOW_FRAMES = 60
N_FORMATION_CLUSTERS = 2