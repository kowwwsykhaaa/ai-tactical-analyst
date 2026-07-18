"""
Live pipeline for the Streamlit Cloud demo: upload a short clip, click 4 pitch
points, and get detection + tracking + homography + team assignment + pitch
control, all computed on the fly (no local install needed by the visitor).

Deliberately capped in scope (short clips, frame skipping) to run within
Streamlit Community Cloud's free CPU-only resource limits. This is NOT the
same as the full local pipeline (detect_track.py etc.) -- it's a lighter,
self-contained version built for a public browser demo.
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans

MAX_SECONDS = 15   # cap how much of the uploaded clip gets processed
FRAME_SKIP = 2      # process every Nth frame to cut compute further
MODEL_PATH = "../models/football-player-detection.pt"

CLASS_BALL, CLASS_GOALKEEPER, CLASS_PLAYER, CLASS_REFEREE = 0, 1, 2, 3
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
PITCH_REFERENCE_POINTS = [
    (0, 0), (PITCH_LENGTH, 0), (PITCH_LENGTH, PITCH_WIDTH), (0, PITCH_WIDTH)
]


def get_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read the uploaded video.")
    return frame  # BGR, as read by OpenCV


def run_detection_and_tracking(video_path: str, progress_callback=None) -> pd.DataFrame:
    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    max_frames = int(fps * MAX_SECONDS)

    rows = []
    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % FRAME_SKIP == 0:
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            for box, track_id, class_id in zip(
                detections.xyxy, detections.tracker_id, detections.class_id
            ):
                if track_id is None:
                    continue
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, y2
                rows.append({
                    "frame": frame_idx, "track_id": int(track_id), "class_id": int(class_id),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy,
                })
            if progress_callback:
                progress_callback(min(frame_idx / max_frames, 1.0))
        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


def compute_homography(image_points: list) -> np.ndarray:
    pitch_points = np.array(PITCH_REFERENCE_POINTS, dtype=np.float32)
    H, _ = cv2.findHomography(np.array(image_points, dtype=np.float32), pitch_points)
    return H


def apply_homography(df: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    pts = df[["cx", "cy"]].to_numpy(dtype=np.float32).reshape(-1, 1, 2)
    pitch_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    df = df.copy()
    df["pitch_x"] = pitch_pts[:, 0]
    df["pitch_y"] = pitch_pts[:, 1]
    return df


def get_dominant_color(crop: np.ndarray) -> np.ndarray:
    h = crop.shape[0]
    torso = crop[: h // 2, :, :]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
    non_grass = cv2.inRange(hsv, lower_green, upper_green) == 0
    hs_pixels = hsv[non_grass][:, :2].astype(np.float32)
    if len(hs_pixels) < 10:
        hs_pixels = hsv.reshape(-1, 3)[:, :2].astype(np.float32)
    if len(hs_pixels) == 0:
        return np.array([0, 0])
    km = KMeans(n_clusters=1, n_init=3, random_state=0).fit(hs_pixels)
    return km.cluster_centers_[0]


def assign_teams(video_path: str, df: pd.DataFrame) -> pd.DataFrame:
    player_mask = df["class_id"].isin([CLASS_PLAYER, CLASS_GOALKEEPER])
    player_df = df[player_mask].copy()

    cap = cv2.VideoCapture(video_path)
    colors = []
    current_frame, frame, ok = -1, None, False

    for _, row in player_df.iterrows():
        if row["frame"] != current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            ok, frame = cap.read()
            current_frame = row["frame"]
        if not ok or frame is None:
            colors.append([0, 0])
            continue
        x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            colors.append([0, 0])
            continue
        colors.append(get_dominant_color(crop))
    cap.release()

    colors = np.array(colors)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(colors)
    player_df["team_id"] = km.labels_

    df = df.merge(
        player_df[["frame", "track_id", "team_id"]],
        on=["frame", "track_id"], how="left"
    )
    return df