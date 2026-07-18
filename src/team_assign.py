"""
Step 3: Assign each tracked player to a team by clustering jersey colors
(K-Means on dominant color extracted from each player's bounding box crop).

Referees are excluded via class_id before clustering.
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import PITCH_TRACKS_CSV, CLASS_PLAYER, CLASS_GOALKEEPER, N_TEAMS


def get_dominant_color(crop: np.ndarray, k: int = 1) -> np.ndarray:
    """Extract dominant jersey color from the top half of a player crop (torso).

    Two robustness improvements over naive RGB clustering:
    1. Grass-green pixels are filtered out first, so background doesn't
       dominate the result (common when boxes include pitch background).
    2. Color is represented in HSV hue+saturation (not raw BGR), since hue
       is far less sensitive to lighting/shadow variation across the pitch
       than raw color channels -- this is what actually separates jerseys
       reliably in outdoor broadcast footage.
    """
    h = crop.shape[0]
    torso = crop[: h // 2, :, :]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_grass_mask = grass_mask == 0

    hs_pixels = hsv[non_grass_mask][:, :2].astype(np.float32)  # hue, saturation only

    if len(hs_pixels) < 10:
        hs_pixels = hsv.reshape(-1, 3)[:, :2].astype(np.float32)

    if len(hs_pixels) == 0:
        return np.array([0, 0])

    kmeans = KMeans(n_clusters=k, n_init=3, random_state=0).fit(hs_pixels)
    return kmeans.cluster_centers_[0]


def assign_teams(video_path: str, tracks_csv: str = PITCH_TRACKS_CSV):
    df = pd.read_csv(tracks_csv)
    player_mask = df["class_id"].isin([CLASS_PLAYER, CLASS_GOALKEEPER])
    player_df = df[player_mask].copy()

    cap = cv2.VideoCapture(video_path)
    colors = []
    current_frame = -1
    frame = None
    frame_read_ok = False

    for _, row in player_df.iterrows():
        if row["frame"] != current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            frame_read_ok, frame = cap.read()
            current_frame = row["frame"]

        if not frame_read_ok or frame is None:
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
    team_kmeans = KMeans(n_clusters=N_TEAMS, n_init=10, random_state=0).fit(colors)
    player_df["team_id"] = team_kmeans.labels_

    df = df.merge(
        player_df[["frame", "track_id", "team_id"]],
        on=["frame", "track_id"], how="left"
    )
    df.to_csv(tracks_csv, index=False)
    print(f"Assigned teams to {len(player_df)} player detections -> {tracks_csv}")
    return df


if __name__ == "__main__":
    from config import VIDEO_PATH
    assign_teams(VIDEO_PATH)