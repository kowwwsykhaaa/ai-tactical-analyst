"""
Step 2: Map image-pixel coordinates to real-world pitch coordinates (meters)
using a homography computed from 4+ manually-selected reference points.

Usage:
    1. Run `select_reference_points(video_path)` once to click 4 known pitch
       landmarks (e.g. corner flags, penalty box corners) in the first frame.
    2. Feed the resulting matrix into `apply_homography` on tracks.csv.
"""

import cv2
import numpy as np
import pandas as pd

from config import TRACKS_CSV, PITCH_TRACKS_CSV, PITCH_REFERENCE_POINTS


def select_reference_points(video_path: str):
    """Interactive tool: click 4 points on the first frame matching
    PITCH_REFERENCE_POINTS (in the same order)."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame")

    clicked = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 pitch reference points", frame)

    cv2.imshow("Select 4 pitch reference points", frame)
    cv2.setMouseCallback("Select 4 pitch reference points", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked) != 4:
        raise ValueError(f"Expected 4 points, got {len(clicked)}")
    return np.array(clicked, dtype=np.float32)


def compute_homography(image_points: np.ndarray) -> np.ndarray:
    pitch_points = np.array(PITCH_REFERENCE_POINTS, dtype=np.float32)
    H, _ = cv2.findHomography(image_points, pitch_points)
    return H


def apply_homography(H: np.ndarray, tracks_csv: str = TRACKS_CSV):
    df = pd.read_csv(tracks_csv)
    pts = df[["cx", "cy"]].to_numpy(dtype=np.float32).reshape(-1, 1, 2)
    pitch_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    df["pitch_x"] = pitch_pts[:, 0]
    df["pitch_y"] = pitch_pts[:, 1]
    df.to_csv(PITCH_TRACKS_CSV, index=False)
    print(f"Saved pitch-space coordinates -> {PITCH_TRACKS_CSV}")
    return df


if __name__ == "__main__":
    from config import VIDEO_PATH, OUTPUT_DIR
    img_pts = select_reference_points(VIDEO_PATH)
    H = compute_homography(img_pts)
    np.save(f"{OUTPUT_DIR}/homography_matrix.npy", H)