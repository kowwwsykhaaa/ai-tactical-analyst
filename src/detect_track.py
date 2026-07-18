"""
Step 1: Detect players/ball/referee per frame with YOLOv8,
track them across frames with ByteTrack (via `supervision`).

Output: outputs/tracks.csv with columns
    frame, track_id, class_id, x1, y1, x2, y2, cx, cy
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

from config import VIDEO_PATH, TRACKS_CSV, YOLO_MODEL_PATH


def run_detection_and_tracking(video_path: str = VIDEO_PATH, save_video: bool = True):
    model = YOLO(YOLO_MODEL_PATH)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("outputs/annotated.mp4", fourcc, fps, (width, height))

    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        for box, track_id, class_id in zip(
            detections.xyxy, detections.tracker_id, detections.class_id
        ):
            if track_id is None:
                continue
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, y2  # foot position, not center, for pitch mapping
            rows.append({
                "frame": frame_idx,
                "track_id": int(track_id),
                "class_id": int(class_id),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy,
            })

        if save_video:
            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(TRACKS_CSV, index=False)
    print(f"Saved {len(df)} detections across {frame_idx} frames -> {TRACKS_CSV}")
    return df


if __name__ == "__main__":
    run_detection_and_tracking()
