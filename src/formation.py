"""
Step 5: Detect formation / shape changes by clustering average player
positions within a rolling time window, per team.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import FORMATION_WINDOW_FRAMES, N_FORMATION_CLUSTERS


def compute_windowed_formations(tracks_df: pd.DataFrame, team_id: int):
    """Average each player's position within each window, then cluster
    the resulting team-shape 'snapshots' to find recurring formation types."""
    df = tracks_df[tracks_df.team_id == team_id].copy()
    df["window"] = df["frame"] // FORMATION_WINDOW_FRAMES

    snapshots = []
    window_ids = []
    for window, wdf in df.groupby("window"):
        # average position per track_id within this window
        avg_pos = wdf.groupby("track_id")[["pitch_x", "pitch_y"]].mean()
        if len(avg_pos) < 4:  # skip incomplete windows (occlusion, subs, etc.)
            continue
        # sort by pitch_y then pitch_x for a consistent flattening order
        avg_pos = avg_pos.sort_values(["pitch_y", "pitch_x"])
        snapshots.append(avg_pos.to_numpy().flatten())
        window_ids.append(window)

    if len(snapshots) < N_FORMATION_CLUSTERS:
        print("Not enough windows to cluster formations reliably.")
        return None, None

    # pad/truncate snapshots to equal length (handles unequal player counts)
    max_len = max(len(s) for s in snapshots)
    padded = np.array([
        np.pad(s, (0, max_len - len(s)), constant_values=np.nan) for s in snapshots
    ])
    padded = np.nan_to_num(padded, nan=np.nanmean(padded))

    kmeans = KMeans(n_clusters=N_FORMATION_CLUSTERS, n_init=10, random_state=0)
    labels = kmeans.fit_predict(padded)

    result = pd.DataFrame({"window": window_ids, "formation_cluster": labels})
    return result, kmeans


if __name__ == "__main__":
    from config import PITCH_TRACKS_CSV
    df = pd.read_csv(PITCH_TRACKS_CSV)
    result, model = compute_windowed_formations(df, team_id=0)
    if result is not None:
        print(result)
