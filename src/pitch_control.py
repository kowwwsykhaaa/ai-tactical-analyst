"""
UNIQUE FEATURE: Pitch Control Model.

For a given frame, computes which team "controls" each point on the pitch —
i.e. which team's nearest player could reach that point first, accounting
for player speed. This is a simplified version of the models used by
professional analytics teams (based on Spearman et al., "Physics-Based
Modelling of Pitch Control").

Simplification used here: time-to-reach a grid point = distance / max_speed
for each player; the team whose fastest player reaches a point wins that
cell. This is the "static Voronoi" baseline — a good, explainable starting
point. A note in the README explains how to extend it to a full
probabilistic model.

Output: a 2D grid per frame representing team control (values in [-1, 1],
negative = team 0, positive = team 1), which the dashboard renders as a
heatmap overlay on the pitch.
"""

import numpy as np
import pandas as pd

from config import (
    PITCH_LENGTH, PITCH_WIDTH, GRID_RESOLUTION,
    PLAYER_MAX_SPEED_MPS,
)


def build_pitch_grid():
    xs = np.arange(0, PITCH_LENGTH + GRID_RESOLUTION, GRID_RESOLUTION)
    ys = np.arange(0, PITCH_WIDTH + GRID_RESOLUTION, GRID_RESOLUTION)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y


def compute_pitch_control_for_frame(frame_df: pd.DataFrame) -> np.ndarray:
    """
    frame_df: rows for one frame, must have columns pitch_x, pitch_y, team_id
    (players only — filter out ball/referee before calling).

    Returns: 2D array, same shape as pitch grid, values in [-1, 1].
    """
    grid_x, grid_y = build_pitch_grid()
    control = np.zeros_like(grid_x)

    team0 = frame_df[frame_df.team_id == 0][["pitch_x", "pitch_y"]].to_numpy()
    team1 = frame_df[frame_df.team_id == 1][["pitch_x", "pitch_y"]].to_numpy()

    if len(team0) == 0 or len(team1) == 0:
        return control  # no data this frame, neutral grid

    # time-to-reach = min distance from any player on the team, / max speed
    def min_time_to_reach(team_positions):
        # shape: (n_players, H, W) distances, take min over players
        dists = np.stack([
            np.sqrt((grid_x - px) ** 2 + (grid_y - py) ** 2)
            for px, py in team_positions
        ])
        return dists.min(axis=0) / PLAYER_MAX_SPEED_MPS

    t0 = min_time_to_reach(team0)
    t1 = min_time_to_reach(team1)

    # Control score: negative -> team0 faster/closer, positive -> team1
    # Normalize with a sigmoid so it's bounded in [-1, 1]
    diff = t0 - t1
    control = np.tanh(diff)  # smooth, bounded, zero when equidistant

    return control


def compute_pitch_control_sequence(tracks_df: pd.DataFrame, frames: list[int]):
    """Compute pitch control grids for a list of frames. Returns dict frame -> grid."""
    results = {}
    for f in frames:
        frame_df = tracks_df[
            (tracks_df.frame == f) & (tracks_df.team_id.notna())
        ]
        results[f] = compute_pitch_control_for_frame(frame_df)
    return results


def team_territory_share(control_grid: np.ndarray) -> tuple[float, float]:
    """Quick summary stat: % of pitch each team controls at this frame."""
    total = control_grid.size
    team1_share = (control_grid > 0).sum() / total
    team0_share = (control_grid < 0).sum() / total
    return team0_share, team1_share


if __name__ == "__main__":
    from config import PITCH_TRACKS_CSV
    df = pd.read_csv(PITCH_TRACKS_CSV)
    sample_frames = sorted(df.frame.unique())[:5]
    grids = compute_pitch_control_sequence(df, sample_frames)
    for f, g in grids.items():
        t0, t1 = team_territory_share(g)
        print(f"Frame {f}: Team 0 controls {t0:.1%}, Team 1 controls {t1:.1%}")
