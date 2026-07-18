"""
Step 6: Interactive Streamlit dashboard.
Run with: streamlit run src/dashboard.py

Tabs:
    - Player positions on pitch (per frame, via mplsoccer)
    - Pitch control heatmap (the unique feature)
    - Formation timeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from config import PITCH_TRACKS_CSV, PITCH_LENGTH, PITCH_WIDTH
from pitch_control import (
    compute_pitch_control_for_frame, build_pitch_grid, team_territory_share,
)
from formation import compute_windowed_formations

st.set_page_config(page_title="AI Tactical Analyst", layout="wide")
st.title("⚽ AI Tactical Analyst")

@st.cache_data
def load_data():
    return pd.read_csv(PITCH_TRACKS_CSV)

try:
    df = load_data()
except FileNotFoundError:
    st.error("No tracking data found. Run the pipeline (detect_track.py -> "
              "homography.py -> team_assign.py) first to generate outputs/pitch_tracks.csv")
    st.stop()

frames = sorted(df.frame.unique())
frame = st.slider("Frame", int(frames[0]), int(frames[-1]), int(frames[0]))

tab1, tab2, tab3 = st.tabs(["Player Positions", "Pitch Control (unique)", "Formation Timeline"])

frame_df = df[(df.frame == frame) & (df.team_id.notna())]

with tab1:
    pitch = Pitch(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                   pitch_color="grass", line_color="white")
    fig, ax = pitch.draw(figsize=(10, 6.5))
    for team_id, color in zip([0, 1], ["red", "blue"]):
        team_pts = frame_df[frame_df.team_id == team_id]
        pitch.scatter(team_pts.pitch_x, team_pts.pitch_y, ax=ax, color=color, s=120)
    st.pyplot(fig)

with tab2:
    st.caption(
        "Each team's controlled zone, computed from time-to-reach every point "
        "on the pitch given player positions and max speed. Red = Team 0 territory, "
        "Blue = Team 1 territory."
    )
    control_grid = compute_pitch_control_for_frame(frame_df)
    if control_grid.any():
        t0_share, t1_share = team_territory_share(control_grid)
        col1, col2 = st.columns(2)
        col1.metric("Team 0 territory", f"{t0_share:.1%}")
        col2.metric("Team 1 territory", f"{t1_share:.1%}")

        grid_x, grid_y = build_pitch_grid()
        pitch = Pitch(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                       pitch_color="grass", line_color="white")
        fig, ax = pitch.draw(figsize=(10, 6.5))
        ax.contourf(grid_x, grid_y, control_grid, levels=20, cmap="RdBu", alpha=0.6)
        for team_id, color in zip([0, 1], ["red", "blue"]):
            team_pts = frame_df[frame_df.team_id == team_id]
            pitch.scatter(team_pts.pitch_x, team_pts.pitch_y, ax=ax, color=color,
                           edgecolors="black", s=120)
        st.pyplot(fig)
    else:
        st.warning("Not enough player data in this frame to compute control.")

with tab3:
    team_choice = st.radio("Team", [0, 1], horizontal=True)
    result, _ = compute_windowed_formations(df, team_id=team_choice)
    if result is not None:
        st.line_chart(result.set_index("window")["formation_cluster"])
        st.caption("Cluster id represents a distinct recurring shape (e.g. "
                    "defensive block vs high press vs attacking overload). "
                    "Cross-reference window -> frame range to inspect video moments.")
    else:
        st.warning("Not enough windows in this clip to detect formation changes.")
