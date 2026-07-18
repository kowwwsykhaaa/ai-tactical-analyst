"""
Interactive Streamlit dashboard.
Run with: streamlit run src/dashboard.py

Tabs:
    - Player positions on pitch (per frame, via mplsoccer)
    - Pitch control heatmap (the unique feature)
    - Formation timeline
    - Upload & Analyze (live): upload your own clip, click pitch corners,
      and get a lightweight live analysis run in the browser
"""

import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from config import PITCH_TRACKS_CSV, PITCH_LENGTH, PITCH_WIDTH
from pitch_control import (
    compute_pitch_control_for_frame, build_pitch_grid, team_territory_share,
)
from formation import compute_windowed_formations
import live_pipeline as lp

st.set_page_config(page_title="AI Tactical Analyst", layout="wide")
st.title("⚽ AI Tactical Analyst")

DASHBOARD_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = DASHBOARD_DIR.parent / "sample_data" / "pitch_tracks.csv"


@st.cache_data
def load_data():
    try:
        return pd.read_csv(PITCH_TRACKS_CSV)
    except FileNotFoundError:
        return pd.read_csv(SAMPLE_DATA_PATH)


try:
    df = load_data()
except FileNotFoundError:
    st.error("No tracking data found. Run the pipeline (detect_track.py -> "
              "homography.py -> team_assign.py) first to generate outputs/pitch_tracks.csv")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Player Positions", "Pitch Control (unique)", "Formation Timeline", "Upload & Analyze (live)"]
)

frames = sorted(df.frame.unique())
frame = st.slider("Frame", int(frames[0]), int(frames[-1]), int(frames[0]))
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

with tab4:
    st.caption(
        f"Upload your own short clip (max {lp.MAX_SECONDS}s processed, CPU-only "
        "so it's noticeably slower than the local pipeline). Click 4 pitch corners "
        "in order: top-left, top-right, bottom-right, bottom-left."
    )

    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov"])

    if uploaded is not None:
        if "live_video_path" not in st.session_state or st.session_state.get("live_video_name") != uploaded.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.close()
            st.session_state.live_video_path = tmp.name
            st.session_state.live_video_name = uploaded.name
            st.session_state.live_points = []

        video_path = st.session_state.live_video_path

        first_frame_bgr = lp.get_first_frame(video_path)
        first_frame_rgb = first_frame_bgr[:, :, ::-1]
        img = Image.fromarray(first_frame_rgb)

        st.write(f"Points clicked: {len(st.session_state.live_points)} / 4")
        coords = streamlit_image_coordinates(img, key="pitch_click")

        if coords is not None and len(st.session_state.live_points) < 4:
            point = (coords["x"], coords["y"])
            if not st.session_state.live_points or st.session_state.live_points[-1] != point:
                st.session_state.live_points.append(point)
                st.rerun()

        col_a, col_b = st.columns(2)
        if col_a.button("Reset points"):
            st.session_state.live_points = []
            st.rerun()

        run_disabled = len(st.session_state.live_points) != 4
        if col_b.button("Run analysis", disabled=run_disabled):
            with st.spinner("Detecting and tracking players..."):
                progress = st.progress(0.0)
                live_df = lp.run_detection_and_tracking(
                    video_path, progress_callback=lambda p: progress.progress(p)
                )
                progress.empty()

            if live_df.empty:
                st.warning("No detections found. Try a different clip.")
            else:
                with st.spinner("Computing pitch coordinates..."):
                    H = lp.compute_homography(st.session_state.live_points)
                    live_df = lp.apply_homography(live_df, H)

                with st.spinner("Assigning teams by jersey color..."):
                    live_df = lp.assign_teams(video_path, live_df)

                st.session_state.live_results = live_df
                st.success(f"Done — {len(live_df)} detections across "
                            f"{live_df.frame.nunique()} frames.")

        if "live_results" in st.session_state:
            live_df = st.session_state.live_results
            live_frames = sorted(live_df.frame.unique())
            live_frame = st.slider(
                "Live result frame", int(live_frames[0]), int(live_frames[-1]),
                int(live_frames[0]), key="live_frame_slider"
            )
            live_frame_df = live_df[(live_df.frame == live_frame) & (live_df.team_id.notna())]

            pitch = Pitch(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                           pitch_color="grass", line_color="white")
            fig, ax = pitch.draw(figsize=(10, 6.5))
            control_grid = compute_pitch_control_for_frame(live_frame_df)
            if control_grid.any():
                grid_x, grid_y = build_pitch_grid()
                ax.contourf(grid_x, grid_y, control_grid, levels=20, cmap="RdBu", alpha=0.6)
            for team_id, color in zip([0, 1], ["red", "blue"]):
                team_pts = live_frame_df[live_frame_df.team_id == team_id]
                pitch.scatter(team_pts.pitch_x, team_pts.pitch_y, ax=ax, color=color,
                               edgecolors="black", s=120)
            st.pyplot(fig)