# AI Tactical Analyst

Detects players and the ball in football video, tracks them across frames,
maps positions onto real pitch coordinates, assigns teams by jersey color,
and visualizes tactical structure — including a **pitch control model**,
the space-control surface used by professional analytics teams.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a pretrained football detection model from Roboflow Universe (search
"football player detection") and save it to `models/football-player-detection.pt`.
Or fine-tune a base YOLOv8 checkpoint on a small labeled clip if you want the
"trained my own model" line on your resume too.

Put a short match clip (30–90 sec is plenty to start) at `data/match_clip.mp4`.

## Pipeline (run in order)

```bash
cd src
python detect_track.py       # -> outputs/tracks.csv, outputs/annotated.mp4
python homography.py         # click 4 pitch reference points -> outputs/pitch_tracks.csv
python team_assign.py        # adds team_id column
python pitch_control.py      # sanity-check the control model on a few frames
python formation.py          # sanity-check formation clustering
streamlit run dashboard.py   # full interactive dashboard
```

## Why this is a strong resume project

- **End-to-end CV pipeline**: detection -> multi-object tracking -> geometric
  calibration -> unsupervised clustering -> interactive visualization. That's
  five distinct technical skills in one pipeline, not a single-notebook demo.
- **Pitch control model** is the differentiator. Most student sports-analytics
  projects stop at "here are dots on a pitch." Computing a Voronoi-style
  space-control surface (time-to-reach per grid cell, per team) shows you can
  translate a research concept (Spearman et al.) into working code — that's
  a strong interview talking point, especially for ML/analytics-adjacent roles.
- Talk about trade-offs you made (e.g. static vs probabilistic pitch control,
  homography calibration limitations, occlusion handling in tracking) —
  interviewers care more about your reasoning than a perfect demo.

## Extending the pitch control model further (optional, for extra depth)

The current version is the "static" Voronoi baseline: nearest-player-wins per
cell. To go further (and have an even stronger differentiator):
- Add a *probabilistic* control model: instead of a hard time-to-reach cutoff,
  use a logistic/sigmoid function over the time difference, and account for
  player *current velocity* and reaction time (not just position) —
  this is what Spearman's original physics-based model does.
- Add **passing lanes**: overlay which teammates are reachable by a safe pass
  given opponent pitch control along the pass line.
- Add an **expected threat (xT) surface**: a static heatmap of how dangerous
  each pitch zone is, combined with control to show "controlled AND dangerous"
  zones — this is close to what club analytics departments actually build.

## Known limitations to mention honestly in interviews

- Homography is computed once from 4 manually-clicked points; a moving camera
  would need per-frame recalibration (could extend with pitch-line detection).
- Team assignment via color clustering can misfire on similar kit colors —
  worth mentioning as a known edge case with a proposed fix (e.g. incorporate
  spatial priors, or manual correction UI).
- Speed constant for pitch control (`PLAYER_MAX_SPEED_MPS`) is a fixed
  assumption, not measured per player — a real system would estimate this
  from tracked displacement over time.
