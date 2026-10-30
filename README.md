# Bicep Curl Counter

A Python coursework project that counts complete elbow-flexion repetitions using OpenCV and MediaPipe Pose.

**Updated submission instructions:** share the GitHub repository link before Monday and demonstrate the counter live on Monday. The teacher confirmed that video submission is not required. The original `assignment/assignment.md` is retained as the technical task specification; its older video-submission instruction is superseded by this update.

## Run

From the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
python assignment/assignment_starter.py --duration 25
```

The provided downloader fetches MediaPipe model weights into `models/`; they are not committed to GitHub. Allow camera access when prompted.

1. Adjust position in **PREVIEW**. Keep the shoulder, elbow and wrist visible, with the arm moving in the image plane.
2. Start with the arm straight, click the counter window and press **Space**.
3. Perform about five curls. Below 50° enters the curled stage; returning above 150° completes one repetition.
4. The 25-second session stops automatically and saves a CSV, JSON settings and PNG plot locally. Space/Q stops early; R resets only the preview.

Use `--record` to additionally save the annotated camera video locally. `assignment/开始弯举计数.command` is a macOS double-click launcher that enables local video recording. No personal camera videos are included in this repository.

## Implementation

| Assignment task | Implementation |
|---|---|
| Elbow angle and visibility checks | `elbow_angle()` in `assignment_starter.py` |
| Two-threshold repetition counting | `RepCounter.update()` |
| Readable count, angle and stage | `draw_info()` |
| Four-column CSV | `RepLogger.write()` |
| Angle plot with completion markers | `assignment/plot_reps.py` |

The main program opens the camera and runs the pose model on each frame. The angle uses pixel-corrected 2D shoulder/elbow/wrist coordinates by default; `--3d` enables estimated world coordinates. The counter requires initial extension, rejects returns shorter than 0.3s, and rearms after a tracking gap longer than 1s while preserving completed reps. These checks do not certify exercise form or guarantee accuracy under occlusion or foreshortening.

Formal sessions start at 0s. The same clock feeds the state machine, logging and video. Plotting automatically reads the exact thresholds from the session’s JSON settings.

## Options

```bash
# Left arm or custom thresholds
python assignment/assignment_starter.py --arm left --duration 25
python assignment/assignment_starter.py --up-angle 60 --down-angle 145 --duration 25
# Plot saved measurements without a camera
python assignment/plot_reps.py example_data/reps_demo.csv --no-show
# Process a previously mirrored video without opening a window
python assignment/assignment_starter.py --video output/input.mp4 --unmirror --no-show
# Run tests after installing dependencies and downloading models
python -m unittest discover -s tests -v
```

## Report and example data

`report.md` answers the four report questions. `example_data/` contains preserved measurements and the graph from the October 23 demonstration, not a newly recorded version 2 trial. State-machine replay produces 5 counts with 50°/150° and 6 with a single 90° threshold. The graph contains no camera images; original camera videos remain local.

## Starter attribution

This assignment extends the teacher-provided [MediaPipe First Steps](https://github.com/Gil-gil-glitch/mediapipe-first-steps) starter, based on commit `8470c28`. The starter supplied camera capture, pose detection and arm drawing. The completed assignment adds elbow-angle calculation, counting, a readable overlay, CSV logging, a separate plotting script, recording-session control and regression tests. The teacher’s starter structure and requested filenames are retained.
