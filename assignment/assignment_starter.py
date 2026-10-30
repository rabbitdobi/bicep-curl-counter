"""
Assignment: Rep counter
=======================

Count bicep curls from the camera, using your elbow angle.

    arm straight (angle large)  ->  curl up (angle small)  ->  back down  =  1 rep

Uses pixel-corrected 2D angles by default, optional 3D world angles,
two angle thresholds, a short-return filter and a CSV row per frame.
Read assignment.md for the assignment and README.md for recording steps.

How to run
----------
    python assignment/assignment_starter.py
    python assignment/assignment_starter.py --arm left
    python assignment/assignment_starter.py --video output/my_curls.mp4 --unmirror

Controls
--------
    Space      start / stop formal recording (camera only starts in preview)
    r          reset the preview counter
    q or Esc   quit and save any recording

Useful lessons: 09 (pose), 10 (states and dead zones), 11 (angles), 12 (CSV).
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Assignment - Rep counter"
REPO_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_DIR / "models" / "pose_landmarker_lite.task"
OUTPUT_DIR = REPO_DIR / "output"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# arm -> (shoulder, elbow, wrist) landmark numbers
ARMS = {"left": (11, 13, 15), "right": (12, 14, 16)}
MIN_VISIBILITY = 0.5

# A rep counts when the angle goes below UP_ANGLE and then back above DOWN_ANGLE.
# You may tune these numbers. Explain your choice in your report.
UP_ANGLE = 50
DOWN_ANGLE = 150
# Reject implausibly short returns caused by isolated pose-estimation jumps.
MIN_RETURN_SECONDS = 0.3


class VideoClock:
    """Monotonic webcam time, or frame-rate-based source time for video files."""

    def __init__(self, video_fps=None):
        self.start = time.monotonic()
        self.last = -1
        self.video_fps = video_fps
        self.frame_index = 0

    def now_ms(self):
        if self.video_fps is None:
            timestamp = int((time.monotonic() - self.start) * 1000)
        else:
            timestamp = round(self.frame_index * 1000 / self.video_fps)
            self.frame_index += 1
        self.last = max(timestamp, self.last + 1)
        return self.last


# ---------------------------------------------------------------------------
# Task 1: the elbow angle
# ---------------------------------------------------------------------------
def elbow_angle(result, arm, frame_size=None):
    """Return the elbow angle of one arm in degrees, or None.

    Args:
        result: A PoseLandmarker result (check it has a person first!).
        arm:    "left" or "right".
        frame_size: (width, height) for a pixel-corrected 2D angle;
                    None selects world-coordinate 3D angles.

    Returns:
        The angle at the elbow between shoulder and wrist (0-180 degrees),
        or None if there is no person or a point's visibility is below
        MIN_VISIBILITY.

    2D avoids unstable estimated depth for an arm moving in the image plane.
    3D remains available for comparison. Visibility is checked in both modes.
    """
    if not result.pose_landmarks:
        return None
    indices = ARMS[arm]
    image = result.pose_landmarks[0]
    for i in indices:
        visibility = image[i].visibility
        if visibility is None or not math.isfinite(visibility) or visibility < MIN_VISIBILITY:
            return None
    if frame_size is not None:
        width, height = frame_size
        a, b, c = [(image[i].x * width, image[i].y * height) for i in indices]
    else:
        if not result.pose_world_landmarks:
            return None
        world = result.pose_world_landmarks[0]
        a, b, c = [(world[i].x, world[i].y, world[i].z) for i in indices]
    if not all(math.isfinite(value) for point in (a, b, c) for value in point):
        return None
    ba = [p - q for p, q in zip(a, b)]
    bc = [p - q for p, q in zip(c, b)]
    length = math.dist(a, b) * math.dist(c, b)
    if length < 1e-12:
        return None
    cosine = sum(p * q for p, q in zip(ba, bc)) / length
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


# ---------------------------------------------------------------------------
# Task 2: counting reps
# ---------------------------------------------------------------------------
class RepCounter:
    """Counts reps from a stream of elbow angles.

    It remembers which "stage" the arm is in:
        "down": arm extended, waiting for a curl
        "up":   arm curled, waiting to go back down
    """

    def __init__(self, up_angle=UP_ANGLE, down_angle=DOWN_ANGLE,
                 min_return_seconds=MIN_RETURN_SECONDS, max_tracking_gap_s=1.0):
        if not 0 < up_angle <= down_angle < 180:
            raise ValueError("Thresholds must satisfy 0 < up <= down < 180 degrees.")
        self.up_angle = up_angle
        self.down_angle = down_angle
        if not math.isfinite(min_return_seconds) or min_return_seconds < 0:
            raise ValueError("Minimum return time must be finite and nonnegative.")
        self.min_return_seconds = min_return_seconds
        if not math.isfinite(max_tracking_gap_s) or max_tracking_gap_s <= 0:
            raise ValueError("Tracking gap must be finite and positive.")
        self.max_tracking_gap_s = max_tracking_gap_s
        self.reset()

    def update(self, angle, timestamp_s=None):
        """Feed in this frame's angle (may be None).

        Returns:
            True if a rep was completed in THIS frame, otherwise False.

        Hold state inside the dead zone and across brief tracking gaps.
        A gap longer than one second discards the pending curl, not the count.
        First observe extension, then a curl, then extension again.
        """
        now = time.monotonic() if timestamp_s is None else timestamp_s
        if not math.isfinite(now) or (self.last_update_s is not None and now < self.last_update_s):
            return False
        self.last_update_s = now
        if self.last_valid_s is not None and now - self.last_valid_s > self.max_tracking_gap_s:
            self.stage, self.ready, self.curl_started_at = "down", False, None
            self.last_valid_s = None
        if angle is None or not math.isfinite(angle) or not 0 <= angle <= 180:
            return False
        self.last_valid_s = now
        if not self.ready:
            self.ready = angle > self.down_angle
            return False
        if self.stage == "down" and angle < self.up_angle:
            self.stage = "up"
            self.curl_started_at = now
        elif self.stage == "up" and angle > self.down_angle:
            self.stage = "down"
            duration = now - self.curl_started_at
            self.curl_started_at = None
            if duration < self.min_return_seconds:
                self.rejected_jumps += 1
                return False
            self.reps += 1
            return True
        return False

    def reset(self):
        self.stage = "down"
        self.reps = 0
        self.ready = False
        self.curl_started_at = None
        self.rejected_jumps = 0
        self.last_update_s = None
        self.last_valid_s = None


# ---------------------------------------------------------------------------
# Task 3: show it on screen
# ---------------------------------------------------------------------------
def draw_info(display, angle, counter, mode="2D", recording=False, elapsed_s=0, duration=None):
    """Draw the rep count, the stage and the current angle on the picture.

    Args:
        display: The (mirrored) picture to draw on.
        angle:   This frame's elbow angle, or None.
        counter: The RepCounter.

    The panel is drawn on a fixed-size canvas and scaled for small cameras.
    """
    import numpy as np

    panel = np.full((310, 360, 3), (31, 25, 20), dtype=np.uint8)
    white = (245, 245, 245)
    accent = (175, 230, 75)
    muted = (195, 190, 180)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(value, x, y, size, color=white, thickness=1):
        cv2.putText(panel, value, (x, y), font, size, color, thickness, cv2.LINE_AA)

    text(f"CURL COUNTER / {mode}", 18, 30, 0.65, accent)
    status = f"REC {elapsed_s:.1f}s" if recording else "PREVIEW"
    if recording and duration is not None:
        status = f"REC {max(0, duration - elapsed_s):.1f}s left"
    text(status, 180, 62, 0.48, accent if recording else muted)
    text(str(counter.reps), 16, 120, 2.7, white, 5)
    text("REPS", 180, 115, 0.9, muted, 2)
    angle_text = "--" if angle is None else f"{angle:.0f} deg"
    text(f"ANGLE  {angle_text}", 18, 155, 0.8, white, 2)
    text(f"STAGE  {counter.stage.upper()}", 18, 186, 0.7, accent, 2)
    cv2.rectangle(panel, (18, 203), (340, 219), (65, 61, 55), -1)
    if angle is not None:
        span = counter.down_angle - counter.up_angle
        progress = ((counter.down_angle - angle) / span if span else float(angle < counter.up_angle))
        fill = round(322 * max(0.0, min(1.0, progress)))
        if fill:
            cv2.rectangle(panel, (18, 203), (18 + fill, 219), accent, -1)
    if angle is None:
        instruction = "ARM NOT VISIBLE - move into view"
    elif not counter.ready:
        instruction = "Straighten your arm to start"
    elif counter.stage == "up":
        instruction = "Lower until your arm is straight"
    else:
        instruction = "Curl your forearm up"
    text(instruction, 18, 249, 0.48, white)
    text(f"UP < {counter.up_angle:g} / DOWN > {counter.down_angle:g} deg", 18, 274, 0.48, muted)
    text("SPACE: start/stop  Q: quit  R: reset preview", 18, 296, 0.38, muted)
    height, width = display.shape[:2]
    scale = min(1.0, (width - 20) / 360, (height - 20) / 310)
    if scale <= 0:
        return
    panel = cv2.resize(panel, (round(360 * scale), round(310 * scale)))
    ph, pw = panel.shape[:2]
    display[10:10 + ph, 10:10 + pw] = panel


# ---------------------------------------------------------------------------
# Task 4: save the data
# ---------------------------------------------------------------------------
class RepLogger:
    """Saves time, angle, stage and rep count to a CSV file, one row per frame."""

    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.started_at = time.monotonic()
        self.writer.writerow(["time_s", "angle_deg", "stage", "reps"])
        self.file.flush()

    def write(self, angle, counter, elapsed_s=None):
        """Use supplied source-video time, or elapsed webcam recording time."""
        if elapsed_s is None:
            elapsed_s = time.monotonic() - self.started_at
        self.writer.writerow([
            f"{elapsed_s:.6f}",
            "" if angle is None else f"{angle:.3f}",
            counter.stage, counter.reps,
        ])
        self.file.flush()

    def close(self):
        self.file.close()


class DisplayRecorder:
    """Save the same annotated camera frames shown in the counter window."""

    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.fps = fps
        self.started_at = None
        self.frames_written = 0
        self.previous = None
        self.pending_last_frame = False
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError(f"Could not create video: {path}")

    def write(self, display, timestamp_s=None):
        if timestamp_s is None:
            self.writer.write(display)
            self.frames_written += 1
            self.pending_last_frame = False
            return
        if self.started_at is None:
            self.started_at = timestamp_s
        target = round((timestamp_s - self.started_at) * self.fps)
        emitted_current = False
        # Hold the preceding frame across gaps; drop extra frames at high FPS.
        # Thus playback duration follows elapsed time rather than CPU speed.
        while self.frames_written <= target:
            frame = self.previous if self.previous is not None and self.frames_written < target else display
            self.writer.write(frame)
            emitted_current = self.frames_written == target
            self.frames_written += 1
        self.previous = display.copy()
        self.pending_last_frame = not emitted_current

    def close(self):
        if self.pending_last_frame and self.previous is not None:
            self.writer.write(self.previous)
            self.frames_written += 1
            self.pending_last_frame = False
        self.writer.release()


class RecordingSession:
    """Own the CSV, optional video and settings for one formal recording."""

    def __init__(self, output_dir, counter, arm="right", mode="2D", record_video=False, fps=15):
        output_dir.mkdir(parents=True, exist_ok=True)
        name = f"reps_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}"
        self.csv_path = output_dir / f"{name}.csv"
        self.logger = RepLogger(self.csv_path)
        self.counter, self.arm, self.mode = counter, arm, mode
        self.record_video, self.fps = record_video, fps
        self.recorder = None
        self.started_at = None
        self.elapsed_s = 0
        self.frames = 0
        self.closed = False

    def write(self, display, angle, timestamp_s):
        if self.started_at is None:
            self.started_at = timestamp_s
        self.elapsed_s = timestamp_s - self.started_at
        self.logger.write(angle, self.counter, elapsed_s=self.elapsed_s)
        if self.record_video:
            if self.recorder is None:
                h, w = display.shape[:2]
                self.recorder = DisplayRecorder(self.csv_path.with_suffix(".mp4"), self.fps, (w, h))
            self.recorder.write(display, timestamp_s=self.elapsed_s)
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.logger.close()
        if self.recorder is not None:
            self.recorder.close()
        settings = {
            "software_version": 2, "angle_mode": self.mode, "arm": self.arm,
            "up_angle": self.counter.up_angle, "down_angle": self.counter.down_angle,
            "minimum_return_s": self.counter.min_return_seconds,
            "max_tracking_gap_s": self.counter.max_tracking_gap_s,
            "frames": self.frames, "duration_s": self.elapsed_s,
            "final_reps": self.counter.reps, "rejected_short_returns": self.counter.rejected_jumps,
            "video_fps": self.fps if self.record_video else None,
            "video_is_mirrored": True if self.record_video else None,
        }
        self.csv_path.with_suffix(".json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        self.closed = True


# ---------------------------------------------------------------------------
# You don't need to change anything below this line (but you may).
# ---------------------------------------------------------------------------
def draw_arm(display, result, arm):
    """Draw the tracked arm on the mirrored display picture."""
    if not result.pose_landmarks:
        return
    landmarks = result.pose_landmarks[0]
    h, w = display.shape[:2]
    points = [(int((1 - landmarks[i].x) * w), int(landmarks[i].y * h)) for i in ARMS[arm]]
    cv2.line(display, points[0], points[1], (255, 255, 255), 4)
    cv2.line(display, points[1], points[2], (255, 255, 255), 4)
    for point in points:
        cv2.circle(display, point, 8, (0, 200, 255), -1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Count bicep curls with MediaPipe Pose.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--video", type=Path,
                        help="Use a video file instead of the camera (e.g. one made with lesson 04).")
    parser.add_argument("--unmirror", action="store_true",
                        help="Flip a mirrored video back before detection. Use this with videos "
                             "recorded by lesson 04, which saves the mirrored picture; otherwise "
                             "your left and right arms swap (see lesson 10).")
    parser.add_argument("--arm", choices=["left", "right"], default="right",
                        help="Which of YOUR arms to track. (default: right)")
    parser.add_argument("--record", action="store_true",
                        help="Save the displayed camera and counter together as a matching MP4.")
    parser.add_argument("--3d", dest="three_d", action="store_true",
                        help="Use estimated world-coordinate 3D angles instead of 2D pixel angles.")
    parser.add_argument("--up-angle", type=float, default=UP_ANGLE, help="Curl threshold in degrees.")
    parser.add_argument("--down-angle", type=float, default=DOWN_ANGLE, help="Extension threshold in degrees.")
    parser.add_argument("--duration", type=float, help="Stop automatically after this many recorded seconds.")
    parser.add_argument("--no-show", action="store_true", help="Process a video file without opening a window.")
    args = parser.parse_args(argv)
    if not 0 < args.up_angle <= args.down_angle < 180:
        parser.error("Thresholds must satisfy 0 < up <= down < 180.")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("Duration must be finite and positive.")
    if args.no_show and args.video is None:
        parser.error("--no-show requires --video; live recordings start with SPACE.")
    return args


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun: python download_models.py")
    counter = RepCounter(args.up_angle, args.down_angle)
    mode = "3D" if args.three_d else "2D"
    session = None

    def close_windows():
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    try:
        with ExitStack() as resources:
            detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=RunningMode.VIDEO, num_poses=1))
            resources.callback(detector.close)
            source = str(args.video) if args.video else args.camera
            cap = cv2.VideoCapture(source)
            resources.callback(cap.release)
            if not cap.isOpened():
                raise SystemExit(f"Could not open {source}.")
            fps = cap.get(cv2.CAP_PROP_FPS) if args.video else None
            if args.video and (not math.isfinite(fps) or fps <= 0):
                raise SystemExit("Video has no usable frame rate. Export it at a fixed frame rate.")
            clock = VideoClock(video_fps=fps)
            if not args.no_show:
                resources.callback(close_windows)
            if args.video:
                session = RecordingSession(OUTPUT_DIR, counter, args.arm, mode, args.record, fps)
            else:
                print("PREVIEW: adjust your position. Press SPACE to start, Q to quit.")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.unmirror:
                    frame = cv2.flip(frame, 1)
                if frame.shape[1] > 960:
                    height = round(frame.shape[0] * 960 / frame.shape[1])
                    frame = cv2.resize(frame, (960, height))
                timestamp_s = clock.now_ms() / 1000
                result = detector.detect_for_video(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), round(timestamp_s * 1000))
                h, w = frame.shape[:2]
                angle = elbow_angle(result, args.arm, None if args.three_d else (w, h))
                if counter.update(angle, timestamp_s=timestamp_s) and session is not None:
                    print(f"Rep {counter.reps}!")
                elapsed = (timestamp_s - session.started_at
                           if session is not None and session.started_at is not None else 0)
                display = cv2.flip(frame, 1)
                draw_arm(display, result, args.arm)
                draw_info(display, angle, counter, mode, session is not None, elapsed, args.duration)
                if session is not None:
                    session.write(display, angle, timestamp_s)
                if not args.no_show:
                    cv2.imshow(WINDOW_NAME, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == 32:
                        if session is not None:
                            break
                        counter.reset()
                        session = RecordingSession(OUTPUT_DIR, counter, args.arm, mode, args.record, 15)
                        print(f"Recording to {session.csv_path}")
                    elif key == ord("r") and session is None:
                        counter.reset()
                if session is not None and args.duration is not None and session.elapsed_s >= args.duration:
                    break
    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        if session is not None:
            session.close()
            print(f"Saved {session.frames} frames, {counter.reps} reps: {session.csv_path}")
            if session.frames:
                plotter = Path(__file__).with_name("plot_reps.py")
                result = subprocess.run([sys.executable, str(plotter), str(session.csv_path), "--no-show"])
                if result.returncode:
                    print("CSV/video saved, but plotting failed; see the message above.")
        else:
            print("Preview closed. No recording was created.")


if __name__ == "__main__":
    main()
