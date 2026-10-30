"""Behavior checks with hand-constructed geometry; no camera required."""
import csv
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from assignment.assignment_starter import RepCounter, RepLogger, elbow_angle


def pose(a=(1, 0, 0), b=(0, 0, 0), c=(0, 1, 0), arm="right"):
    from assignment.assignment_starter import ARMS
    image = [NS(x=0, y=0, z=0, visibility=1.0) for _ in range(33)]
    world = [NS(x=0, y=0, z=0, visibility=1.0) for _ in range(33)]
    for index, point in zip(ARMS[arm], (a, b, c)):
        world[index] = NS(x=point[0], y=point[1], z=point[2], visibility=1.0)
    return NS(pose_landmarks=[image], pose_world_landmarks=[world])


class AngleTests(unittest.TestCase):
    def test_known_3d_angles_on_both_arms(self):
        for arm in ("left", "right"):
            for wrist, want in [((-1, 0, 0), 180), ((0, 0, 1), 90), ((1, 1, 0), 45)]:
                with self.subTest(arm=arm, want=want):
                    self.assertAlmostEqual(elbow_angle(pose(c=wrist, arm=arm), arm), want)

    def test_no_person_or_world_points_returns_none(self):
        for result in (NS(pose_landmarks=[], pose_world_landmarks=[]),
                       NS(pose_landmarks=pose().pose_landmarks, pose_world_landmarks=[])):
            self.assertIsNone(elbow_angle(result, "right"))

    def test_each_hidden_joint_returns_none(self):
        for index in (12, 14, 16):
            result = pose()
            result.pose_landmarks[0][index].visibility = 0.49
            self.assertIsNone(elbow_angle(result, "right"))

    def test_degenerate_and_nonfinite_points_return_none(self):
        for wrist in ((0, 0, 0), (math.nan, 1, 0), (math.inf, 0, 0)):
            self.assertIsNone(elbow_angle(pose(c=wrist), "right"))


class CounterTests(unittest.TestCase):
    def test_five_complete_curls(self):
        counter = RepCounter(min_return_seconds=0)
        events = [counter.update(x) for x in [170] + [100, 40, 90, 160, 170] * 5]
        self.assertEqual(sum(events), 5)
        self.assertEqual(counter.reps, 5)
        self.assertEqual(counter.stage, "down")

    def test_jitter_and_partial_curls_never_double_count(self):
        counter = RepCounter(min_return_seconds=0)
        for angle in [170, 80, 149, 151, 100, 49, 51, 48, 50, 100, 149, 150]:
            self.assertFalse(counter.update(angle))
        self.assertTrue(counter.update(151))
        for angle in [149, 151, 149, 170, 170]:
            self.assertFalse(counter.update(angle))
        self.assertEqual(counter.reps, 1)

    def test_initially_bent_arm_must_first_straighten(self):
        counter = RepCounter(min_return_seconds=0)
        for angle in [40, 30, 100, 170]:
            self.assertFalse(counter.update(angle))
        self.assertEqual(counter.reps, 0)
        counter.update(40)
        self.assertTrue(counter.update(170))

    def test_tracking_loss_holds_state_and_reset_rearms(self):
        counter = RepCounter(min_return_seconds=0)
        counter.update(170)
        counter.update(40)
        for angle in [None, math.nan, math.inf, -1, 181]:
            self.assertFalse(counter.update(angle))
        self.assertEqual(counter.stage, "up")
        self.assertTrue(counter.update(170))
        counter.reset()
        for angle in [40, 170]:
            self.assertFalse(counter.update(angle))
        self.assertEqual(counter.reps, 0)

    def test_invalid_thresholds_are_rejected(self):
        for up, down in [(150, 50), (0, 150), (50, 180), (math.nan, 150)]:
            with self.assertRaises(ValueError):
                RepCounter(up, down)


class LoggerTests(unittest.TestCase):
    def test_csv_round_trip_has_missing_angle_and_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reps.csv"
            logger = RepLogger(path)
            counter = RepCounter(min_return_seconds=0)
            logger.write(None, counter)
            for angle in (170, 40, 170):
                counter.update(angle)
                logger.write(angle, counter)
            logger.close()
            with path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["angle_deg"], "")
            self.assertEqual(rows[-1]["reps"], "1")
            self.assertEqual(rows[-1]["stage"], "down")
            times = [float(row["time_s"]) for row in rows]
            self.assertGreaterEqual(times[0], 0)
            self.assertEqual(times, sorted(times))

class ReplayTimeTests(unittest.TestCase):
    def test_video_frames_use_source_duration_not_processing_speed(self):
        from assignment.assignment_starter import VideoClock
        clock = VideoClock(video_fps=25)
        times = [clock.now_ms() for _ in range(26)]
        self.assertEqual(times[:3], [0, 40, 80])
        self.assertEqual(times[-1], 1000)

    def test_csv_accepts_source_time_for_video_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'video.csv'
            logger = RepLogger(path)
            try:
                logger.write(170, RepCounter(), elapsed_s=2.5)
            finally:
                logger.close()
            with path.open(newline='') as file:
                row = next(csv.DictReader(file))
            self.assertEqual(float(row['time_s']), 2.5)

class CalibratedThresholdTests(unittest.TestCase):
    def test_custom_threshold_accepts_observed_3d_range(self):
        counter = RepCounter(up_angle=80, min_return_seconds=0)
        events = [counter.update(angle) for angle in (170, 140, 110, 76, 90, 130, 160)]
        self.assertEqual(sum(events), 1)
        self.assertEqual(counter.reps, 1)


class TimingFilterTests(unittest.TestCase):
    def test_short_tracking_jump_does_not_create_an_extra_rep(self):
        counter = RepCounter()
        observed = [(0, 170), (1, 40), (2, 160), (2.10, 40), (2.236, 152.320)]
        events = [counter.update(angle, timestamp_s=seconds) for seconds, angle in observed]
        self.assertEqual(events, [False, False, True, False, False])
        self.assertEqual(counter.reps, 1)
        self.assertEqual(counter.stage, "down")

    def test_normal_curls_still_count_after_a_rejected_jump(self):
        counter = RepCounter()
        observed = [(0, 170), (1, 40), (1.1, 160), (2, 40), (2.5, 160),
                    (3, 40), (3.5, 160)]
        events = [counter.update(angle, timestamp_s=seconds) for seconds, angle in observed]
        self.assertEqual(sum(events), 2)
        self.assertEqual(counter.reps, 2)


class PixelAngleTests(unittest.TestCase):
    def test_pixel_angle_uses_frame_aspect_ratio_instead_of_noisy_depth(self):
        result = pose()
        # Pixels: shoulder (320,0), elbow (0,0), wrist (320,320): exactly 45°.
        for i, x, y in [(12, 0.5, 0), (14, 0, 0), (16, 0.5, 2/3)]:
            result.pose_landmarks[0][i].x = x
            result.pose_landmarks[0][i].y = y
        self.assertAlmostEqual(elbow_angle(result, "right", frame_size=(640,480)), 45)
        self.assertAlmostEqual(elbow_angle(result, "right"), 90)

    def test_2d_does_not_require_estimated_world_coordinates(self):
        result = pose()
        result.pose_world_landmarks = []
        for i, x, y in [(12, 0.1, 0.5), (14, 0.5, 0.5), (16, 0.9, 0.5)]:
            result.pose_landmarks[0][i].x = x
            result.pose_landmarks[0][i].y = y
        self.assertAlmostEqual(elbow_angle(result, "right", frame_size=(640,480)), 180)


class TrackingContinuityTests(unittest.TestCase):
    def test_long_occlusion_cannot_complete_a_hidden_curl(self):
        counter = RepCounter()
        for seconds, angle in [(0,170),(.2,40),(1.3,None),(1.4,170)]:
            self.assertFalse(counter.update(angle,timestamp_s=seconds))
        self.assertEqual(counter.reps,0)
        counter.update(40,timestamp_s=1.8)
        self.assertTrue(counter.update(170,timestamp_s=2.2))

    def test_long_gap_without_missing_rows_still_rearms(self):
        counter = RepCounter()
        counter.update(170,timestamp_s=0)
        counter.update(40,timestamp_s=.2)
        self.assertFalse(counter.update(170,timestamp_s=5))
        self.assertEqual(counter.reps,0)

    def test_brief_occlusion_keeps_a_visible_cycle(self):
        counter = RepCounter()
        for seconds,angle in [(0,170),(.2,40),(.4,None),(.5,90)]:
            self.assertFalse(counter.update(angle,timestamp_s=seconds))
        self.assertTrue(counter.update(170,timestamp_s=.8))

    def test_out_of_order_frame_never_changes_state(self):
        counter = RepCounter()
        counter.update(170,timestamp_s=2)
        counter.update(40,timestamp_s=2.2)
        self.assertFalse(counter.update(170,timestamp_s=1))
        self.assertEqual(counter.stage,'up')
        self.assertTrue(counter.update(170,timestamp_s=2.6))


if __name__ == "__main__":
    unittest.main()
