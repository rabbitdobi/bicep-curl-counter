"""Ensure exported video contains the exact displayed frames, in order."""
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from assignment import assignment_starter as app


class DisplayRecorderTests(unittest.TestCase):
    def test_saved_video_contains_each_display_frame(self):
        self.assertTrue(hasattr(app, 'DisplayRecorder'), 'Display recorder not implemented')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reps.mp4'
            recorder = app.DisplayRecorder(path, fps=25, size=(320, 240))
            for shade in (30, 100, 200):
                recorder.write(np.full((240, 320, 3), shade, dtype=np.uint8))
            recorder.close()
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 25)
                frames = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
            finally:
                capture.release()
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].shape, (240, 320, 3))
            self.assertLess(abs(float(frames[0].mean())-30), 8)
            self.assertLess(abs(float(frames[-1].mean())-200), 8)

    def test_recording_duration_tracks_elapsed_time_when_processing_rate_varies(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'timed.mp4'
            recorder = app.DisplayRecorder(path, fps=10, size=(320, 240))
            try:
                for seconds, shade in [(0, 30), (0.2, 100), (0.4, 200)]:
                    recorder.write(np.full((240, 320, 3), shade, dtype=np.uint8), timestamp_s=seconds)
            finally:
                recorder.close()
            cap = cv2.VideoCapture(str(path))
            try:
                self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 5)
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 10)
            finally:
                cap.release()

    def test_final_display_frame_is_saved_when_it_arrives_between_video_ticks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last_frame.mp4'
            recorder = app.DisplayRecorder(path,fps=15,size=(320,240))
            recorder.write(np.full((240,320,3),30,dtype=np.uint8),timestamp_s=0)
            recorder.write(np.full((240,320,3),200,dtype=np.uint8),timestamp_s=.02)
            recorder.close()
            cap=cv2.VideoCapture(str(path))
            frames=[]
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    frames.append(frame)
            finally:
                cap.release()
            self.assertLess(abs(float(frames[-1].mean())-200),8)
