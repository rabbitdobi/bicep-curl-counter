"""Recording lifecycle tests use real files and a fake external camera."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
from assignment import assignment_starter as app


class SessionTests(unittest.TestCase):
    def test_session_normalizes_time_and_keeps_exact_settings(self):
        self.assertTrue(hasattr(app,'RecordingSession'),'Formal recording session is missing')
        with tempfile.TemporaryDirectory() as folder:
            counter=app.RepCounter(up_angle=60,down_angle=145)
            session=app.RecordingSession(Path(folder),counter,arm='left',mode='3D',record_video=True,fps=10)
            for seconds,angle in [(10,170),(10.2,40),(10.6,170)]:
                counter.update(angle,timestamp_s=seconds)
                session.write(np.full((240,320,3),80,dtype=np.uint8),angle,seconds)
            session.close()
            session.close()  # Shutdown must be safe even when called twice.
            rows=list(csv.DictReader(session.csv_path.read_text().splitlines()))
            self.assertEqual([float(r['time_s']) for r in rows],[0,.2,.6])
            self.assertEqual(rows[-1]['reps'],'1')
            info=json.loads(session.csv_path.with_suffix('.json').read_text())
            self.assertEqual((info['up_angle'],info['down_angle'],info['arm'],info['angle_mode']),
                             (60,145,'left','3D'))
            self.assertEqual(info['frames'],3)
            self.assertEqual(info['final_reps'],1)
            video=cv2.VideoCapture(str(session.csv_path.with_suffix('.mp4')))
            try:
                self.assertTrue(video.isOpened())
                self.assertEqual(video.get(cv2.CAP_PROP_FRAME_COUNT),7)
            finally:
                video.release()

    def run_camera(self,folder,keys):
        frame=np.zeros((240,320,3),dtype=np.uint8)
        capture=MagicMock()
        capture.isOpened.return_value=True
        capture.read.side_effect=[(True,frame.copy()) for _ in keys]+[(False,None)]
        capture.get.return_value=25
        detector=MagicMock()
        detector.detect_for_video.return_value=SimpleNamespace(pose_landmarks=[],pose_world_landmarks=[])
        with patch.object(app,'OUTPUT_DIR',Path(folder)), \
             patch.object(sys,'argv',['counter']), \
             patch.object(app.cv2,'VideoCapture',return_value=capture), \
             patch.object(app.PoseLandmarker,'create_from_options',return_value=detector), \
             patch.object(app.cv2,'imshow'), \
             patch.object(app.cv2,'waitKey',side_effect=keys+[0]), \
             patch.object(app.cv2,'getWindowProperty',return_value=1), \
             patch.object(app.cv2,'destroyAllWindows'):
            app.main()
        self.assertTrue(capture.release.called)
        self.assertTrue(detector.close.called)

    def test_quitting_preview_does_not_create_empty_recordings(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_camera(folder,[ord('q')])
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_space_starts_recording_and_quit_saves_only_its_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_camera(folder,[32,ord('q')])
            files=list(Path(folder).glob('reps_*.csv'))
            self.assertEqual(len(files),1)
            rows=list(csv.DictReader(files[0].read_text().splitlines()))
            self.assertEqual(len(rows),1)
            self.assertEqual(float(rows[0]['time_s']),0)
            self.assertTrue(files[0].with_suffix('.json').exists())
            self.assertTrue(files[0].with_suffix('.png').exists())
