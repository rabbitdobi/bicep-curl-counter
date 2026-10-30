"""CSV/plot checks: resets, missing values, bad input and exported PNG."""
import csv
import json
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTTER = Path(__file__).resolve().parents[1] / "assignment" / "plot_reps.py"


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PLOTTER.exists(), "Assignment plot_reps.py is missing")
        spec = importlib.util.spec_from_file_location("plot_reps", PLOTTER)
        self.plotter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.plotter)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "reps.csv"
        self.addCleanup(plt.close, "all")

    def write_rows(self, rows, header=("time_s", "angle_deg", "stage", "reps")):
        with self.path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

    def test_missing_angle_leaves_gap_and_reset_allows_new_markers(self):
        self.write_rows([(0, 170, "down", 0), (1, "", "down", 0),
                         (2, 40, "up", 0), (3, 170, "down", 1),
                         (4, 170, "down", 0), (5, 40, "up", 0),
                         (6, 170, "down", 1)])
        data = self.plotter.load_csv(self.path)
        self.assertTrue(math.isnan(data["angle_deg"][1]))
        self.assertEqual(self.plotter.completed_reps(data), [3.0, 6.0])
        figure = self.plotter.make_plot(data, "test")
        axis = figure.axes[0]
        self.assertEqual(axis.get_xlabel(), "Time (s)")
        self.assertEqual(list(axis.lines[0].get_xdata()), [0, 1, 2, 3, 4, 5, 6])
        verticals = [list(line.get_xdata()) for line in axis.lines[3:]]
        self.assertEqual(verticals, [[3, 3], [6, 6]])

    def test_bad_or_empty_csv_has_actionable_error(self):
        cases = [([], ("time_s", "angle_deg", "stage", "reps")),
                 ([(0, 170)], ("time_s", "angle_deg")),
                 ([(0, "bad", "down", 0)], ("time_s", "angle_deg", "stage", "reps")),
                 ([(1, 170, "down", 0), (0, 40, "up", 0)],
                  ("time_s", "angle_deg", "stage", "reps")),
                 ([(0, 190, "down", 0)], ("time_s", "angle_deg", "stage", "reps")),
                 ([(0, 170, "down", -1)], ("time_s", "angle_deg", "stage", "reps")),
                 ([(0, 170, "bad", 0)], ("time_s", "angle_deg", "stage", "reps"))]
        for rows, header in cases:
            with self.subTest(rows=rows, header=header):
                self.write_rows(rows, header)
                with self.assertRaises(ValueError):
                    self.plotter.load_csv(self.path)

    def test_initial_tracking_gap_remains_visible_on_time_axis(self):
        self.write_rows([(0, "", "down", 0), (1, "", "down", 0),
                         (10, 170, "down", 0), (12, 40, "up", 0)])
        figure = self.plotter.make_plot(self.plotter.load_csv(self.path), "gap")
        self.assertEqual(figure.axes[0].get_xlim(), (0, 12))

    def test_plot_uses_saved_thresholds_unless_explicitly_overridden(self):
        self.path.with_suffix('.json').write_text(json.dumps({'up_angle':60,'down_angle':145}))
        self.assertEqual(self.plotter.recording_thresholds(self.path),(60,145))
        self.assertEqual(self.plotter.recording_thresholds(self.path,55,155),(55,155))

    def test_bad_recording_settings_fail_instead_of_drawing_wrong_lines(self):
        for content in ('{broken', '[]', '{"up_angle":150,"down_angle":50}'):
            self.path.with_suffix('.json').write_text(content)
            with self.assertRaises(ValueError):
                self.plotter.recording_thresholds(self.path)

    def test_command_exports_png_without_camera_or_window(self):
        self.write_rows([(0, 170, "down", 0), (1, 40, "up", 0), (2, 170, "down", 1)])
        result = subprocess.run([sys.executable, str(PLOTTER), str(self.path), "--no-show"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        png = self.path.with_suffix(".png")
        self.assertTrue(png.exists())
        self.assertGreater(png.stat().st_size, 10000)
        self.assertIn("Completed reps: 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
