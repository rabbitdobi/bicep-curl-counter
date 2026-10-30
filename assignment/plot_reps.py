"""Plot a Rep Counter CSV, with a vertical marker for each completed curl.

Run from the repository folder:
    python assignment/plot_reps.py                         # newest reps CSV
    python assignment/plot_reps.py output/reps_DATE.csv --no-show

A missing angle is plotted as a gap, rather than as zero degrees.
R resets are supported: each increase relative to the previous row is a rep.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def load_csv(path):
    """Read and validate the four columns produced by RepLogger."""
    data = {name: [] for name in ("time_s", "angle_deg", "stage", "reps")}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not set(data).issubset(reader.fieldnames):
            raise ValueError("CSV needs columns: time_s, angle_deg, stage, reps.")
        for row_number, row in enumerate(reader, start=2):
            try:
                seconds = float(row["time_s"])
                angle = float(row["angle_deg"]) if row["angle_deg"].strip() else math.nan
                reps = int(row["reps"])
                stage = row["stage"]
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("time must be finite and nonnegative")
                if data["time_s"] and seconds < data["time_s"][-1]:
                    raise ValueError("time must not run backwards")
                if row["angle_deg"].strip() and (not math.isfinite(angle) or not 0 <= angle <= 180):
                    raise ValueError("angle must be blank or between 0 and 180")
                if reps < 0 or stage not in ("up", "down"):
                    raise ValueError("reps must be nonnegative and stage must be up/down")
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid CSV row {row_number}: {error}") from error
            for key, value in zip(data, (seconds, angle, stage, reps)):
                data[key].append(value)
    if not data["time_s"]:
        raise ValueError("CSV has no frames. Run the counter and record some curls first.")
    return data


def completed_reps(data):
    """Find increases in the recorded count, including after a reset."""
    times = []
    previous = 0
    for seconds, reps in zip(data["time_s"], data["reps"]):
        if reps > previous:
            times.append(seconds)
        previous = reps
    return times


def recording_thresholds(path, up_angle=None, down_angle=None):
    """Use explicit options, then saved session settings, then legacy defaults."""
    settings_path = Path(path).with_suffix(".json")
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object")
        up = float(up_angle if up_angle is not None else settings.get("up_angle", 50))
        down = float(down_angle if down_angle is not None else settings.get("down_angle", 150))
        if not 0 < up <= down < 180:
            raise ValueError("thresholds must satisfy 0 < up <= down < 180")
        return up, down
    except (ValueError, TypeError, OSError) as error:
        raise ValueError(f"Invalid recording settings in {settings_path.name}: {error}") from error


def make_plot(data, title, up_angle=50, down_angle=150):
    """Return a figure; the caller chooses whether to show or save it."""
    figure, axis = plt.subplots(figsize=(11, 5.5))
    figure.set_facecolor("#f7f9fc")
    axis.plot(data["time_s"], data["angle_deg"], color="#2563eb", linewidth=1.8,
              label="Elbow angle")
    axis.axhline(up_angle, color="#d97706", linestyle="--", linewidth=1,
                 label=f"Curl threshold ({up_angle:g}°)")
    axis.axhline(down_angle, color="#7c3aed", linestyle="--", linewidth=1,
                 label=f"Extension threshold ({down_angle:g}°)")
    markers = completed_reps(data)
    for index, seconds in enumerate(markers, start=1):
        axis.axvline(seconds, color="#059669", alpha=0.7, linewidth=1.3,
                     label="Completed rep" if index == 1 else None)
        axis.text(seconds, 183, str(index), ha="center", color="#047857", fontsize=9)
    axis.set(xlabel="Time (s)", ylabel="Elbow angle (degrees)", ylim=(0, 190),
             title=f"{title}\nCompleted reps: {len(markers)}")
    if data["time_s"][-1] > 0:
        axis.set_xlim(0, data["time_s"][-1])
    axis.grid(alpha=0.18)
    axis.legend(loc="lower left", fontsize=9, ncol=2)
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path, help="CSV to plot; defaults to newest output/reps_*.csv")
    parser.add_argument("--no-show", action="store_true", help="Save PNG without opening a plot window")
    parser.add_argument("--up-angle", type=float, help="Override saved curl threshold (legacy default: 50)")
    parser.add_argument("--down-angle", type=float, help="Override saved extension threshold (legacy default: 150)")
    args = parser.parse_args()
    path = args.csv
    if path is None:
        files = list(OUTPUT_DIR.glob("reps_*.csv"))
        if not files:
            parser.error("No reps CSV found. Run assignment/assignment_starter.py first.")
        path = max(files, key=lambda candidate: candidate.stat().st_mtime)
    try:
        data = load_csv(path)
        up_angle, down_angle = recording_thresholds(path, args.up_angle, args.down_angle)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    figure = make_plot(data, path.name, up_angle, down_angle)
    png = path.with_suffix(".png")
    figure.savefig(png, dpi=160)
    print(f"Frames: {len(data['time_s'])}; duration: {data['time_s'][-1]:.2f} s")
    print(f"Completed reps: {len(completed_reps(data))}; final counter: {data['reps'][-1]}")
    print(f"Saved {png}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
