"""
Download the MediaPipe model files used in Part 2 of the lessons.

MediaPipe doesn't ship its AI models inside the Python package. Each "task"
(hand landmarks, gestures, body pose) needs a small model file ending in
.task. This script downloads all of them into the models/ folder.

How to run (from the repository folder, with your virtual environment active):

    python download_models.py

You only need to do this once. Files that already exist are skipped.
"""

import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Folder next to this script, so it works no matter where you run it from.
MODELS_DIR = Path(__file__).resolve().parent / "models"

# file name -> download URL (from Google's MediaPipe documentation)
MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    "gesture_recognizer.task":
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/latest/gesture_recognizer.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def download(name, url):
    """Download one model file into MODELS_DIR.

    Args:
        name: File name to save as, e.g. "hand_landmarker.task".
        url:  Where to download it from.

    Returns:
        True if the file is now present, False if the download failed.
    """
    target = MODELS_DIR / name
    if target.exists():
        print(f"  [skip] {name} already downloaded")
        return True

    print(f"  [....] downloading {name}")
    try:
        urllib.request.urlretrieve(url, target)
    except urllib.error.URLError as error:
        # A common problem on macOS: Python from python.org can't check
        # HTTPS certificates until you run its "Install Certificates" script.
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            print("  [fail] SSL certificate error. On a Mac, open Finder ->")
            print("         Applications -> Python 3.x and double-click")
            print("         'Install Certificates.command', then run this again.")
        else:
            print(f"  [fail] {name}: {error.reason}")
        target.unlink(missing_ok=True)   # don't leave a half-downloaded file
        return False

    size_mb = target.stat().st_size / 1_000_000
    print(f"  [done] {name} ({size_mb:.1f} MB)")
    return True


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Saving models to {MODELS_DIR}")
    results = [download(name, url) for name, url in MODELS.items()]
    if all(results):
        print("All models ready.")
    else:
        print("Some downloads failed. Fix the problem above and run this script again.")
        sys.exit(1)


if __name__ == "__main__":
    main()