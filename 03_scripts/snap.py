#!/usr/bin/env python3
import cv2
from datetime import datetime
from pathlib import Path

CAM_INDEX = 1  # same index matcher.py uses
OUT_DIR = Path(__file__).resolve().parent.parent / "04_reports"
OUT_DIR.mkdir(exist_ok=True)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {CAM_INDEX}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from camera")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"debug_frame_{ts}.png"
    cv2.imwrite(str(path), frame)
    print(f"Saved snapshot to {path}")

if __name__ == "__main__":
    main()