#!/usr/bin/env python3
"""Simple webcam capture loop for future live-counting experiments.

This is intentionally lightweight:
- shows a live webcam preview
- draws a guide ROI where the paper should be placed
- captures frames on a timer and/or when the user presses space

It does not yet run the pupa pipeline automatically; it is a local capture
prototype to help converge on the real lab workflow before building a full
live-counting mode.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam capture prototype for pupa counting")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", default="data/webcam_captures")
    parser.add_argument("--interval-s", type=float, default=0.0, help="Auto-capture interval; 0 disables timer")
    parser.add_argument("--roi-width-frac", type=float, default=0.72)
    parser.add_argument("--roi-height-frac", type=float, default=0.72)
    parser.add_argument("--prefix", default="webcam")
    return parser.parse_args()


def centered_roi(width: int, height: int, width_frac: float, height_frac: float) -> tuple[int, int, int, int]:
    roi_width = int(round(width * width_frac))
    roi_height = int(round(height * height_frac))
    x0 = max(0, (width - roi_width) // 2)
    y0 = max(0, (height - roi_height) // 2)
    return x0, y0, x0 + roi_width, y0 + roi_height


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.camera_index}")

    last_capture = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            x0, y0, x1, y1 = centered_roi(width, height, args.roi_width_frac, args.roi_height_frac)

            preview = frame.copy()
            cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 200, 255), 2)
            cv2.putText(
                preview,
                "Place paper inside box | SPACE=capture | q=quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            now = time.time()
            if args.interval_s > 0 and now - last_capture >= args.interval_s:
                last_capture = now
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                capture_path = output_dir / f"{args.prefix}_{timestamp}.png"
                cv2.imwrite(str(capture_path), frame[y0:y1, x0:x1])
                print(f"captured={capture_path}")

            cv2.imshow("pupa-counter webcam prototype", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                capture_path = output_dir / f"{args.prefix}_{timestamp}.png"
                cv2.imwrite(str(capture_path), frame[y0:y1, x0:x1])
                print(f"captured={capture_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
