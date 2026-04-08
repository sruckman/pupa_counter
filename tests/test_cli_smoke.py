from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pupa_counter.cli import main


def test_cli_smoke_run(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = np.full((240, 140, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (50, 50), (9, 4), 15, 0, 360, (170, 120, 70), -1)
    cv2.ellipse(image, (70, 100), (9, 4), 0, 0, 360, (175, 125, 80), -1)
    cv2.ellipse(image, (80, 180), (9, 4), -12, 0, 360, (180, 128, 82), -1)
    cv2.line(image, (5, 20), (135, 20), (20, 120, 255), 2)
    image_path = input_dir / "sample.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    output_root = tmp_path / "runs"
    exit_code = main(
        [
            "run",
            "--input-root",
            str(input_dir),
            "--config",
            str(Path("configs/base.yaml").resolve()),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    counts_path = output_root / "baseline_v1" / "counts.csv"
    assert counts_path.exists()
