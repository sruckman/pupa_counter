from __future__ import annotations

import cv2
import numpy as np

from pupa_counter.config import AppConfig
from pupa_counter.preprocess.blue_mask import detect_blue_annotations


def test_blue_mask_hits_blue_and_not_brown():
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    cv2.line(image, (5, 10), (70, 10), (20, 120, 255), 4)
    cv2.ellipse(image, (40, 50), (8, 4), 20, 0, 360, (170, 120, 70), -1)

    mask = detect_blue_annotations(image, AppConfig())
    assert mask[10, 20] > 0
    assert mask[50, 40] == 0
