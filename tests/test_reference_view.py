from __future__ import annotations

import numpy as np

from pupa_counter.config import AppConfig
from pupa_counter.preprocess.normalize import build_reference_view, normalize_background


def test_reference_view_stays_closer_to_original_than_full_normalization():
    image = np.full((32, 32, 3), 248, dtype=np.uint8)
    image[8:24, 10:22, :] = [170, 110, 90]

    cfg = AppConfig()
    reference = build_reference_view(image, cfg)
    normalized = normalize_background(image, cfg)

    original_dark_mean = float(image[8:24, 10:22, :].mean())
    reference_dark_mean = float(reference[8:24, 10:22, :].mean())
    normalized_dark_mean = float(normalized[8:24, 10:22, :].mean())

    assert abs(reference_dark_mean - original_dark_mean) < abs(normalized_dark_mean - original_dark_mean)
    assert reference.shape == image.shape
