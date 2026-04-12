"""Regression tests for the brown candidate detector, including the
grayscale auto-mode added for clean / unannotated PDF inputs."""

from __future__ import annotations

import cv2
import numpy as np

from pupa_counter.config import AppConfig
from pupa_counter.detect.brown_mask import (
    compute_brown_score,
    detect_brown_candidates,
    is_grayscale_image,
)


def _white_canvas(height: int = 240, width: int = 200) -> np.ndarray:
    return np.full((height, width, 3), 245, dtype=np.uint8)


def _draw_brown_pupa(image: np.ndarray, center, axes=(10, 5)) -> None:
    cv2.ellipse(image, center, axes, 25, 0, 360, (90, 55, 35), -1)


def _draw_dark_pupa(image: np.ndarray, center, axes=(10, 5)) -> None:
    cv2.ellipse(image, center, axes, 25, 0, 360, (40, 40, 40), -1)


def test_grayscale_detection_fires_on_desaturated_input():
    image = _white_canvas()
    for cy in (60, 120, 180):
        _draw_dark_pupa(image, (100, cy))

    assert is_grayscale_image(image, AppConfig())


def test_color_path_used_for_brown_image():
    image = _white_canvas()
    for cy in (60, 120, 180):
        _draw_brown_pupa(image, (100, cy))

    assert not is_grayscale_image(image, AppConfig())


def test_grayscale_path_detects_dark_pupae_color_path_misses():
    image = _white_canvas()
    centers = [(60, 60), (140, 60), (60, 180), (140, 180)]
    for center in centers:
        _draw_dark_pupa(image, center)

    cfg_grayscale_on = AppConfig()
    cfg_grayscale_off = AppConfig()
    cfg_grayscale_off.brown_detection.auto_grayscale_mode = False

    mask_with_grayscale = detect_brown_candidates(image, cfg=cfg_grayscale_on)
    mask_color_only = detect_brown_candidates(image, cfg=cfg_grayscale_off)

    # Sample one pixel inside each pupa and one in the background.
    for cx, cy in centers:
        assert mask_with_grayscale[cy, cx] > 0, f"grayscale path missed pupa at {(cx, cy)}"
    assert mask_with_grayscale[10, 10] == 0  # background stays clean

    # The legacy color-only path should miss the desaturated pupae entirely.
    color_hits = sum(1 for cx, cy in centers if mask_color_only[cy, cx] > 0)
    assert color_hits == 0


def test_brown_score_still_passes_threshold_for_dark_pupa_pixel():
    """Component-level color_score uses compute_brown_score under the hood.
    A near-black pupa pixel must score above the rule_filter min_color_score
    threshold (0.20) so the rule classifier still labels grayscale pupae as
    'pupa', not 'uncertain'."""
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    score = float(compute_brown_score(image)[0, 0])
    assert score >= 0.20

    medium_dark = np.zeros((1, 1, 3), dtype=np.uint8)
    medium_dark[:] = (80, 80, 80)
    medium_score = float(compute_brown_score(medium_dark)[0, 0])
    assert medium_score >= 0.18  # passes brown_score_threshold for detection
