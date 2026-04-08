"""Blue annotation masking."""

from __future__ import annotations

import cv2
import numpy as np

from pupa_counter.config import AppConfig


def detect_blue_annotations(image: np.ndarray, cfg: AppConfig = None) -> np.ndarray:
    cfg = cfg or AppConfig()
    if not cfg.blue_mask.enabled:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lower = np.array(cfg.blue_mask.hsv_lower_1, dtype=np.uint8)
    upper = np.array(cfg.blue_mask.hsv_upper_1, dtype=np.uint8)

    hsv_mask = cv2.inRange(hsv, lower, upper)

    # Permissive LAB+RGB path. Originally any near-neutral pixel with B
    # marginally above R/G could trip this — fine for crisp PNG annotations
    # but on grayscale PDF scans it fires on dark dust/noise and ends up
    # erasing real brown pupae downstream. Tighten the gap requirement and
    # demand a minimum saturation so the path only catches genuine blue ink.
    lab_mask = (lab[:, :, 2] <= cfg.blue_mask.lab_b_max).astype(np.uint8) * 255
    rgb_mask = (
        (image[:, :, 2].astype(np.int16) > image[:, :, 1].astype(np.int16) + 15)
        & (image[:, :, 2].astype(np.int16) > image[:, :, 0].astype(np.int16) + 15)
    ).astype(np.uint8) * 255
    sat_mask = (hsv[:, :, 1] >= 30).astype(np.uint8) * 255
    permissive_mask = cv2.bitwise_and(cv2.bitwise_and(lab_mask, rgb_mask), sat_mask)
    mask = cv2.bitwise_or(hsv_mask, permissive_mask)

    open_kernel = np.ones(
        (cfg.blue_mask.morphology_open_kernel, cfg.blue_mask.morphology_open_kernel), dtype=np.uint8
    )
    close_kernel = np.ones(
        (cfg.blue_mask.morphology_close_kernel, cfg.blue_mask.morphology_close_kernel), dtype=np.uint8
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask
