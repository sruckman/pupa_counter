"""Blue annotation removal strategies."""

from __future__ import annotations

import cv2
import numpy as np

from pupa_counter.config import AppConfig


def remove_or_ignore_blue(image: np.ndarray, blue_mask: np.ndarray, cfg: AppConfig = None) -> np.ndarray:
    cfg = cfg or AppConfig()
    if blue_mask is None or blue_mask.size == 0:
        return image
    if cfg.blue_mask.remove_mode == "inpaint":
        return cv2.inpaint(image, blue_mask, cfg.blue_mask.inpaint_radius, cv2.INPAINT_TELEA)
    cleaned = image.copy()
    cleaned[blue_mask > 0] = 255
    return cleaned
