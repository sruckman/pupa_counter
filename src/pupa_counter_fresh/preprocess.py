"""Preprocessing for the fresh peak-first detector.

Stage A of the pipeline:

1. Load image as RGB uint8
2. Optional fast downscale
3. Blue ink annotation mask (user wrote counts on the paper in blue pen; those
   strokes must never be counted as pupae)

The blue-mask implementation here is a simplified, self-contained version of
the logic inside ``pupa_counter.preprocess.blue_mask``. It is deliberately not
imported so the fresh package stays decoupled from the frozen legacy code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image_rgb(path: Path | str) -> np.ndarray:
    """Read an image from disk and return an ``HxWx3`` uint8 RGB array."""
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def downscale(rgb: np.ndarray, scale: float) -> Tuple[np.ndarray, float]:
    """Return a downscaled copy of ``rgb`` and the actual scale factor used.

    ``scale`` is clipped into ``[0.25, 1.0]`` to keep the detector sane.
    If ``scale`` is within a few percent of 1.0 the original image is returned
    untouched so we don't pay the resize cost on full-resolution runs.
    """
    scale = float(max(0.25, min(1.0, scale)))
    if abs(scale - 1.0) < 1e-3:
        return rgb, 1.0
    h, w = rgb.shape[:2]
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    actual_scale = new_size[0] / float(w)
    return resized, actual_scale


# ---------------------------------------------------------------------------
# Blue annotation mask
# ---------------------------------------------------------------------------


def build_blue_mask(rgb: np.ndarray) -> np.ndarray:
    """Return an ``HxW`` uint8 mask (0/255) of blue annotation ink.

    Combines three permissive checks — HSV band, LAB b* negativity, RGB blue
    dominance with minimum saturation — then applies a light morphological
    open/close so thin pen strokes come through cleanly.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    hsv_lower = np.array((90, 40, 40), dtype=np.uint8)
    hsv_upper = np.array((140, 255, 255), dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    b_channel = rgb[:, :, 2].astype(np.int16)
    g_channel = rgb[:, :, 1].astype(np.int16)
    r_channel = rgb[:, :, 0].astype(np.int16)
    rgb_mask = (
        (b_channel > g_channel + 15)
        & (b_channel > r_channel + 15)
        & (hsv[:, :, 1] >= 30)
    ).astype(np.uint8) * 255

    lab_mask = ((lab[:, :, 2] <= 115) & (hsv[:, :, 1] >= 30)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(hsv_mask, cv2.bitwise_or(rgb_mask, lab_mask))

    open_kernel = np.ones((3, 3), dtype=np.uint8)
    close_kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask
