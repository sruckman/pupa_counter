"""Brown / dark response map for the fresh peak-first detector.

Stage A continued:

* ``compute_response_map`` — a scalar in ``[0, 1]`` that separates pupa-like
  pixels from paper well enough on its own (the fresh-start pixel separability
  probe reported ROC AUC ≈ 0.946 for a similar brown score).
* ``build_allowed_mask`` — a permissive foreground mask derived from the
  response map. This is *not* the final foreground segmentation. It is only a
  gate that tells the peak stage where to look.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_response_map(
    rgb: np.ndarray,
    blue_mask: np.ndarray | None = None,
    *,
    smooth_sigma: float = 1.2,
) -> np.ndarray:
    """Return a ``HxW`` float32 scalar response in ``[0, 1]``.

    Weights are a simplified rebalance of the legacy ``compute_brown_score``.
    Empirically the dominant contributors are:

    * ``darkness`` (``1 - V``) — pupae are darker than paper
    * ``saturation`` — warm brown pupae have higher HSV S than cleaned paper
    * ``lab_a`` — positive a* corresponds to the red/brown axis
    """
    image_float = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    darkness = np.clip((0.92 - value) / 0.92, 0.0, 1.0)

    lab_a = np.clip((lab[:, :, 1] - 128.0) / 60.0, 0.0, 1.0)
    lab_b = np.clip((lab[:, :, 2] - 118.0) / 80.0, 0.0, 1.0)
    red_minus_blue = np.clip(
        (image_float[:, :, 0] - image_float[:, :, 2]) / 0.55, 0.0, 1.0
    )

    response = (
        0.35 * darkness
        + 0.25 * saturation
        + 0.18 * lab_a
        + 0.12 * lab_b
        + 0.10 * red_minus_blue
    )
    # Weights sum to 1.0, so ``response`` is already in ``[0, 1]``.

    if blue_mask is not None and blue_mask.size:
        response = response.copy()
        response[blue_mask > 0] = 0.0

    if smooth_sigma > 0:
        ksize = max(3, int(round(smooth_sigma * 6)) | 1)
        response = cv2.GaussianBlur(response, (ksize, ksize), smooth_sigma)

    return response.astype(np.float32)


def build_allowed_mask(
    response: np.ndarray,
    *,
    abs_threshold: float = 0.28,
    min_percentile: float = 70.0,
) -> np.ndarray:
    """Return an ``HxW`` uint8 mask (0/255) of plausibly-foreground pixels.

    The mask is the *intersection* of two conditions:

    1. ``response >= abs_threshold`` — cuts uniformly-paper pixels on every
       scan.
    2. ``response >= percentile(response, min_percentile)`` — extra safety for
       unusually dark/colorful scans where ``abs_threshold`` would let
       paper through.

    Taking the larger of the two thresholds is what we want: both conditions
    must hold, and the *stricter* one governs. (The v0 smoke test caught the
    opposite polarity bug — using ``min`` let through ~97% of the image.)
    """
    percentile_value = float(np.percentile(response, min_percentile))
    threshold = max(float(abs_threshold), percentile_value)

    mask = (response >= threshold).astype(np.uint8) * 255

    # A single light open removes isolated stray pixels without swallowing
    # narrow valleys between touching pupae. Connected-components closing is
    # intentionally omitted — we want the mask permissive.
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return mask
