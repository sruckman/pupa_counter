"""Brown / dark response map for the fresh peak-first detector.

Stage A continued:

* ``compute_response_map`` — a scalar in ``[0, 1]`` that separates pupa-like
  pixels from paper well enough on its own (the fresh-start pixel separability
  probe reported ROC AUC ≈ 0.946 for a similar brown score).
* ``build_allowed_mask`` — a permissive foreground mask derived from the
  response map. This is *not* the final foreground segmentation. It is only a
  gate that tells the peak stage where to look.

Response modes
==============

The v1 pipeline used a single Gaussian smoothing (``smooth`` mode) which
silently merges touching-pupae peaks. v2 adds three additional modes that
preserve two distinct peaks when two pupae touch:

* ``smooth``   — the v1 default, plain Gaussian blur with ``smooth_sigma``.
* ``log``      — single-sigma Difference-of-Gaussians approximation to the
  Laplacian-of-Gaussian. Computes ``Gauss(σ) − Gauss(2σ)``, re-clips to ≥0,
  and renormalizes. Preserves bimodality of touching pairs.
* ``dog``      — traditional Difference-of-Gaussians with independent
  ``σ_low`` and ``σ_high`` parameters. Same idea as ``log`` but gives the
  sweep loop two knobs to tune independently.
* ``adaptive`` — per-component sigma selection. Builds a temporary
  permissive mask from the raw response, labels connected components, and
  applies ``adaptive_small_sigma`` to components smaller than
  ``adaptive_area_threshold_px`` and ``adaptive_large_sigma`` to larger
  components. Background is smoothed with the large sigma. The idea is
  that small components are "probably single pupae" and can tolerate a
  sharper sigma without breaking into false peaks, while large merged
  clusters get the aggressive sharpening.

In every mode the blue-ink mask is applied **after** the sharpening step so
LoG/DoG ringing inside a blue stroke cannot leak into neighboring real
regions. Negative LoG/DoG output is clipped to zero and the result is
renormalized into ``[0, 1]``.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from scipy import ndimage as ndi


ResponseMode = Literal["smooth", "log", "dog", "adaptive"]


# ---------------------------------------------------------------------------
# Raw brown/dark score (unchanged from v1)
# ---------------------------------------------------------------------------


def _compute_raw_brown_score(
    rgb: np.ndarray,
    *,
    saturation_floor: float = 0.0,
) -> np.ndarray:
    """Return the raw weighted brown/dark scalar in ``[0, 1]``.

    Weights are a simplified rebalance of the legacy ``compute_brown_score``.
    Dominant contributors:

    * ``darkness`` (``1 - V``) — pupae are darker than paper
    * ``saturation`` — warm brown pupae have higher HSV S than cleaned paper
    * ``lab_a`` — positive a* corresponds to the red/brown axis

    ``saturation_floor`` is a hard gate added 2026-04-11: any pixel with
    HSV saturation below the floor gets zero response, regardless of
    darkness / brown score. This kills shadows, dirt, and blank-area
    false positives that have a dark-but-uncolored appearance. Real
    pupae have saturation 0.20-0.45; shadows and dirt have <0.10.
    Default 0.0 preserves the legacy behavior; set to 0.08-0.12 for
    the fix. See docs/DENSE_CLUSTER_RESEARCH_2026-04-11.md for evidence.
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

    if saturation_floor > 0:
        response = np.where(saturation >= float(saturation_floor), response, 0.0)

    # Brightness floor: zero out scanner-bar and other very dark pixels.
    # Scanner bar: V ≈ 0.05 (pitch black). Pupae: V ≈ 0.40-0.70.
    # Without this, the scanner bar's darkness score (0.35 * 1.0 = 0.35)
    # exceeds the allowed mask threshold (0.12), merging bar pixels with
    # adjacent pupa components and causing peak detection to waste slots
    # on bar peaks that get edge-killed.
    brightness_floor = 0.15
    response = np.where(value >= brightness_floor, response, 0.0)

    return response.astype(np.float32)


# ---------------------------------------------------------------------------
# Gaussian helper — consistent kernel sizing for every mode
# ---------------------------------------------------------------------------


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float32, copy=True)
    ksize = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(image, (ksize, ksize), float(sigma))


def _normalize_clip(image: np.ndarray) -> np.ndarray:
    """Clip negatives to zero and rescale the positive support to ``[0, 1]``.

    DoG / LoG outputs are naturally zero-centered; we only care about the
    positive support (bright blob centers), so we clip and renormalize.
    """
    positive = np.clip(image, 0.0, None)
    peak = float(positive.max())
    if peak <= 1e-6:
        return positive.astype(np.float32)
    return (positive / peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Response-mode dispatchers
# ---------------------------------------------------------------------------


def _smooth_mode(raw: np.ndarray, smooth_sigma: float) -> np.ndarray:
    return _gaussian(raw, smooth_sigma)


def _log_mode(raw: np.ndarray, log_sigma: float) -> np.ndarray:
    """``Gauss(σ) − Gauss(2σ)``, re-clipped and renormalized.

    This is the classic LoG ≈ DoG approximation with a 1:2 sigma ratio.
    The ratio 1:1.6 is more "pure" but 1:2 is standard in the literature
    and gives a stronger sharpening effect, which is what we want for the
    touching-pair problem.
    """
    low = _gaussian(raw, log_sigma)
    high = _gaussian(raw, 2.0 * log_sigma)
    return _normalize_clip(low - high)


def _dog_mode(raw: np.ndarray, sigma_low: float, sigma_high: float) -> np.ndarray:
    """Difference-of-Gaussians with independent low/high sigmas."""
    if sigma_high <= sigma_low:
        # Caller misconfigured; degenerate to smooth at sigma_low so the
        # pipeline keeps producing a usable output instead of zeros.
        return _gaussian(raw, sigma_low)
    low = _gaussian(raw, sigma_low)
    high = _gaussian(raw, sigma_high)
    return _normalize_clip(low - high)


def _adaptive_mode(
    raw: np.ndarray,
    *,
    small_sigma: float,
    large_sigma: float,
    area_threshold_px: int,
    label_threshold: float = 0.10,
) -> np.ndarray:
    """Per-component sigma selection.

    Step 1: Build a permissive mask from the raw response.
    Step 2: Label connected components.
    Step 3: For each component, decide which sigma to use based on area.
    Step 4: Composite the small-sigma and large-sigma blurs by component.
    """
    small_blur = _gaussian(raw, small_sigma)
    large_blur = _gaussian(raw, large_sigma)

    gate = raw >= float(label_threshold)
    if not gate.any():
        # No foreground at all; fall back to global large-sigma smoothing
        return large_blur

    labels, n_components = ndi.label(gate)
    if n_components == 0:
        return large_blur

    sizes = np.bincount(labels.ravel())
    small_component_ids = np.where(sizes < int(area_threshold_px))[0]
    # Exclude the background label (0)
    small_component_ids = small_component_ids[small_component_ids != 0]

    use_small = np.isin(labels, small_component_ids)
    composite = np.where(use_small, small_blur, large_blur)
    return composite.astype(np.float32)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_response_map(
    rgb: np.ndarray,
    blue_mask: np.ndarray | None = None,
    *,
    smooth_sigma: float = 1.2,
    response_mode: ResponseMode = "smooth",
    log_sigma: float = 1.0,
    dog_sigma_low: float = 0.8,
    dog_sigma_high: float = 2.0,
    adaptive_small_sigma: float = 0.6,
    adaptive_large_sigma: float = 1.4,
    adaptive_area_threshold_px: int = 500,
    saturation_floor: float = 0.0,
) -> np.ndarray:
    """Return an ``HxW`` float32 scalar response in ``[0, 1]``.

    ``response_mode`` selects one of four computation strategies — see the
    module docstring for details. When omitted, the function is
    bit-for-bit compatible with the v1 ``smooth`` behavior so existing
    runs remain reproducible.
    """
    raw = _compute_raw_brown_score(rgb, saturation_floor=saturation_floor)

    if response_mode == "smooth":
        result = _smooth_mode(raw, smooth_sigma)
    elif response_mode == "log":
        result = _log_mode(raw, log_sigma)
    elif response_mode == "dog":
        result = _dog_mode(raw, dog_sigma_low, dog_sigma_high)
    elif response_mode == "adaptive":
        result = _adaptive_mode(
            raw,
            small_sigma=adaptive_small_sigma,
            large_sigma=adaptive_large_sigma,
            area_threshold_px=adaptive_area_threshold_px,
        )
    else:
        raise ValueError(f"unknown response_mode: {response_mode!r}")

    # Apply blue mask AFTER sharpening so LoG / DoG ringing inside a blue
    # stroke cannot leak into neighboring real regions. Negative ringing is
    # already clipped to zero by ``_normalize_clip``.
    if blue_mask is not None and blue_mask.size:
        result = result.copy()
        result[blue_mask > 0] = 0.0

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Allowed mask (unchanged)
# ---------------------------------------------------------------------------


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
