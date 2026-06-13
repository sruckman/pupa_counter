"""Estimate the white-paper working region.

Annotated scans usually show a mostly white sheet with residual dark scanner
bars at one or more edges. We only want to count pupae that lie on the white
paper itself, not fragments inside those dark margins.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from pupa_counter.config import AppConfig


PaperBounds = Tuple[int, int, int, int]


def _smooth_ratio(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(values) <= 2:
        return values.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def estimate_paper_bounds(
    image: np.ndarray,
    *,
    blue_mask: np.ndarray | None = None,
    cfg: AppConfig | None = None,
) -> PaperBounds:
    """Return `(left, top, right, bottom)` bounds for the white paper area."""
    cfg = cfg or AppConfig()
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return (0, 0, width, height)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white_mask = (
        (hsv[:, :, 2] >= cfg.preprocess.paper_white_min_value)
        & (hsv[:, :, 1] <= cfg.preprocess.paper_white_max_saturation)
    )
    if blue_mask is not None and blue_mask.size:
        white_mask &= blue_mask == 0

    raw_row_ratio = white_mask.mean(axis=1)
    raw_col_ratio = white_mask.mean(axis=0)
    row_ratio = _smooth_ratio(raw_row_ratio, cfg.preprocess.paper_smooth_window_px)
    col_ratio = _smooth_ratio(raw_col_ratio, cfg.preprocess.paper_smooth_window_px)
    threshold = float(cfg.preprocess.paper_support_ratio_threshold)
    col_threshold = max(
        threshold,
        float(np.quantile(raw_col_ratio, cfg.preprocess.paper_dynamic_col_quantile))
        * float(cfg.preprocess.paper_dynamic_col_scale),
    )

    row_idx = np.flatnonzero((row_ratio >= threshold) & (raw_row_ratio >= threshold * 0.35))
    col_idx = np.flatnonzero((col_ratio >= col_threshold) & (raw_col_ratio >= col_threshold * 0.55))
    if row_idx.size == 0 or col_idx.size == 0:
        return (0, 0, width, height)

    pad = int(cfg.preprocess.paper_bbox_padding_px)
    top = max(0, int(row_idx[0]) - pad)
    bottom = min(height, int(row_idx[-1]) + pad + 1)
    left = max(0, int(col_idx[0]) - pad)
    right = min(width, int(col_idx[-1]) + pad + 1)
    return (left, top, right, bottom)


def centroid_inside_paper_bounds(
    centroid_x: float,
    centroid_y: float,
    bounds: PaperBounds | None,
) -> bool:
    if bounds is None:
        return True
    left, top, right, bottom = bounds
    return (left <= float(centroid_x) <= right) and (top <= float(centroid_y) <= bottom)


def bbox_fraction_inside_paper_bounds(
    bbox_x0: float,
    bbox_y0: float,
    bbox_x1: float,
    bbox_y1: float,
    bounds: PaperBounds | None,
) -> float:
    if bounds is None:
        return 1.0
    left, top, right, bottom = bounds
    bbox_w = max(0.0, float(bbox_x1) - float(bbox_x0))
    bbox_h = max(0.0, float(bbox_y1) - float(bbox_y0))
    bbox_area = bbox_w * bbox_h
    if bbox_area <= 0.0:
        return 0.0
    inter_x0 = max(float(bbox_x0), float(left))
    inter_y0 = max(float(bbox_y0), float(top))
    inter_x1 = min(float(bbox_x1), float(right))
    inter_y1 = min(float(bbox_y1), float(bottom))
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    return float(inter_area / bbox_area)
