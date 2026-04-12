"""Tests for the paper ROI detector.

The 2026-04-10 visual audit found that the dominant v1 false-positive
failure mode was pupae "detected" on the scanner gray strip at the left
edge of the image. This file is the regression gate for the paper-ROI
fix — synthetic scanner bars must be masked out and pupae painted on top
of a bar must not be counted.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pupa_counter_fresh import DetectorConfig, run_detector
from pupa_counter_fresh.paper_roi import (
    PaperROIConfig,
    apply_paper_roi_to_response,
    detect_paper_roi,
)
from pupa_counter_fresh.response import compute_response_map


def _scanner_bar_image(rgb: np.ndarray, bar_width: int = 12, bar_value: int = 60) -> np.ndarray:
    """Paint a dark vertical strip on the left edge of ``rgb`` in-place.

    Simulates the chrome bar the real scanner produces at the left margin.
    """
    out = rgb.copy()
    out[:, :bar_width, :] = bar_value
    return out


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_detect_paper_roi_on_clean_paper(make_synthetic_pupa_image):
    rgb = make_synthetic_pupa_image(size=(128, 128), centers=[(64, 64)])
    mask = detect_paper_roi(rgb)
    assert mask is not None
    assert mask.shape == rgb.shape[:2]
    assert mask.dtype == np.uint8
    # The paper covers almost the entire image except for a small eroded
    # border — must be at least 70% to prove the detector is not chopping
    # the paper up.
    assert (mask > 0).mean() > 0.70


def test_detect_paper_roi_returns_none_on_black_image():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = detect_paper_roi(rgb)
    assert mask is None


# ---------------------------------------------------------------------------
# Scanner bar regression gate
# ---------------------------------------------------------------------------


def test_scanner_bar_is_excluded_from_paper_mask(make_synthetic_pupa_image):
    rgb = make_synthetic_pupa_image(size=(128, 128), centers=[(80, 64)])
    rgb = _scanner_bar_image(rgb, bar_width=12, bar_value=40)
    mask = detect_paper_roi(rgb)
    assert mask is not None
    # The leftmost 12 pixels must be entirely outside the paper mask
    assert (mask[:, :12] > 0).sum() == 0, (
        f"scanner bar pixels should not be part of paper mask; got "
        f"{(mask[:, :12] > 0).sum()} inside-paper pixels on the bar"
    )
    # The pupa at (80, 64) must be well inside the paper mask
    assert mask[64, 80] > 0


def test_pupa_on_scanner_bar_is_not_counted(make_synthetic_pupa_image, tmp_path):
    """End-to-end: a pupa painted on the scanner bar should not appear in
    the detector's instances.csv when ``use_paper_roi=True``."""
    rgb = make_synthetic_pupa_image(
        size=(192, 192),
        centers=[(6, 100), (96, 96)],  # first pupa is *on* the bar
    )
    rgb = _scanner_bar_image(rgb, bar_width=15, bar_value=40)

    path = tmp_path / "scanner_bar.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    cfg_no_roi = DetectorConfig(
        work_scale=1.0,
        allowed_abs_threshold=0.08,
        peak_abs_score_threshold=0.12,
        peak_edge_margin_px=2,
        use_component_split=True,
        component_single_pupa_area_px=120.0,
        component_area_ratio_threshold=1.20,
        component_min_peak_distance_px=3,
        component_abs_score_threshold=0.10,
        component_min_component_area_px=20,
        use_paper_roi=False,
    )
    cfg_with_roi = DetectorConfig(
        work_scale=1.0,
        allowed_abs_threshold=0.08,
        peak_abs_score_threshold=0.12,
        peak_edge_margin_px=2,
        use_component_split=True,
        component_single_pupa_area_px=120.0,
        component_area_ratio_threshold=1.20,
        component_min_peak_distance_px=3,
        component_abs_score_threshold=0.10,
        component_min_component_area_px=20,
        use_paper_roi=True,
    )

    result_no = run_detector(str(path), cfg=cfg_no_roi)
    result_yes = run_detector(str(path), cfg=cfg_with_roi)

    # Without paper ROI, the bar-pupa may be detected (or at least its
    # column may produce something). With paper ROI, the bar pupa must
    # be eliminated. The real-pupa-at-(96,96) must be detected in both.
    def _nearest_distance(df, cx, cy):
        if df.empty:
            return float("inf")
        return float(
            ((df["centroid_x"] - cx) ** 2 + (df["centroid_y"] - cy) ** 2).min() ** 0.5
        )

    # The real pupa is caught both ways
    assert _nearest_distance(result_no.instances, 96, 96) < 8
    assert _nearest_distance(result_yes.instances, 96, 96) < 8

    # The ROI-enabled run must have no prediction inside the bar (x<=15)
    inside_bar_roi = result_yes.instances[result_yes.instances["centroid_x"] <= 15]
    assert len(inside_bar_roi) == 0, (
        f"paper ROI should eliminate predictions inside the scanner bar; "
        f"got {len(inside_bar_roi)} predictions with x<=15"
    )


# ---------------------------------------------------------------------------
# Interaction with response map
# ---------------------------------------------------------------------------


def test_apply_paper_roi_to_response_zeros_outside(make_synthetic_pupa_image):
    rgb = make_synthetic_pupa_image(size=(128, 128), centers=[(80, 64)])
    rgb = _scanner_bar_image(rgb, bar_width=12, bar_value=40)
    response = compute_response_map(rgb, response_mode="smooth")
    mask = detect_paper_roi(rgb)
    assert mask is not None
    gated = apply_paper_roi_to_response(response, mask)
    # Inside bar: response must be 0
    assert gated[:, :12].max() < 1e-6
    # Outside bar: some response from the pupa
    assert gated[60:70, 70:90].max() > 0.2


def test_apply_paper_roi_to_response_is_noop_when_mask_none():
    response = np.ones((32, 32), dtype=np.float32) * 0.5
    out = apply_paper_roi_to_response(response, None)
    assert np.array_equal(out, response)
