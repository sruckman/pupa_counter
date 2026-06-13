"""End-to-end smoke test for ``run_detector``.

The detector pipeline (preprocess → response → allowed mask → peak
detection → export) must not crash on a freshly-synthesized image and must
find the pupae we painted. This is the guard that prevents accidental
breakage of the v1 baseline while we iterate on response sharpening.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pupa_counter_fresh import DetectorConfig, run_detector
from pupa_counter_fresh.detector import DetectorOutput


def _write_tmp_image(tmp_path, rgb: np.ndarray) -> str:
    """Write ``rgb`` to a temp PNG and return the absolute path."""
    import cv2

    path = tmp_path / "synthetic.png"
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    return str(path)


def _detector_config_native_resolution() -> DetectorConfig:
    """Config tuned for the synthetic tests.

    ``work_scale=1.0`` so coordinates in tests map 1:1 with painted pixels.
    Lower thresholds and smaller min-distance because the synthetic pupae
    are smaller than real-scale benchmark pupae.
    """
    return DetectorConfig(
        work_scale=1.0,
        smooth_sigma=1.2,
        allowed_abs_threshold=0.08,
        peak_min_distance_px=5,
        peak_abs_score_threshold=0.12,
        peak_edge_margin_px=2,
        use_component_split=True,
        component_single_pupa_area_px=120.0,
        component_area_ratio_threshold=1.20,
        component_min_peak_distance_px=3,
        component_abs_score_threshold=0.10,
        component_min_component_area_px=20,
        component_max_peaks=10,
    )


def test_detector_returns_expected_shape(make_synthetic_pupa_image, tmp_path):
    rgb = make_synthetic_pupa_image(
        size=(256, 256),
        centers=[(60, 60), (180, 60), (60, 180), (180, 180)],
    )
    path = _write_tmp_image(tmp_path, rgb)
    result: DetectorOutput = run_detector(
        path,
        image_id="synthetic_4pupa",
        cfg=_detector_config_native_resolution(),
    )
    # Contract: return value is a DetectorOutput with an instances frame
    assert isinstance(result, DetectorOutput)
    assert isinstance(result.instances, pd.DataFrame)
    # Debug maps are populated so tests can probe them
    for key in ("rgb_native", "response_map", "allowed_mask", "peak_map"):
        assert key in result.debug, f"missing debug key: {key}"
    assert result.runtime_ms >= 0.0


def test_detector_finds_four_isolated_pupae(make_synthetic_pupa_image, tmp_path):
    rgb = make_synthetic_pupa_image(
        size=(256, 256),
        centers=[(60, 60), (180, 60), (60, 180), (180, 180)],
    )
    path = _write_tmp_image(tmp_path, rgb)
    result = run_detector(
        path,
        image_id="synthetic_4pupa",
        cfg=_detector_config_native_resolution(),
    )
    assert len(result.instances) == 4, (
        f"expected 4 isolated pupae, got {len(result.instances)}"
    )
    # Each prediction must land within 6 px of one painted center
    predicted = result.instances[["centroid_x", "centroid_y"]].to_numpy()
    painted = np.array([(60, 60), (180, 60), (60, 180), (180, 180)], dtype=float)
    for gt in painted:
        dists = np.hypot(predicted[:, 0] - gt[0], predicted[:, 1] - gt[1])
        assert dists.min() < 6.0, (
            f"no prediction within 6 px of painted center {gt}; min={dists.min():.2f}"
        )


def test_detector_respects_blue_mask(make_synthetic_pupa_image, tmp_path):
    """Pupa painted on top of a blue ink stroke must not be counted.

    We paint an X across the image in blue, then one pupa exactly on top of
    it. The blue mask at the preprocess stage should suppress the response
    inside the stroke, so the pupa becomes invisible to the detector.
    """
    import cv2

    rgb = make_synthetic_pupa_image(
        size=(256, 256),
        centers=[(128, 128)],
    )
    # Paint a thick blue X that completely covers the pupa
    cv2.line(rgb, (0, 0), (256, 256), (0, 80, 230), thickness=20)
    cv2.line(rgb, (0, 256), (256, 0), (0, 80, 230), thickness=20)
    path = _write_tmp_image(tmp_path, rgb)
    result = run_detector(
        path,
        image_id="blue_occluded",
        cfg=_detector_config_native_resolution(),
    )
    # No predictions: the blue mask zeros out the pupa's response
    assert len(result.instances) == 0, (
        f"blue-occluded pupa should not be counted, got {len(result.instances)}"
    )


def test_detector_handles_empty_paper(make_synthetic_pupa_image, tmp_path):
    rgb = make_synthetic_pupa_image(size=(128, 128), centers=[])
    path = _write_tmp_image(tmp_path, rgb)
    result = run_detector(
        path,
        image_id="empty_paper",
        cfg=_detector_config_native_resolution(),
    )
    assert len(result.instances) == 0
    assert not np.isnan(result.debug["response_map"]).any()
