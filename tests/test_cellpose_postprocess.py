from __future__ import annotations

import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.cellpose_postprocess import build_clean_png_supplement, calibrate_cellpose_detections


def _row(**overrides):
    base = {
        "component_id": "cp_00001",
        "bbox_x0": 100,
        "bbox_y0": 100,
        "bbox_x1": 120,
        "bbox_y1": 112,
        "area_px": 95.0,
        "solidity": 0.92,
        "eccentricity": 0.86,
        "aspect_ratio": 1.95,
        "color_score": 0.42,
        "local_contrast": 48.0,
        "mean_s": 92.0,
        "mean_v": 132.0,
        "blue_overlap_ratio": 0.0,
        "border_touch_ratio": 0.0,
        "is_active": True,
        "label": "pupa",
        "confidence": 0.80,
        "cluster_unresolved": False,
        "cluster_area_threshold": 0.0,
    }
    base.update(overrides)
    return base


def test_calibrate_cellpose_rejects_clean_bright_low_color_artifact():
    frame = pd.DataFrame(
        [
            _row(component_id="good"),
            _row(component_id="bad", mean_v=210.0, color_score=0.28, local_contrast=18.0),
        ]
    )

    calibrated = calibrate_cellpose_detections(frame, source_type="clean_png", cfg=AppConfig())
    labels = dict(zip(calibrated["component_id"], calibrated["label"]))
    assert labels["good"] == "pupa"
    assert labels["bad"] == "artifact"


def test_calibrate_cellpose_does_not_apply_clean_artifact_gate_to_annotated_png():
    frame = pd.DataFrame([_row(component_id="candidate", mean_v=210.0, color_score=0.28, local_contrast=18.0)])

    calibrated = calibrate_cellpose_detections(frame, source_type="annotated_png", cfg=AppConfig())
    assert calibrated.iloc[0]["label"] == "pupa"


def test_clean_png_supplement_adds_only_strong_non_overlapping_small_candidates():
    cellpose_df = pd.DataFrame([_row(component_id="cp_1", bbox_x0=100, bbox_y0=100, bbox_x1=120, bbox_y1=112)])
    classical_df = pd.DataFrame(
        [
            _row(component_id="cc_keep", bbox_x0=200, bbox_y0=200, bbox_x1=212, bbox_y1=210, area_px=88.0, mean_v=96.0),
            _row(component_id="cc_overlap", bbox_x0=102, bbox_y0=101, bbox_x1=118, bbox_y1=111, area_px=84.0, mean_v=90.0),
            _row(component_id="cc_too_big", bbox_x0=300, bbox_y0=300, bbox_x1=340, bbox_y1=330, area_px=240.0, mean_v=90.0),
        ]
    )

    cfg = AppConfig()
    cfg.detector.clean_png_supplement_max_unmatched_ratio = 3.0

    supplement = build_clean_png_supplement(
        cellpose_df,
        classical_df,
        source_type="clean_png",
        cfg=cfg,
    )
    assert list(supplement["component_id"]) == ["cc_keep"]
    assert supplement.iloc[0]["detector_source"] == "classical_addon"


def test_clean_png_supplement_skips_noisy_unmatched_batches():
    cellpose_df = pd.DataFrame([_row(component_id="cp_1") for _ in range(10)])
    classical_df = pd.DataFrame(
        [
            _row(component_id="cc_1", bbox_x0=200, bbox_y0=200, bbox_x1=212, bbox_y1=210),
            _row(component_id="cc_2", bbox_x0=240, bbox_y0=200, bbox_x1=252, bbox_y1=210),
            _row(component_id="cc_3", bbox_x0=280, bbox_y0=200, bbox_x1=292, bbox_y1=210),
        ]
    )

    supplement = build_clean_png_supplement(
        cellpose_df,
        classical_df,
        source_type="clean_png",
        cfg=AppConfig(),
    )
    assert supplement.empty
