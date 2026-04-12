from __future__ import annotations

import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.cellpose_postprocess import (
    build_annotated_png_supplement,
    calibrate_cellpose_detections,
    prune_annotated_false_positives,
)


def test_build_annotated_png_supplement_adds_unmatched_strong_classical_candidates():
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_png_supplement_max_unmatched_ratio = 1.0

    cellpose_df = pd.DataFrame(
        [
            {
                "component_id": "cp_1",
                "bbox_x0": 10,
                "bbox_y0": 10,
                "bbox_x1": 30,
                "bbox_y1": 30,
                "area_px": 120.0,
                "mean_v": 120.0,
                "color_score": 0.40,
                "local_contrast": 12.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "is_active": True,
                "label": "pupa",
                "confidence": 0.90,
                "cluster_area_threshold": 0.0,
            }
        ]
    )

    classical_df = pd.DataFrame(
        [
            {
                "component_id": "cc_1",
                "bbox_x0": 12,
                "bbox_y0": 12,
                "bbox_x1": 29,
                "bbox_y1": 29,
                "area_px": 118.0,
                "mean_v": 118.0,
                "color_score": 0.41,
                "local_contrast": 13.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "is_active": True,
                "label": "pupa",
                "confidence": 0.90,
                "cluster_area_threshold": 0.0,
            },
            {
                "component_id": "cc_2",
                "bbox_x0": 80,
                "bbox_y0": 50,
                "bbox_x1": 96,
                "bbox_y1": 72,
                "area_px": 92.0,
                "mean_v": 110.0,
                "color_score": 0.33,
                "local_contrast": 10.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "is_active": True,
                "label": "pupa",
                "confidence": 0.72,
                "cluster_area_threshold": 0.0,
            },
        ]
    )

    supplement = build_annotated_png_supplement(
        cellpose_df,
        classical_df,
        source_type="annotated_png",
        cfg=cfg,
    )

    assert supplement["component_id"].tolist() == ["cc_2"]
    assert supplement["detector_source"].tolist() == ["annotated_classical_addon"]


def test_build_annotated_png_supplement_rejects_border_touching_candidates():
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_png_supplement_max_unmatched_ratio = 1.0

    cellpose_df = pd.DataFrame(
        [
            {
                "component_id": "cp_1",
                "bbox_x0": 30,
                "bbox_y0": 30,
                "bbox_x1": 50,
                "bbox_y1": 50,
                "area_px": 120.0,
                "mean_v": 120.0,
                "color_score": 0.40,
                "local_contrast": 12.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "is_active": True,
                "label": "pupa",
                "confidence": 0.90,
                "cluster_area_threshold": 0.0,
                "touches_image_border": False,
            }
        ]
    )

    classical_df = pd.DataFrame(
        [
            {
                "component_id": "cc_border",
                "bbox_x0": 0,
                "bbox_y0": 20,
                "bbox_x1": 14,
                "bbox_y1": 42,
                "area_px": 92.0,
                "mean_v": 100.0,
                "color_score": 0.30,
                "local_contrast": 10.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.32,
                "is_active": True,
                "label": "pupa",
                "confidence": 0.80,
                "cluster_area_threshold": 0.0,
                "touches_image_border": True,
            }
        ]
    )

    supplement = build_annotated_png_supplement(
        cellpose_df,
        classical_df,
        source_type="annotated_png",
        cfg=cfg,
    )

    assert supplement.empty


def test_calibrate_cellpose_detections_only_rejects_extreme_bright_artifacts():
    cfg = AppConfig()
    features = pd.DataFrame(
        [
            {
                "component_id": "bad_bright",
                "area_px": 420.0,
                "solidity": 0.90,
                "eccentricity": 0.80,
                "aspect_ratio": 1.8,
                "color_score": 0.05,
                "local_contrast": -5.0,
                "mean_s": 8.0,
                "mean_v": 250.0,
                "blue_overlap_ratio": 0.88,
                "border_touch_ratio": 0.03,
                "touches_image_border": False,
            },
            {
                "component_id": "border_candidate",
                "area_px": 40.0,
                "solidity": 0.92,
                "eccentricity": 0.75,
                "aspect_ratio": 1.6,
                "color_score": 0.30,
                "local_contrast": 30.0,
                "mean_s": 40.0,
                "mean_v": 120.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.70,
                "touches_image_border": True,
            },
        ]
    )

    labeled = calibrate_cellpose_detections(features, source_type="annotated_png", cfg=cfg)
    assert labeled["label"].tolist() == ["artifact", "pupa"]


def test_prune_annotated_false_positives_filters_blue_and_border_artifacts():
    cfg = AppConfig()
    features = pd.DataFrame(
        [
            {
                "component_id": "blue_fp",
                "area_px": 150.0,
                "bbox_x0": 2.0,
                "bbox_y0": 22.0,
                "bbox_x1": 12.0,
                "bbox_y1": 38.0,
                "color_score": 0.10,
                "local_contrast": 10.0,
                "mean_v": 210.0,
                "blue_overlap_ratio": 0.22,
                "border_touch_ratio": 0.0,
                "centroid_x": 5.0,
                "centroid_y": 30.0,
                "touches_image_border": False,
                "detector_source": "cellpose_dense_patch",
                "is_active": True,
                "label": "pupa",
                "confidence": 0.90,
            },
            {
                "component_id": "border_fp",
                "area_px": 90.0,
                "bbox_x0": 0.0,
                "bbox_y0": 25.0,
                "bbox_x1": 12.0,
                "bbox_y1": 45.0,
                "color_score": 0.28,
                "local_contrast": 12.0,
                "mean_v": 130.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.60,
                "centroid_x": 6.0,
                "centroid_y": 35.0,
                "touches_image_border": True,
                "detector_source": "annotated_classical_addon",
                "is_active": True,
                "label": "pupa",
                "confidence": 0.85,
            },
            {
                "component_id": "good_pair",
                "area_px": 300.0,
                "bbox_x0": 30.0,
                "bbox_y0": 24.0,
                "bbox_x1": 52.0,
                "bbox_y1": 48.0,
                "color_score": 0.42,
                "local_contrast": 100.0,
                "mean_v": 120.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "centroid_x": 40.0,
                "centroid_y": 35.0,
                "touches_image_border": False,
                "detector_source": "cellpose",
                "is_active": True,
                "label": "pupa",
                "confidence": 0.92,
            },
        ]
    )

    pruned = prune_annotated_false_positives(
        features,
        source_type="annotated_png",
        cfg=cfg,
        paper_bounds=(14, 0, 120, 80),
    )
    labels = dict(zip(pruned["component_id"], pruned["label"]))
    assert labels["blue_fp"] == "artifact"
    assert labels["border_fp"] == "artifact"
    assert labels["good_pair"] == "pupa"
