from __future__ import annotations

import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.cellpose_postprocess import (
    build_annotated_png_supplement,
    prune_annotated_false_positives,
)
from pupa_counter.preprocess.paper_region import bbox_fraction_inside_paper_bounds, estimate_paper_bounds


def test_estimate_paper_bounds_excludes_left_dark_scanner_band():
    image = np.full((80, 120, 3), 245, dtype=np.uint8)
    image[:, :14, :] = 35
    image[20:28, 30:40, :] = [150, 95, 70]

    cfg = AppConfig()
    left, top, right, bottom = estimate_paper_bounds(image, cfg=cfg)

    assert left >= 8
    assert right > 100
    assert top == 0
    assert bottom == 80


def test_prune_annotated_false_positives_rejects_candidates_outside_paper():
    cfg = AppConfig()
    candidates = pd.DataFrame(
        [
            {
                "component_id": "edge_fp",
                "centroid_x": 5.0,
                "centroid_y": 40.0,
                "bbox_x0": 0.0,
                "bbox_y0": 30.0,
                "bbox_x1": 10.0,
                "bbox_y1": 50.0,
                "area_px": 90.0,
                "color_score": 0.28,
                "local_contrast": 12.0,
                "mean_v": 120.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.20,
                "touches_image_border": True,
                "detector_source": "cellpose",
                "is_active": True,
                "label": "pupa",
                "confidence": 0.80,
            },
            {
                "component_id": "paper_ok",
                "centroid_x": 40.0,
                "centroid_y": 40.0,
                "bbox_x0": 32.0,
                "bbox_y0": 30.0,
                "bbox_x1": 48.0,
                "bbox_y1": 50.0,
                "area_px": 120.0,
                "color_score": 0.45,
                "local_contrast": 60.0,
                "mean_v": 105.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "touches_image_border": False,
                "detector_source": "cellpose",
                "is_active": True,
                "label": "pupa",
                "confidence": 0.90,
            },
        ]
    )

    pruned = prune_annotated_false_positives(
        candidates,
        source_type="annotated_png",
        cfg=cfg,
        paper_bounds=(14, 0, 120, 80),
    )
    labels = dict(zip(pruned["component_id"], pruned["label"]))
    assert labels["edge_fp"] == "artifact"
    assert labels["paper_ok"] == "pupa"


def test_bbox_fraction_inside_paper_bounds_tracks_partial_overlap():
    fraction = bbox_fraction_inside_paper_bounds(0, 10, 20, 30, (10, 0, 120, 80))
    assert round(fraction, 2) == 0.50


def test_build_annotated_png_supplement_rejects_border_candidate_even_if_centroid_is_on_paper():
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
                "centroid_x": 40.0,
                "centroid_y": 40.0,
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
                "component_id": "cc_paper_border",
                "bbox_x0": 0,
                "bbox_y0": 20,
                "bbox_x1": 18,
                "bbox_y1": 42,
                "centroid_x": 16.0,
                "centroid_y": 31.0,
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
        paper_bounds=(10, 0, 120, 80),
    )

    assert supplement.empty
