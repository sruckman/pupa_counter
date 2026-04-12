from __future__ import annotations

import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.cellpose_dual_path import merge_annotated_detection_paths, merge_annotated_pair_rescue
from pupa_counter.detect.components import build_component_row


def _ellipse_mask(shape, center_y, center_x, radius_y=5, radius_x=9):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - center_y) ** 2) / float(radius_y ** 2) + ((xx - center_x) ** 2) / float(radius_x ** 2) <= 1.0


def _build_row(mask, component_id, detector_source="cellpose", confidence=0.9):
    row = build_component_row(mask.astype(bool), 0, 0, mask.shape, component_id)
    row["image_height"] = int(mask.shape[0])
    row["image_width"] = int(mask.shape[1])
    row["label"] = "pupa"
    row["confidence"] = confidence
    row["is_active"] = True
    row["blue_overlap_ratio"] = 0.0
    row["detector_source"] = detector_source
    return row


def test_dual_path_merge_adds_extra_normalized_detection_in_dense_patch():
    shape = (120, 120)
    clean_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 40, 30), "cp_00001"),
            _build_row(_ellipse_mask(shape, 40, 70), "cp_00002"),
        ]
    )
    normalized_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 40, 30), "np_00001"),
            _build_row(_ellipse_mask(shape, 40, 50), "np_00002"),
            _build_row(_ellipse_mask(shape, 40, 70), "np_00003"),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_dual_path_min_instances = 2
    cfg.detector.cellpose_annotated_dual_path_max_gain_ratio = 1.6

    merged = merge_annotated_detection_paths(clean_df, normalized_df, image_shape=shape, cfg=cfg)

    assert len(merged) == 3
    assert int(merged["dual_path_selected"].astype(bool).sum()) == 3
    assert any(str(value).startswith("normalized_") for value in merged["detector_source"].tolist())


def test_dual_path_merge_rejects_unreasonably_large_gain():
    shape = (120, 120)
    clean_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 40, 30), "cp_00001"),
            _build_row(_ellipse_mask(shape, 40, 70), "cp_00002"),
        ]
    )
    normalized_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 35, 20), "np_00001"),
            _build_row(_ellipse_mask(shape, 35, 40), "np_00002"),
            _build_row(_ellipse_mask(shape, 40, 60), "np_00003"),
            _build_row(_ellipse_mask(shape, 45, 80), "np_00004"),
            _build_row(_ellipse_mask(shape, 45, 100), "np_00005"),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_dual_path_min_instances = 2
    cfg.detector.cellpose_annotated_dual_path_max_gain_ratio = 1.4

    merged = merge_annotated_detection_paths(clean_df, normalized_df, image_shape=shape, cfg=cfg)

    assert len(merged) == len(clean_df)
    assert not merged.get("dual_path_selected", pd.Series([], dtype=bool)).astype(bool).any()


def test_pair_rescue_replaces_one_primary_blob_with_two_split_children():
    shape = (120, 120)
    primary_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 40, 50, radius_y=8, radius_x=18), "cp_00001"),
        ]
    )
    classical_df = pd.DataFrame(
        [
            dict(
                _build_row(_ellipse_mask(shape, 40, 42), "cc_00001_child_01", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 40, 58), "cc_00001_child_02", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_pair_rescue_min_children = 2
    cfg.detector.cellpose_annotated_pair_rescue_max_children = 3
    cfg.detector.cellpose_annotated_pair_rescue_max_gain_ratio = 2.5

    merged = merge_annotated_pair_rescue(primary_df, classical_df, image_shape=shape, cfg=cfg)

    assert len(merged) == 2
    assert int(merged["pair_rescue_selected"].astype(bool).sum()) == 2
    assert all(str(value).startswith("pair_") for value in merged["detector_source"].tolist())


def test_pair_rescue_uses_child_scale_not_primary_blob_scale():
    shape = (160, 160)
    primary_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 70, 80, radius_y=14, radius_x=34), "cp_00001"),
        ]
    )
    classical_df = pd.DataFrame(
        [
            dict(
                _build_row(_ellipse_mask(shape, 70, 70, radius_y=5, radius_x=10), "cc_00001_child_01", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 70, 90, radius_y=5, radius_x=10), "cc_00001_child_02", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_pair_rescue_min_children = 2
    cfg.detector.cellpose_annotated_pair_rescue_max_children = 3
    cfg.detector.cellpose_annotated_pair_rescue_max_gain_ratio = 2.5

    merged = merge_annotated_pair_rescue(primary_df, classical_df, image_shape=shape, cfg=cfg)

    assert len(merged) == 2
    assert sorted(merged["component_id"].tolist()) == ["cc_00001_child_01", "cc_00001_child_02"]


def test_pair_rescue_can_promote_uncertain_children():
    shape = (120, 120)
    primary_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 40, 50, radius_y=8, radius_x=18), "cp_00001"),
        ]
    )
    classical_df = pd.DataFrame(
        [
            dict(
                _build_row(_ellipse_mask(shape, 40, 42), "cc_00001_child_01", detector_source="classical", confidence=0.57),
                split_from_cluster=True,
                parent_component_id="cc_00001",
                label="uncertain",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 40, 58), "cc_00001_child_02", detector_source="classical", confidence=0.59),
                split_from_cluster=True,
                parent_component_id="cc_00001",
                label="uncertain",
            ),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_pair_rescue_min_children = 2
    cfg.detector.cellpose_annotated_pair_rescue_max_children = 3
    cfg.detector.cellpose_annotated_pair_rescue_max_gain_ratio = 2.5
    cfg.detector.cellpose_annotated_pair_rescue_min_confidence = 0.50
    cfg.counting.min_instance_confidence = 0.55

    merged = merge_annotated_pair_rescue(primary_df, classical_df, image_shape=shape, cfg=cfg)

    assert len(merged) == 2
    assert set(merged["label"].tolist()) == {"pupa"}
    assert (merged["confidence"].astype(float) >= 0.55).all()


def test_pair_rescue_skips_regions_with_multiple_primary_matches():
    shape = (140, 140)
    primary_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 50, 52), "cp_00001"),
            _build_row(_ellipse_mask(shape, 50, 68), "cp_00002"),
        ]
    )
    classical_df = pd.DataFrame(
        [
            dict(
                _build_row(_ellipse_mask(shape, 50, 48), "cc_00001_child_01", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 50, 72), "cc_00001_child_02", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_pair_rescue_min_children = 2
    cfg.detector.cellpose_annotated_pair_rescue_max_children = 3

    merged = merge_annotated_pair_rescue(primary_df, classical_df, image_shape=shape, cfg=cfg)

    assert len(merged) == len(primary_df)
    assert not merged.get("pair_rescue_selected", pd.Series([], dtype=bool)).fillna(False).astype(bool).any()


def test_pair_rescue_can_add_strong_group_without_primary_match_when_on_paper():
    shape = (140, 140)
    primary_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 30, 30), "cp_00001"),
        ]
    )
    classical_df = pd.DataFrame(
        [
            dict(
                _build_row(_ellipse_mask(shape, 90, 78), "cc_00001_child_01", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 90, 98), "cc_00001_child_02", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_pair_rescue_min_children = 2
    cfg.detector.cellpose_annotated_pair_rescue_max_children = 3

    merged = merge_annotated_pair_rescue(
        primary_df,
        classical_df,
        image_shape=shape,
        cfg=cfg,
        paper_bounds=(10, 0, 140, 140),
    )

    assert len(merged) == len(primary_df) + 2
    assert int(merged["pair_rescue_selected"].astype(bool).sum()) == 2


def test_pair_rescue_skips_groups_without_primary_match_if_outside_paper():
    shape = (140, 140)
    primary_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 30, 30), "cp_00001"),
        ]
    )
    classical_df = pd.DataFrame(
        [
            dict(
                _build_row(_ellipse_mask(shape, 90, 6), "cc_00001_child_01", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 90, 18), "cc_00001_child_02", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_pair_rescue_min_children = 2
    cfg.detector.cellpose_annotated_pair_rescue_max_children = 3

    merged = merge_annotated_pair_rescue(
        primary_df,
        classical_df,
        image_shape=shape,
        cfg=cfg,
        paper_bounds=(20, 0, 140, 140),
    )

    assert len(merged) == len(primary_df)
    assert not merged.get("pair_rescue_selected", pd.Series([], dtype=bool)).fillna(False).astype(bool).any()


def test_pair_rescue_skips_parents_that_were_complex_before_filtering():
    shape = (160, 160)
    primary_df = pd.DataFrame(
        [
            _build_row(_ellipse_mask(shape, 70, 80, radius_y=10, radius_x=24), "cp_00001"),
        ]
    )
    classical_df = pd.DataFrame(
        [
            dict(
                _build_row(_ellipse_mask(shape, 70, 68, radius_y=5, radius_x=9), "cc_00001_child_01", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 70, 92, radius_y=5, radius_x=9), "cc_00001_child_02", detector_source="classical"),
                split_from_cluster=True,
                parent_component_id="cc_00001",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 55, 80, radius_y=3, radius_x=5), "cc_00001_child_03", detector_source="classical", confidence=0.20),
                split_from_cluster=True,
                parent_component_id="cc_00001",
                label="artifact",
            ),
            dict(
                _build_row(_ellipse_mask(shape, 85, 80, radius_y=3, radius_x=5), "cc_00001_child_04", detector_source="classical", confidence=0.20),
                split_from_cluster=True,
                parent_component_id="cc_00001",
                label="artifact",
            ),
        ]
    )
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_pair_rescue_min_children = 2
    cfg.detector.cellpose_annotated_pair_rescue_max_children = 3

    merged = merge_annotated_pair_rescue(primary_df, classical_df, image_shape=shape, cfg=cfg)

    assert len(merged) == len(primary_df)
    assert not merged.get("pair_rescue_selected", pd.Series([], dtype=bool)).fillna(False).astype(bool).any()
