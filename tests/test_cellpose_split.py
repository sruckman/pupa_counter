from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.cellpose_split import split_large_cellpose_instances
from pupa_counter.detect.components import build_component_row


def _components_from_masks(masks):
    rows = []
    for idx, mask in enumerate(masks, start=1):
        row = build_component_row(mask, 0, 0, mask.shape, "cp_%05d" % idx)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["aspect_ratio"] = frame["major_axis_px"] / frame["minor_axis_px"].clip(lower=1.0)
    return frame


def test_cellpose_split_splits_two_touching_blobs():
    single = np.zeros((60, 60), dtype=bool)
    single[22:38, 12:24] = True

    merged = np.zeros((60, 60), dtype=bool)
    rr, cc = np.ogrid[:60, :60]
    left = ((rr - 30) ** 2) / (7**2) + ((cc - 18) ** 2) / (5**2) <= 1.0
    right = ((rr - 30) ** 2) / (7**2) + ((cc - 40) ** 2) / (5**2) <= 1.0
    bridge = (rr == 30) & (cc >= 23) & (cc <= 35)
    merged[left | right | bridge] = True

    frame = _components_from_masks([single, merged])
    cfg = AppConfig()
    cfg.detector.cellpose_overlap_split_area_ratio = 1.0
    cfg.detector.cellpose_overlap_split_max_aspect_ratio = 4.0
    cfg.detector.cellpose_overlap_split_max_eccentricity = 0.99
    cfg.detector.cellpose_overlap_split_peak_min_distance = 2
    cfg.detector.cellpose_overlap_split_peak_abs_threshold = 1.0

    split = split_large_cellpose_instances(frame, merged.shape, source_type="annotated_png", cfg=cfg)

    active = split.loc[split["is_active"].astype(bool)]
    assert len(active) == 3
    assert int(split["parent_component_id"].notna().sum()) == 2


def test_cellpose_split_ignores_non_annotated_sources():
    single = np.zeros((60, 60), dtype=bool)
    single[22:38, 12:24] = True

    merged = np.zeros((60, 60), dtype=bool)
    merged[20:38, 12:24] = True
    merged[20:38, 24:36] = True
    merged[24:34, 22:26] = True

    frame = _components_from_masks([single, merged])
    cfg = AppConfig()

    split = split_large_cellpose_instances(frame, merged.shape, source_type="clean_pdf", cfg=cfg)
    assert len(split) == len(frame)
    assert split["component_id"].tolist() == frame["component_id"].tolist()


def test_cellpose_split_allows_elongated_annotated_touching_pair():
    single = np.zeros((70, 70), dtype=bool)
    single[24:42, 10:22] = True

    merged = np.zeros((70, 70), dtype=bool)
    rr, cc = np.ogrid[:70, :70]
    top = ((rr - 20) ** 2) / (7**2) + ((cc - 34) ** 2) / (5**2) <= 1.0
    bottom = ((rr - 50) ** 2) / (7**2) + ((cc - 34) ** 2) / (5**2) <= 1.0
    bridge = (cc == 34) & (rr >= 27) & (rr <= 43)
    merged[top | bottom | bridge] = True

    frame = _components_from_masks([single, merged])
    cfg = AppConfig()
    cfg.detector.cellpose_overlap_split_area_ratio = 1.0
    cfg.detector.cellpose_overlap_split_max_aspect_ratio = 1.20
    cfg.detector.cellpose_overlap_split_max_eccentricity = 0.60
    cfg.detector.cellpose_overlap_split_peak_min_distance = 3
    cfg.detector.cellpose_overlap_split_peak_abs_threshold = 1.0
    cfg.detector.cellpose_overlap_split_annotated_ignore_shape = True

    split = split_large_cellpose_instances(frame, merged.shape, source_type="annotated_png", cfg=cfg)

    active = split.loc[split["is_active"].astype(bool)]
    assert len(active) == 3
    assert int(split["parent_component_id"].notna().sum()) == 2


def test_cellpose_split_can_restrict_to_dense_patch_components_only():
    merged = np.zeros((60, 60), dtype=bool)
    rr, cc = np.ogrid[:60, :60]
    left = ((rr - 30) ** 2) / (7**2) + ((cc - 18) ** 2) / (5**2) <= 1.0
    right = ((rr - 30) ** 2) / (7**2) + ((cc - 40) ** 2) / (5**2) <= 1.0
    bridge = (rr == 30) & (cc >= 23) & (cc <= 35)
    merged[left | right | bridge] = True

    other = np.zeros((60, 60), dtype=bool)
    other[12:28, 10:22] = True

    frame = _components_from_masks([merged, other])
    frame.loc[0, "dense_patch_refined"] = True
    frame.loc[1, "dense_patch_refined"] = False

    cfg = AppConfig()
    cfg.detector.cellpose_overlap_split_area_ratio = 1.0
    cfg.detector.cellpose_overlap_split_peak_min_distance = 2

    split = split_large_cellpose_instances(
        frame,
        merged.shape,
        source_type="annotated_png",
        cfg=cfg,
        restrict_to_dense_patch=True,
    )

    active = split.loc[split["is_active"].astype(bool)]
    assert len(active) == 3
    assert "cp_00002" in active["component_id"].tolist()


def test_cellpose_split_can_use_guide_image_to_split_touching_pair():
    single = np.zeros((80, 80), dtype=bool)
    single[28:44, 10:22] = True

    merged = np.zeros((80, 80), dtype=bool)
    rr, cc = np.ogrid[:80, :80]
    left = ((rr - 40) ** 2) / (10**2) + ((cc - 34) ** 2) / (8**2) <= 1.0
    right = ((rr - 40) ** 2) / (10**2) + ((cc - 50) ** 2) / (8**2) <= 1.0
    bridge = ((rr - 40) ** 2) / (7**2) + ((cc - 42) ** 2) / (10**2) <= 1.0
    merged[left | right | bridge] = True

    frame = _components_from_masks([single, merged])
    cfg = AppConfig()
    cfg.detector.cellpose_overlap_split_peak_abs_threshold = 999.0
    cfg.detector.cellpose_overlap_split_combo_area_ratio = 1.0
    cfg.detector.cellpose_overlap_split_combo_peak_min_distance = 3
    cfg.detector.cellpose_overlap_split_combo_peak_abs_threshold = 0.35
    cfg.detector.cellpose_overlap_split_combo_min_child_area_ratio = 0.18

    guide = np.full((80, 80, 3), 255, dtype=np.uint8)
    cv_mask = merged.astype(np.uint8) * 255
    guide[cv_mask > 0] = np.array([210, 170, 130], dtype=np.uint8)
    guide[34:46, 29:37] = np.array([140, 90, 50], dtype=np.uint8)
    guide[34:46, 47:55] = np.array([140, 90, 50], dtype=np.uint8)

    unsplit = split_large_cellpose_instances(frame, merged.shape, source_type="annotated_png", cfg=cfg)
    assert len(unsplit.loc[unsplit["is_active"].astype(bool)]) == 2

    split = split_large_cellpose_instances(
        frame,
        merged.shape,
        source_type="annotated_png",
        cfg=cfg,
        guide_image=guide,
    )

    active = split.loc[split["is_active"].astype(bool)]
    assert len(active) == 3
    assert int(split["parent_component_id"].notna().sum()) == 2
