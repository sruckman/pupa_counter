from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

import pupa_counter.detect.cv_peak_deblend as cv_peak_deblend
from pupa_counter.config import AppConfig
from pupa_counter.detect.components import extract_components
from pupa_counter.detect.cv_peak_deblend import (
    build_foreground_mask,
    compute_fast_brown_score,
    detect_instances,
    refine_component_candidates,
    refine_labeled_candidates,
)
from pupa_counter.detect.features import featurize_components
from pupa_counter.detect.rule_filter import rule_classify_components


def _blank(shape=(180, 180)):
    return np.full((shape[0], shape[1], 3), 255, dtype=np.uint8)


def _draw_pupa(image, center, axes=(10, 5), angle=0, color=(175, 125, 80)):
    cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)


def test_cv_peak_deblend_splits_touching_pair():
    image = _blank(shape=(220, 220))
    _draw_pupa(image, (80, 110), axes=(12, 6), angle=20)
    _draw_pupa(image, (102, 110), axes=(12, 6), angle=-20)
    _draw_pupa(image, (150, 50), axes=(11, 5), angle=8)
    _draw_pupa(image, (170, 90), axes=(11, 5), angle=-10)
    _draw_pupa(image, (145, 150), axes=(11, 5), angle=12)
    _draw_pupa(image, (50, 50), axes=(11, 5), angle=-15)

    cfg = AppConfig()
    cfg.detector.cv_max_side_px = 1024
    cfg.detector.cv_min_area_px = 20
    cfg.detector.cv_suspicious_area_ratio = 1.10
    cfg.detector.cv_suspicious_major_ratio = 1.10
    cfg.detector.cv_local_peak_threshold_percentile = 98.5
    cfg.detector.cv_peak_min_distance_minor_scale = 0.45

    blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    detected = detect_instances(
        image,
        cfg,
        source_type="annotated_png",
        blue_mask=blue_mask,
        paper_bounds=None,
    )

    assert len(detected) >= 6
    assert (detected["cv_seed_count"].astype(int) > 1).any()
    assert detected["component_id"].astype(str).str.startswith("cv_").all()


def test_cv_peak_deblend_respects_paper_bounds_for_edge_artifacts():
    image = _blank(shape=(160, 220))
    _draw_pupa(image, (30, 80), axes=(11, 5), angle=0)
    _draw_pupa(image, (145, 80), axes=(11, 5), angle=0)

    cfg = AppConfig()
    cfg.detector.cv_max_side_px = 1024
    cfg.detector.cv_min_area_px = 20

    blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bounds = (60, 0, image.shape[1], image.shape[0])
    foreground = build_foreground_mask(image, cfg, blue_mask=blue_mask, paper_bounds=bounds)
    detected = detect_instances(
        image,
        cfg,
        source_type="annotated_png",
        blue_mask=blue_mask,
        paper_bounds=bounds,
    )

    assert int(np.sum(foreground[:, :50] > 0)) == 0
    assert len(detected) >= 1
    assert (detected["centroid_x"].astype(float) > 60).all()


def test_cv_peak_deblend_post_refine_recovers_extra_instances():
    image = _blank(shape=(240, 240))
    _draw_pupa(image, (74, 90), axes=(13, 6), angle=18)
    _draw_pupa(image, (95, 94), axes=(13, 6), angle=-12)
    _draw_pupa(image, (162, 80), axes=(12, 6), angle=10)
    _draw_pupa(image, (182, 92), axes=(12, 6), angle=-18)
    _draw_pupa(image, (145, 165), axes=(11, 5), angle=20)
    _draw_pupa(image, (55, 165), axes=(11, 5), angle=-15)

    cfg = AppConfig()
    cfg.detector.cv_min_area_px = 20
    cfg.detector.cv_suspicious_area_ratio = 1.10
    cfg.detector.cv_suspicious_major_ratio = 1.10
    cfg.detector.cv_local_peak_threshold_percentile = 98.5
    cfg.detector.cv_peak_min_distance_minor_scale = 0.45
    cfg.detector.cv_patch_supplement_max_extra_ratio = 0.15

    blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    foreground = build_foreground_mask(image, cfg, blue_mask=blue_mask, paper_bounds=None)
    base_components = extract_components(foreground, cfg)
    score = compute_fast_brown_score(image)
    refined_components = refine_component_candidates(
        base_components,
        score,
        foreground > 0,
        cfg,
        blue_mask=blue_mask,
        paper_bounds=None,
        component_prefix="cv",
    )
    labeled = rule_classify_components(featurize_components(image, blue_mask, refined_components), cfg)
    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground > 0,
        feature_image=image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    before_pupa = int((labeled["label"] == "pupa").sum())
    after_pupa = int((refined["label"] == "pupa").sum())
    assert after_pupa >= before_pupa
    assert len(refined) >= len(labeled)

def test_cv_peak_deblend_promotes_strong_single_like_artifact():
    cfg = AppConfig()
    feature_image = _blank(shape=(120, 120))
    score = compute_fast_brown_score(feature_image)
    foreground = np.zeros(feature_image.shape[:2], dtype=bool)
    blue_mask = np.zeros(feature_image.shape[:2], dtype=np.uint8)

    labeled = pd.DataFrame(
        [
            {
                "component_id": "cmp_ref",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.72,
                "color_score": 0.58,
                "local_contrast": 24.0,
                "mean_v": 150.0,
                "mean_lab_b": 154.0,
                "aspect_ratio": 1.9,
                "eccentricity": 0.82,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.02,
                "area_px": 110.0,
                "major_axis_px": 20.0,
                "minor_axis_px": 7.0,
                "solidity": 0.88,
                "touches_image_border": False,
                "bbox_y0": 10,
                "bbox_x0": 10,
                "bbox_y1": 24,
                "bbox_x1": 30,
                "centroid_y": 17.0,
                "centroid_x": 20.0,
                "mask": np.ones((14, 20), dtype=bool),
                "cv_seed_count": 1,
            },
            {
                "component_id": "cmp_art",
                "label": "artifact",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.56,
                "color_score": 0.60,
                "local_contrast": 28.0,
                "mean_v": 145.0,
                "mean_lab_b": 156.0,
                "aspect_ratio": 1.6,
                "eccentricity": 0.73,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.06,
                "area_px": 105.0,
                "major_axis_px": 19.0,
                "minor_axis_px": 7.0,
                "solidity": 0.83,
                "touches_image_border": False,
                "bbox_y0": 60,
                "bbox_x0": 55,
                "bbox_y1": 72,
                "bbox_x1": 74,
                "centroid_y": 66.0,
                "centroid_x": 64.5,
                "mask": np.ones((12, 19), dtype=bool),
                "cv_seed_count": 1,
            },
        ]
    )

    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground,
        feature_image=feature_image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    promoted = refined.loc[refined["component_id"] == "cmp_art"].iloc[0]
    assert promoted["label"] == "pupa"
    assert float(promoted["confidence"]) >= 0.60


def test_cv_peak_deblend_resplits_strong_artifact_blob():
    image = _blank(shape=(220, 220))
    _draw_pupa(image, (92, 110), axes=(12, 6), angle=20)
    _draw_pupa(image, (114, 110), axes=(12, 6), angle=-20)
    _draw_pupa(image, (160, 70), axes=(11, 5), angle=8)
    _draw_pupa(image, (55, 55), axes=(11, 5), angle=-12)

    cfg = AppConfig()
    cfg.detector.cv_min_area_px = 20
    cfg.detector.cv_suspicious_area_ratio = 1.10
    cfg.detector.cv_suspicious_major_ratio = 1.10
    cfg.detector.cv_local_peak_threshold_percentile = 98.5
    cfg.detector.cv_peak_min_distance_minor_scale = 0.45

    blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    foreground = build_foreground_mask(image, cfg, blue_mask=blue_mask, paper_bounds=None)
    base_components = extract_components(foreground, cfg)
    score = compute_fast_brown_score(image)
    refined_components = refine_component_candidates(
        base_components,
        score,
        foreground > 0,
        cfg,
        blue_mask=blue_mask,
        paper_bounds=None,
        component_prefix="cv",
    )
    labeled = rule_classify_components(featurize_components(image, blue_mask, refined_components), cfg)
    target_idx = labeled["area_px"].astype(float).idxmax()
    labeled.loc[target_idx, "label"] = "artifact"
    labeled.loc[target_idx, "color_score"] = max(float(labeled.loc[target_idx, "color_score"]), 0.58)
    labeled.loc[target_idx, "local_contrast"] = max(float(labeled.loc[target_idx, "local_contrast"]), 28.0)
    labeled.loc[target_idx, "mean_v"] = min(float(labeled.loc[target_idx, "mean_v"]), 150.0)
    labeled.loc[target_idx, "mean_lab_b"] = min(float(labeled.loc[target_idx, "mean_lab_b"]), 158.0)

    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground > 0,
        feature_image=image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    before_pupa = int((labeled["label"] == "pupa").sum())
    after_pupa = int((refined["label"] == "pupa").sum())
    assert after_pupa >= before_pupa + 1


def test_cv_peak_deblend_suppresses_border_split_false_positives():
    cfg = AppConfig()
    feature_image = _blank(shape=(120, 120))
    score = compute_fast_brown_score(feature_image)
    foreground = np.zeros(feature_image.shape[:2], dtype=bool)
    blue_mask = np.zeros(feature_image.shape[:2], dtype=np.uint8)

    labeled = pd.DataFrame(
        [
            {
                "component_id": "cmp_keep",
                "parent_component_id": "",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.70,
                "color_score": 0.50,
                "local_contrast": 36.0,
                "mean_v": 150.0,
                "mean_lab_b": 150.0,
                "aspect_ratio": 1.8,
                "eccentricity": 0.82,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.16,
                "touches_image_border": True,
                "area_px": 120.0,
                "major_axis_px": 22.0,
                "minor_axis_px": 7.0,
                "solidity": 0.90,
                "bbox_y0": 6,
                "bbox_x0": 2,
                "bbox_y1": 18,
                "bbox_x1": 23,
                "centroid_y": 12.0,
                "centroid_x": 10.0,
                "mask": np.ones((12, 21), dtype=bool),
                "cv_seed_count": 1,
            },
            {
                "component_id": "cmp_drop",
                "parent_component_id": "cmp_parent",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.74,
                "color_score": 0.31,
                "local_contrast": 70.0,
                "mean_v": 145.0,
                "mean_lab_b": 148.0,
                "aspect_ratio": 1.7,
                "eccentricity": 0.80,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.24,
                "touches_image_border": True,
                "area_px": 150.0,
                "major_axis_px": 24.0,
                "minor_axis_px": 7.0,
                "solidity": 0.86,
                "bbox_y0": 20,
                "bbox_x0": 0,
                "bbox_y1": 34,
                "bbox_x1": 19,
                "centroid_y": 27.0,
                "centroid_x": 9.0,
                "mask": np.ones((14, 19), dtype=bool),
                "cv_seed_count": 3,
            },
        ]
    )

    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground,
        feature_image=feature_image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    keep = refined.loc[refined["component_id"] == "cmp_keep"].iloc[0]
    drop = refined.loc[refined["component_id"] == "cmp_drop"].iloc[0]
    assert keep["label"] == "pupa"
    assert drop["label"] == "artifact"


def test_cv_peak_deblend_promotes_large_single_like_artifact():
    cfg = AppConfig()
    cfg.detector.cv_artifact_resplit_enabled = False
    feature_image = _blank(shape=(120, 120))
    score = compute_fast_brown_score(feature_image)
    foreground = np.zeros(feature_image.shape[:2], dtype=bool)
    blue_mask = np.zeros(feature_image.shape[:2], dtype=np.uint8)

    labeled = pd.DataFrame(
        [
            {
                "component_id": "cmp_ref",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.70,
                "color_score": 0.58,
                "local_contrast": 28.0,
                "mean_v": 148.0,
                "mean_lab_b": 156.0,
                "whitespace_ratio": 0.24,
                "aspect_ratio": 1.9,
                "eccentricity": 0.84,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.03,
                "area_px": 130.0,
                "major_axis_px": 22.0,
                "minor_axis_px": 7.0,
                "solidity": 0.88,
                "touches_image_border": False,
                "bbox_y0": 10,
                "bbox_x0": 12,
                "bbox_y1": 24,
                "bbox_x1": 34,
                "centroid_y": 17.0,
                "centroid_x": 23.0,
                "mask": np.ones((14, 22), dtype=bool),
                "cv_seed_count": 1,
            },
            {
                "component_id": "cmp_large",
                "label": "artifact",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.52,
                "color_score": 0.56,
                "local_contrast": 33.0,
                "mean_v": 152.0,
                "mean_lab_b": 158.0,
                "whitespace_ratio": 0.34,
                "aspect_ratio": 2.1,
                "eccentricity": 0.89,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.08,
                "area_px": 380.0,
                "major_axis_px": 34.0,
                "minor_axis_px": 11.0,
                "solidity": 0.85,
                "touches_image_border": False,
                "bbox_y0": 60,
                "bbox_x0": 50,
                "bbox_y1": 78,
                "bbox_x1": 86,
                "centroid_y": 69.0,
                "centroid_x": 68.0,
                "mask": np.ones((18, 36), dtype=bool),
                "cv_seed_count": 1,
            },
        ]
    )

    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground,
        feature_image=feature_image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    promoted = refined.loc[refined["component_id"] == "cmp_large"].iloc[0]
    assert promoted["label"] == "pupa"
    assert float(promoted["confidence"]) >= 0.58


def test_cv_peak_deblend_resplits_large_cluster_like_row():
    image = _blank(shape=(220, 220))
    _draw_pupa(image, (92, 110), axes=(12, 6), angle=16)
    _draw_pupa(image, (116, 110), axes=(12, 6), angle=-18)
    _draw_pupa(image, (158, 70), axes=(11, 5), angle=8)
    _draw_pupa(image, (55, 55), axes=(11, 5), angle=-12)

    cfg = AppConfig()
    cfg.detector.cv_min_area_px = 20
    cfg.detector.cv_suspicious_area_ratio = 1.10
    cfg.detector.cv_suspicious_major_ratio = 1.10
    cfg.detector.cv_local_peak_threshold_percentile = 98.5
    cfg.detector.cv_peak_min_distance_minor_scale = 0.45
    cfg.detector.cv_large_cluster_resplit_enabled = True
    cfg.detector.cv_large_cluster_resplit_min_area_ratio = 2.0

    blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    foreground = build_foreground_mask(image, cfg, blue_mask=blue_mask, paper_bounds=None)
    base_components = extract_components(foreground, cfg)
    score = compute_fast_brown_score(image)
    refined_components = refine_component_candidates(
        base_components,
        score,
        foreground > 0,
        cfg,
        blue_mask=blue_mask,
        paper_bounds=None,
        component_prefix="cv",
    )
    labeled = rule_classify_components(featurize_components(image, blue_mask, refined_components), cfg)
    target_idx = labeled["area_px"].astype(float).idxmax()
    labeled.loc[target_idx, "label"] = "cluster"
    labeled.loc[target_idx, "cluster_unresolved"] = True
    labeled.loc[target_idx, "cv_seed_count"] = 1
    labeled.loc[target_idx, "color_score"] = max(float(labeled.loc[target_idx, "color_score"]), 0.50)
    labeled.loc[target_idx, "local_contrast"] = max(float(labeled.loc[target_idx, "local_contrast"]), 32.0)
    labeled.loc[target_idx, "border_touch_ratio"] = 0.02

    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground > 0,
        feature_image=image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    assert len(refined) >= len(labeled) + 1
    assert (refined["parent_component_id"].astype(str) == str(labeled.loc[target_idx, "component_id"])).any()


def test_cv_peak_deblend_resplits_large_pupa_like_row(monkeypatch):
    cfg = AppConfig()
    cfg.detector.cv_large_pupa_resplit_enabled = True
    feature_image = _blank(shape=(120, 120))
    score = compute_fast_brown_score(feature_image)
    foreground = np.zeros(feature_image.shape[:2], dtype=bool)
    foreground[45:75, 45:85] = True
    blue_mask = np.zeros(feature_image.shape[:2], dtype=np.uint8)

    labeled = pd.DataFrame(
        [
            {
                "component_id": "cmp_ref",
                "parent_component_id": "",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.72,
                "color_score": 0.58,
                "local_contrast": 24.0,
                "mean_v": 150.0,
                "mean_lab_b": 154.0,
                "aspect_ratio": 1.9,
                "eccentricity": 0.82,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.02,
                "area_px": 110.0,
                "major_axis_px": 20.0,
                "minor_axis_px": 7.0,
                "solidity": 0.88,
                "touches_image_border": False,
                "bbox_y0": 10,
                "bbox_x0": 10,
                "bbox_y1": 24,
                "bbox_x1": 30,
                "centroid_y": 17.0,
                "centroid_x": 20.0,
                "mask": np.ones((14, 20), dtype=bool),
                "cv_seed_count": 1,
            },
            {
                "component_id": "cmp_pair",
                "parent_component_id": "",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.70,
                "color_score": 0.60,
                "local_contrast": 30.0,
                "mean_v": 148.0,
                "mean_lab_b": 154.0,
                "aspect_ratio": 2.0,
                "eccentricity": 0.86,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.02,
                "area_px": 420.0,
                "major_axis_px": 42.0,
                "minor_axis_px": 12.0,
                "solidity": 0.82,
                "touches_image_border": False,
                "bbox_y0": 45,
                "bbox_x0": 45,
                "bbox_y1": 75,
                "bbox_x1": 85,
                "centroid_y": 60.0,
                "centroid_x": 65.0,
                "mask": np.ones((30, 40), dtype=bool),
                "cv_seed_count": 2,
            },
        ]
    )

    def fake_deblend_component(*args, **kwargs):
        row = args[0]
        if str(row["component_id"]) != "cmp_pair":
            return None
        child_a = cv_peak_deblend._build_child_row(
            np.ones((14, 16), dtype=bool),
            patch_y0=46,
            patch_x0=46,
            image_shape=feature_image.shape[:2],
            component_id="cmp_pair_cv_01",
            parent_component_id="cmp_pair",
            seed_count=2,
            dense_deblend=False,
            center_only=False,
        )
        child_b = cv_peak_deblend._build_child_row(
            np.ones((14, 16), dtype=bool),
            patch_y0=58,
            patch_x0=63,
            image_shape=feature_image.shape[:2],
            component_id="cmp_pair_cv_02",
            parent_component_id="cmp_pair",
            seed_count=2,
            dense_deblend=False,
            center_only=False,
        )
        return [child_a, child_b]

    monkeypatch.setattr(cv_peak_deblend, "_deblend_component", fake_deblend_component)

    def fake_reclassify(component_df, *, feature_image, blue_mask, cfg):
        updated = component_df.copy()
        updated["label"] = "pupa"
        updated["confidence"] = 0.72
        updated["rule_score"] = 0.72
        updated["color_score"] = 0.60
        updated["local_contrast"] = 30.0
        updated["blue_overlap_ratio"] = 0.0
        updated["border_touch_ratio"] = 0.02
        updated["mean_v"] = 148.0
        updated["mean_lab_b"] = 154.0
        return updated

    monkeypatch.setattr(cv_peak_deblend, "_reclassify_component_frame", fake_reclassify)

    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground,
        feature_image=feature_image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    children = refined.loc[refined["parent_component_id"].astype(str) == "cmp_pair"]
    assert len(children) >= 2


def test_cv_peak_deblend_pairlike_resplit_runs_inside_refine(monkeypatch):
    cfg = AppConfig()
    cfg.detector.cv_post_resplit_enabled = False
    cfg.detector.cv_artifact_resplit_enabled = False
    cfg.detector.cv_large_cluster_resplit_enabled = False
    cfg.detector.cv_large_pupa_resplit_enabled = False
    cfg.detector.cv_global_candidate_supplement_enabled = False
    cfg.detector.cv_patch_supplement_enabled = False
    cfg.detector.cv_global_peak_supplement_enabled = False
    cfg.detector.cv_weak_child_suppress_enabled = False
    cfg.detector.cv_pairlike_resplit_min_area_ratio = 1.20

    feature_image = _blank(shape=(140, 140))
    score = compute_fast_brown_score(feature_image)
    foreground = np.zeros(feature_image.shape[:2], dtype=bool)
    foreground[55:90, 45:95] = True
    blue_mask = np.zeros(feature_image.shape[:2], dtype=np.uint8)

    labeled = pd.DataFrame(
        [
            {
                "component_id": "cmp_ref",
                "parent_component_id": "",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.76,
                "color_score": 0.60,
                "local_contrast": 28.0,
                "mean_v": 150.0,
                "mean_lab_b": 152.0,
                "aspect_ratio": 1.8,
                "eccentricity": 0.86,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.01,
                "area_px": 110.0,
                "major_axis_px": 20.0,
                "minor_axis_px": 8.0,
                "solidity": 0.84,
                "touches_image_border": False,
                "bbox_y0": 10,
                "bbox_x0": 10,
                "bbox_y1": 24,
                "bbox_x1": 30,
                "centroid_y": 17.0,
                "centroid_x": 20.0,
                "mask": np.ones((14, 20), dtype=bool),
                "cv_seed_count": 1,
                "cv_center_only": False,
                "image_height": 140,
                "image_width": 140,
            },
            {
                "component_id": "cmp_pair",
                "parent_component_id": "",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.72,
                "color_score": 0.58,
                "local_contrast": 24.0,
                "mean_v": 148.0,
                "mean_lab_b": 151.0,
                "aspect_ratio": 1.35,
                "eccentricity": 0.72,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.02,
                "area_px": 240.0,
                "major_axis_px": 28.0,
                "minor_axis_px": 19.0,
                "solidity": 0.78,
                "touches_image_border": False,
                "bbox_y0": 55,
                "bbox_x0": 45,
                "bbox_y1": 90,
                "bbox_x1": 95,
                "centroid_y": 72.0,
                "centroid_x": 70.0,
                "mask": np.ones((35, 50), dtype=bool),
                "cv_seed_count": 1,
                "cv_center_only": False,
                "image_height": 140,
                "image_width": 140,
            },
        ]
    )

    def fake_pairlike(row, foreground_mask, single_area, single_minor, image_shape):
        if str(row["component_id"]) != "cmp_pair":
            return None
        return [
            cv_peak_deblend._build_child_row(
                np.ones((10, 12), dtype=bool),
                patch_y0=58,
                patch_x0=48,
                image_shape=image_shape,
                component_id="cmp_pair_pair_01",
                parent_component_id="cmp_pair",
                seed_count=2,
                dense_deblend=False,
                center_only=False,
            ),
            cv_peak_deblend._build_child_row(
                np.ones((10, 12), dtype=bool),
                patch_y0=58,
                patch_x0=70,
                image_shape=image_shape,
                component_id="cmp_pair_pair_02",
                parent_component_id="cmp_pair",
                seed_count=2,
                dense_deblend=False,
                center_only=False,
            ),
        ]

    def fake_reclassify(component_df, *, feature_image, blue_mask, cfg):
        frame = component_df.copy()
        frame["label"] = "pupa"
        frame["is_active"] = True
        frame["confidence"] = 0.70
        frame["color_score"] = 0.58
        frame["local_contrast"] = 24.0
        frame["mean_v"] = 148.0
        frame["mean_lab_b"] = 151.0
        frame["aspect_ratio"] = 1.8
        frame["eccentricity"] = 0.82
        frame["blue_overlap_ratio"] = 0.0
        frame["border_touch_ratio"] = 0.02
        frame["solidity"] = 0.82
        frame["touches_image_border"] = False
        frame["cluster_unresolved"] = False
        return frame

    monkeypatch.setattr(cv_peak_deblend, "_pairlike_split_component", fake_pairlike)
    monkeypatch.setattr(cv_peak_deblend, "_reclassify_component_frame", fake_reclassify)

    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground,
        feature_image=feature_image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    assert "cmp_pair" not in refined["component_id"].tolist()
    assert {"cmp_pair_pair_01", "cmp_pair_pair_02"}.issubset(set(refined["component_id"].tolist()))

def test_cv_peak_deblend_global_candidate_supplement_skips_child_like_rows():
    cfg = AppConfig()
    cfg.detector.cv_post_resplit_enabled = False
    cfg.detector.cv_artifact_resplit_enabled = False
    cfg.detector.cv_large_cluster_resplit_enabled = False
    cfg.detector.cv_large_pupa_resplit_enabled = False
    cfg.detector.cv_patch_supplement_enabled = False
    cfg.detector.cv_global_peak_supplement_enabled = False
    cfg.detector.cv_global_candidate_supplement_enabled = True
    cfg.detector.cv_weak_child_suppress_enabled = False
    cfg.detector.cv_promote_pairlike_artifacts_enabled = False
    cfg.detector.cv_promote_strong_single_artifacts_enabled = False
    cfg.detector.cv_promote_large_single_artifacts_enabled = False

    feature_image = _blank(shape=(160, 160))
    score = compute_fast_brown_score(feature_image)
    foreground = np.zeros(feature_image.shape[:2], dtype=bool)
    blue_mask = np.zeros(feature_image.shape[:2], dtype=np.uint8)

    labeled = pd.DataFrame(
        [
            {
                "component_id": "cmp_ref",
                "parent_component_id": "",
                "label": "pupa",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.74,
                "color_score": 0.60,
                "local_contrast": 26.0,
                "mean_v": 150.0,
                "mean_lab_b": 152.0,
                "aspect_ratio": 1.8,
                "eccentricity": 0.84,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.01,
                "area_px": 110.0,
                "major_axis_px": 20.0,
                "minor_axis_px": 8.0,
                "solidity": 0.86,
                "touches_image_border": False,
                "bbox_y0": 10,
                "bbox_x0": 10,
                "bbox_y1": 24,
                "bbox_x1": 30,
                "centroid_y": 17.0,
                "centroid_x": 20.0,
                "mask": np.ones((14, 20), dtype=bool),
                "cv_seed_count": 1,
                "cv_center_only": False,
                "image_height": 160,
                "image_width": 160,
            },
            {
                "component_id": "cmp_top",
                "parent_component_id": "",
                "label": "artifact",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.70,
                "color_score": 0.72,
                "local_contrast": 26.0,
                "mean_v": 145.0,
                "mean_lab_b": 150.0,
                "aspect_ratio": 1.8,
                "eccentricity": 0.82,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.02,
                "area_px": 95.0,
                "major_axis_px": 18.0,
                "minor_axis_px": 7.0,
                "solidity": 0.82,
                "touches_image_border": False,
                "bbox_y0": 80,
                "bbox_x0": 95,
                "bbox_y1": 92,
                "bbox_x1": 112,
                "centroid_y": 86.0,
                "centroid_x": 103.5,
                "mask": np.ones((12, 17), dtype=bool),
                "cv_seed_count": 1,
                "cv_center_only": False,
                "image_height": 160,
                "image_width": 160,
            },
            {
                "component_id": "cmp_child",
                "parent_component_id": "cmp_parent",
                "label": "artifact",
                "is_active": True,
                "cluster_unresolved": False,
                "confidence": 0.78,
                "color_score": 0.80,
                "local_contrast": 30.0,
                "mean_v": 142.0,
                "mean_lab_b": 149.0,
                "aspect_ratio": 1.9,
                "eccentricity": 0.85,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.02,
                "area_px": 92.0,
                "major_axis_px": 18.0,
                "minor_axis_px": 7.0,
                "solidity": 0.84,
                "touches_image_border": False,
                "bbox_y0": 105,
                "bbox_x0": 92,
                "bbox_y1": 117,
                "bbox_x1": 108,
                "centroid_y": 111.0,
                "centroid_x": 100.0,
                "mask": np.ones((12, 16), dtype=bool),
                "cv_seed_count": 1,
                "cv_center_only": True,
                "image_height": 160,
                "image_width": 160,
            },
        ]
    )

    refined = refine_labeled_candidates(
        labeled,
        score_image=score,
        foreground_mask=foreground,
        feature_image=feature_image,
        blue_mask=blue_mask,
        paper_bounds=None,
        cfg=cfg,
    )

    top_row = refined.loc[refined["component_id"] == "cmp_top"].iloc[0]
    child_row = refined.loc[refined["component_id"] == "cmp_child"].iloc[0]
    assert top_row["label"] == "pupa"
    assert child_row["label"] == "artifact"
