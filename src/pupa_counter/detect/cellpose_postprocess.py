"""Lightweight post-processing for Cellpose detections.

The current Cellpose backend is strong on dense annotated PNGs, but it needs
some source-aware guardrails:

- On clean images, extremely bright / low-color masks are usually paper noise.
- On clean PNG exports, Cellpose can miss a handful of tiny but otherwise
  strong candidates that the classical detector still finds.

This module keeps the learned detector as the primary source of truth and only
applies conservative cleanup / supplementation around the edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.preprocess.paper_region import bbox_fraction_inside_paper_bounds, centroid_inside_paper_bounds


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def calibrate_cellpose_detections(
    features_df: pd.DataFrame,
    *,
    source_type: str,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Stamp labels/confidences onto Cellpose instances.

    We deliberately do *not* run the full classical rule filter here. Its
    appearance heuristics were tuned for brown-mask blobs and over-reject
    valid Cellpose instances on dense annotated sheets. Instead we:

    - keep a smooth confidence score for anchors / QA;
    - apply one clean-only artifact guard for extremely bright, weak-color
      masks that are unlikely to be real pupae.
    """
    if features_df.empty:
        return features_df.copy()

    frame = features_df.copy()
    labels = []
    confidences = []
    raw_scores = []
    clean_source = source_type in {"clean_png", "clean_pdf"}

    for _, row in frame.iterrows():
        area_px = float(row["area_px"])
        solidity = float(row["solidity"])
        eccentricity = float(row["eccentricity"])
        aspect_ratio = float(row["aspect_ratio"])
        color_score = float(row["color_score"])
        local_contrast = float(row["local_contrast"])
        mean_s = float(row["mean_s"])
        mean_v = float(row["mean_v"])
        shape_score = np.mean(
            [
                _clip01((aspect_ratio - 1.15) / 1.2),
                _clip01((eccentricity - 0.45) / 0.45),
                _clip01((solidity - 0.60) / 0.35),
            ]
        )
        sat_score = _clip01(mean_s / 120.0)
        darkness_score = _clip01((255.0 - mean_v) / 180.0)
        appearance_score = np.mean(
            [
                _clip01((color_score - 0.18) / 0.45),
                _clip01((local_contrast + 10.0) / 45.0),
                max(sat_score, darkness_score),
            ]
        )
        raw_confidence = float(np.mean([shape_score, appearance_score]))
        confidence = max(raw_confidence, 0.58 if clean_source else 0.72)
        label = "pupa"

        if mean_v > 235.0 and color_score < 0.15:
            label = "artifact"
            confidence = max(confidence, 0.70)
        if clean_source and (
            (mean_v > cfg.detector.clean_filter_max_mean_v)
            and (color_score < cfg.detector.clean_filter_max_color_score)
        ):
            label = "artifact"
            confidence = max(confidence, 0.55)

        labels.append(label)
        confidences.append(float(np.clip(confidence, 0.0, 0.99)))
        raw_scores.append(float(np.clip(raw_confidence, 0.0, 1.0)))

    frame["label"] = labels
    frame["confidence"] = confidences
    frame["rule_score"] = raw_scores
    frame["cluster_unresolved"] = False
    frame["cluster_area_threshold"] = 0.0
    parent_ids = frame["parent_component_id"] if "parent_component_id" in frame.columns else pd.Series([None] * len(frame))
    dense_patch_refined = frame["dense_patch_refined"] if "dense_patch_refined" in frame.columns else pd.Series([False] * len(frame))
    detector_source = np.where(
        parent_ids.notna(),
        "cellpose_split",
        np.where(dense_patch_refined.astype(bool), "cellpose_dense_patch", "cellpose"),
    )
    frame["detector_source"] = detector_source
    return frame


def prune_annotated_false_positives(
    candidate_df: pd.DataFrame,
    *,
    source_type: str,
    cfg: AppConfig,
    paper_bounds=None,
) -> pd.DataFrame:
    """Remove obvious annotated-sheet false positives after recall rescue.

    We intentionally run this *after* the dense-patch / dual-path / pair-rescue
    logic. Those stages benefit from permissive seeds; applying the same
    artifact rules too early suppresses the very crowded regions we are trying
    to recover. Here we only demote candidates that look decisively wrong:

    - blue-dot / blue-line remnants,
    - tiny edge slivers hugging the image border,
    - very bright low-color paper marks,
    - classical add-on fragments still touching the page edge.
    """
    if candidate_df.empty or source_type != "annotated_png":
        return candidate_df.copy()

    frame = candidate_df.copy()
    is_pupa = frame["label"].isin(["pupa"]) & frame["is_active"].astype(bool)
    blue_overlap = frame["blue_overlap_ratio"].fillna(0.0).astype(float)
    mean_v = frame["mean_v"].fillna(255.0).astype(float)
    color_score = frame["color_score"].fillna(0.0).astype(float)
    local_contrast = frame["local_contrast"].fillna(0.0).astype(float)
    area_px = frame["area_px"].fillna(0.0).astype(float)
    border_touch_ratio = frame["border_touch_ratio"].fillna(0.0).astype(float)
    touches_image_border = frame["touches_image_border"].fillna(False).astype(bool)
    detector_source = frame["detector_source"].fillna("").astype(str)
    centroid_inside = frame.apply(
        lambda row: centroid_inside_paper_bounds(
            row.get("centroid_x", 0.0),
            row.get("centroid_y", 0.0),
            paper_bounds,
        ),
        axis=1,
    )
    bbox_inside_fraction = frame.apply(
        lambda row: bbox_fraction_inside_paper_bounds(
            row.get("bbox_x0", 0.0),
            row.get("bbox_y0", 0.0),
            row.get("bbox_x1", 0.0),
            row.get("bbox_y1", 0.0),
            paper_bounds,
        ),
        axis=1,
    )

    blue_artifact = is_pupa & (
        (blue_overlap >= 0.18)
        | (
            (blue_overlap >= 0.10)
            & (
                (mean_v >= 170.0)
                | (color_score <= 0.20)
                | (local_contrast <= 20.0)
                | (area_px <= 220.0)
            )
        )
    )
    outside_paper_artifact = is_pupa & (
        ~centroid_inside.astype(bool)
        | (bbox_inside_fraction.astype(float) < 0.50)
        | (
            touches_image_border
            & (bbox_inside_fraction.astype(float) < cfg.preprocess.paper_min_bbox_inside_fraction)
        )
    )
    border_artifact = is_pupa & touches_image_border & (
        (~centroid_inside.astype(bool))
        | (bbox_inside_fraction.astype(float) < cfg.preprocess.paper_min_bbox_inside_fraction)
    ) & (
        (detector_source == "annotated_classical_addon")
        | ((border_touch_ratio >= 0.55) & (area_px <= 180.0))
    )
    bright_artifact = is_pupa & (mean_v >= 235.0) & (color_score <= 0.15)
    stain_artifact = (
        is_pupa
        & centroid_inside.astype(bool)
        & (area_px <= 220.0)
        & (mean_v >= 150.0)
        & (color_score <= 0.22)
        & (local_contrast <= 18.0)
    )

    artifact_mask = blue_artifact | outside_paper_artifact | border_artifact | bright_artifact | stain_artifact
    if not artifact_mask.any():
        return frame

    frame.loc[artifact_mask, "label"] = "artifact"
    frame.loc[artifact_mask, "confidence"] = np.maximum(
        frame.loc[artifact_mask, "confidence"].astype(float),
        0.70,
    )
    return frame


def _bbox_iou(left: pd.Series, right: pd.Series) -> float:
    left_x0, left_y0, left_x1, left_y1 = (
        int(left["bbox_x0"]),
        int(left["bbox_y0"]),
        int(left["bbox_x1"]),
        int(left["bbox_y1"]),
    )
    right_x0, right_y0, right_x1, right_y1 = (
        int(right["bbox_x0"]),
        int(right["bbox_y0"]),
        int(right["bbox_x1"]),
        int(right["bbox_y1"]),
    )
    inter_x0 = max(left_x0, right_x0)
    inter_y0 = max(left_y0, right_y0)
    inter_x1 = min(left_x1, right_x1)
    inter_y1 = min(left_y1, right_y1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
    left_area = float((left_x1 - left_x0) * (left_y1 - left_y0))
    right_area = float((right_x1 - right_x0) * (right_y1 - right_y0))
    union = left_area + right_area - inter_area
    return 0.0 if union <= 0.0 else inter_area / union


def build_clean_png_supplement(
    cellpose_df: pd.DataFrame,
    classical_df: pd.DataFrame,
    *,
    source_type: str,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Return a small set of strong classical-only detections for clean PNGs.

    This is intentionally narrow:

    - only active on ``clean_png``;
    - only when the unmatched classical set is small relative to the current
      Cellpose output;
    - only for small, dark, strong classical candidates.
    """
    if source_type != "clean_png" or not cfg.detector.clean_png_supplement_enabled:
        return pd.DataFrame()
    if classical_df.empty:
        return pd.DataFrame()

    strong = classical_df.loc[
        classical_df["is_active"].astype(bool)
        & classical_df["label"].isin(["pupa"])
        & (classical_df["confidence"].astype(float) >= cfg.counting.min_instance_confidence)
    ].copy()
    if strong.empty:
        return pd.DataFrame()

    unmatched_rows = []
    for _, classical_row in strong.iterrows():
        best_iou = 0.0
        for _, cellpose_row in cellpose_df.iterrows():
            best_iou = max(best_iou, _bbox_iou(classical_row, cellpose_row))
            if best_iou >= 0.30:
                break
        if best_iou < 0.30:
            unmatched_rows.append(classical_row)

    if not unmatched_rows:
        return pd.DataFrame()

    unmatched = pd.DataFrame(unmatched_rows)
    unmatched_ratio = float(len(unmatched) / max(len(cellpose_df), 1))
    if unmatched_ratio > cfg.detector.clean_png_supplement_max_unmatched_ratio:
        return pd.DataFrame()
    if len(cellpose_df) > cfg.detector.clean_png_supplement_max_cellpose_count:
        return pd.DataFrame()

    keep_mask = (
        (unmatched["area_px"].astype(float) >= cfg.detector.clean_png_supplement_min_area_px)
        & (unmatched["area_px"].astype(float) <= cfg.detector.clean_png_supplement_max_area_px)
        & (unmatched["mean_v"].astype(float) <= cfg.detector.clean_png_supplement_max_mean_v)
        & (unmatched["color_score"].astype(float) >= cfg.detector.clean_png_supplement_min_color_score)
        & (unmatched["local_contrast"].astype(float) >= cfg.detector.clean_png_supplement_min_local_contrast)
    )
    supplement = unmatched.loc[keep_mask].copy()
    if supplement.empty:
        return supplement

    supplement["cluster_unresolved"] = False
    supplement["cluster_area_threshold"] = supplement["cluster_area_threshold"].fillna(0.0)
    supplement["detector_source"] = "classical_addon"
    return supplement.reset_index(drop=True)


def build_annotated_png_supplement(
    cellpose_df: pd.DataFrame,
    classical_df: pd.DataFrame,
    *,
    source_type: str,
    cfg: AppConfig,
    paper_bounds=None,
) -> pd.DataFrame:
    """Add a small number of strong classical-only candidates on annotated PNGs.

    This is a recall-first patch for the failure mode the user highlighted:
    two touching pupae can disappear entirely from the Cellpose route, but the
    classical split path still produces plausible small brown components. We
    only add unmatched classical detections that look pupa-like and keep the
    cap low so the supplement cannot dominate the image.
    """
    if source_type != "annotated_png" or not cfg.detector.cellpose_annotated_png_supplement_enabled:
        return pd.DataFrame()
    if classical_df.empty:
        return pd.DataFrame()

    strong = classical_df.loc[
        classical_df["is_active"].astype(bool)
        & classical_df["label"].isin(["pupa"])
        & (classical_df["confidence"].astype(float) >= cfg.counting.min_instance_confidence)
    ].copy()
    if strong.empty:
        return pd.DataFrame()

    unmatched_rows = []
    for _, classical_row in strong.iterrows():
        best_iou = 0.0
        for _, cellpose_row in cellpose_df.iterrows():
            best_iou = max(best_iou, _bbox_iou(classical_row, cellpose_row))
            if best_iou >= 0.25:
                break
        if best_iou < 0.25:
            unmatched_rows.append(classical_row)

    if not unmatched_rows:
        return pd.DataFrame()

    unmatched = pd.DataFrame(unmatched_rows)
    unmatched_ratio = float(len(unmatched) / max(len(cellpose_df), 1))
    if unmatched_ratio > cfg.detector.cellpose_annotated_png_supplement_max_unmatched_ratio:
        return pd.DataFrame()

    keep_mask = (
        (unmatched["area_px"].astype(float) >= cfg.detector.cellpose_annotated_png_supplement_min_area_px)
        & (unmatched["area_px"].astype(float) <= cfg.detector.cellpose_annotated_png_supplement_max_area_px)
        & (unmatched["mean_v"].astype(float) <= cfg.detector.cellpose_annotated_png_supplement_max_mean_v)
        & (unmatched["color_score"].astype(float) >= cfg.detector.cellpose_annotated_png_supplement_min_color_score)
        & (unmatched["local_contrast"].astype(float) >= cfg.detector.cellpose_annotated_png_supplement_min_local_contrast)
        & (
            unmatched["blue_overlap_ratio"].fillna(0.0).astype(float)
            <= cfg.detector.cellpose_annotated_png_supplement_max_blue_overlap_ratio
        )
        & (
            unmatched["border_touch_ratio"].fillna(0.0).astype(float)
            <= cfg.detector.cellpose_annotated_png_supplement_max_border_touch_ratio
        )
    )
    if "touches_image_border" in unmatched.columns:
        keep_mask &= ~unmatched["touches_image_border"].fillna(False).astype(bool)
    if paper_bounds is not None:
        left, top, right, bottom = paper_bounds
        centroid_inside = (
            (unmatched["centroid_x"].astype(float) >= left)
            & (unmatched["centroid_x"].astype(float) <= right)
            & (unmatched["centroid_y"].astype(float) >= top)
            & (unmatched["centroid_y"].astype(float) <= bottom)
        )
        bbox_inside_fraction = unmatched.apply(
            lambda row: bbox_fraction_inside_paper_bounds(
                row.get("bbox_x0", 0.0),
                row.get("bbox_y0", 0.0),
                row.get("bbox_x1", 0.0),
                row.get("bbox_y1", 0.0),
                paper_bounds,
            ),
            axis=1,
        )
        keep_mask &= centroid_inside
        keep_mask &= bbox_inside_fraction.astype(float) >= cfg.preprocess.paper_min_bbox_inside_fraction
    supplement = unmatched.loc[keep_mask].copy()
    if supplement.empty:
        return supplement

    if len(supplement) > cfg.detector.cellpose_annotated_png_supplement_max_added_count:
        supplement = supplement.sort_values(
            ["confidence", "color_score", "local_contrast"],
            ascending=[False, False, False],
        ).head(cfg.detector.cellpose_annotated_png_supplement_max_added_count)

    supplement["cluster_unresolved"] = False
    supplement["cluster_area_threshold"] = supplement["cluster_area_threshold"].fillna(0.0)
    supplement["detector_source"] = "annotated_classical_addon"
    return supplement.reset_index(drop=True)
