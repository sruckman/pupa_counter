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
    frame["detector_source"] = "cellpose"
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
