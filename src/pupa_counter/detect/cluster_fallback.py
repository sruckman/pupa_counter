"""Fallback strategies for unresolved clusters."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _build_cluster_fallback_mask(
    frame: pd.DataFrame,
    cfg: AppConfig,
    reference_area: float,
    reference_major: float,
) -> pd.Series:
    area_ratio = frame["area_px"].astype(float) / max(reference_area, 1.0)
    major_ratio = frame["major_axis_px"].astype(float) / max(reference_major, 1.0)
    frame["cluster_fallback_area_ratio"] = area_ratio
    frame["cluster_fallback_major_ratio"] = major_ratio

    eligible = (
        frame["is_active"].map(_bool_value)
        & frame["cluster_unresolved"].map(_bool_value)
        & (frame["label"].astype(str) == "cluster")
        & (area_ratio >= cfg.cluster_fallback.min_area_ratio)
        & (major_ratio >= cfg.cluster_fallback.min_major_axis_ratio)
    )
    if cfg.cluster_fallback.exclude_split_children and "split_from_cluster" in frame.columns:
        eligible &= ~frame["split_from_cluster"].map(_bool_value)
    return eligible


def attach_cluster_count_estimates(candidate_df: pd.DataFrame, cfg: AppConfig = None) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    if candidate_df.empty or not cfg.cluster_fallback.enabled:
        return candidate_df.copy()

    frame = candidate_df.copy()
    reference = frame.loc[
        frame["is_active"].astype(bool)
        & frame["label"].isin(["pupa"])
        & (frame["confidence"].astype(float) >= cfg.cluster_fallback.reference_confidence_min)
    ]
    if reference.empty:
        reference = frame.loc[frame["label"].isin(["pupa"])]
    if reference.empty:
        reference = frame.loc[
            frame["is_active"].map(_bool_value)
            & ~frame["label"].isin(["artifact", "cluster"])
            & (~frame["split_from_cluster"].map(_bool_value) if "split_from_cluster" in frame.columns else True)
        ]
    reference_area = float(reference["area_px"].median()) if not reference.empty else np.nan
    reference_major = float(reference["major_axis_px"].median()) if not reference.empty else np.nan
    if not np.isfinite(reference_area) or not np.isfinite(reference_major):
        frame["estimated_cluster_count"] = 0
        frame["cluster_count_source"] = ""
        frame["cluster_fallback_eligible"] = False
        frame["cluster_fallback_area_ratio"] = 0.0
        frame["cluster_fallback_major_ratio"] = 0.0
        return frame
    reference_area = max(reference_area, 1.0)
    reference_major = max(reference_major, 1.0)
    eligible_mask = _build_cluster_fallback_mask(frame, cfg, reference_area, reference_major)
    frame["cluster_fallback_eligible"] = eligible_mask

    estimates = []
    for _, row in frame.iterrows():
        if not bool(row.get("cluster_fallback_eligible", False)):
            estimates.append(0)
            continue
        area_term = float(row["area_px"]) / reference_area
        major_term = float(row["major_axis_px"]) / reference_major
        combined = cfg.cluster_fallback.area_weight * area_term + cfg.cluster_fallback.major_axis_weight * major_term
        estimate = int(round(combined))
        estimate = max(cfg.cluster_fallback.min_estimated_instances, estimate)
        estimate = min(cfg.cluster_fallback.max_estimated_instances, estimate)
        estimates.append(estimate)

    frame["estimated_cluster_count"] = estimates
    frame["cluster_count_source"] = np.where(
        frame["estimated_cluster_count"].astype(int) > 0,
        "estimate",
        "",
    )
    return frame


def apply_vision_cluster_counts(
    candidate_df: pd.DataFrame,
    vision_df: Optional[pd.DataFrame],
    cfg: AppConfig = None,
) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    if candidate_df.empty or vision_df is None or vision_df.empty:
        return candidate_df.copy()
    frame = candidate_df.merge(vision_df, on="component_id", how="left")
    if not cfg.vision_fallback.use_for_counts:
        return frame

    count_values = []
    count_sources = []
    for _, row in frame.iterrows():
        source = row.get("cluster_count_source", "")
        count_value = int(row.get("estimated_cluster_count", 0) or 0)
        vision_count = row.get("vision_cluster_count")
        vision_confidence = row.get("vision_cluster_confidence")
        if pd.notna(vision_count) and pd.notna(vision_confidence) and float(vision_confidence) >= cfg.vision_fallback.confidence_threshold:
            count_value = int(vision_count)
            source = "vision"
        count_values.append(count_value)
        count_sources.append(source)
    frame["estimated_cluster_count"] = count_values
    frame["cluster_count_source"] = count_sources
    return frame


def synthesize_cluster_instances(candidate_df: pd.DataFrame, cfg: AppConfig = None) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    if candidate_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in candidate_df.iterrows():
        source = str(row.get("cluster_count_source", "") or "")
        can_use_local_estimate = cfg.cluster_fallback.use_for_counts and bool(row.get("cluster_fallback_eligible", False))
        can_use_vision = cfg.vision_fallback.use_for_counts and source == "vision"
        if not (can_use_local_estimate or can_use_vision):
            continue
        estimated_count = int(row.get("estimated_cluster_count", 0) or 0)
        if estimated_count <= 0:
            continue
        for index in range(estimated_count):
            row_dict = row.to_dict()
            row_dict["component_id"] = "%s_est_%02d" % (row["component_id"], index + 1)
            row_dict["synthetic_instance"] = True
            row_dict["synthetic_parent_component_id"] = row["component_id"]
            row_dict["confidence"] = float(max(float(row.get("confidence", 0.0)) * 0.8, 0.35))
            row_dict["label"] = "pupa"
            row_dict["mask"] = np.zeros((1, 1), dtype=bool)
            rows.append(row_dict)
    return pd.DataFrame(rows)
