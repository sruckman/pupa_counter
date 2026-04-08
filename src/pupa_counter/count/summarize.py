"""Count summarization."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.types import CountSummary, ImageRecord


def select_final_instances(candidate_df: pd.DataFrame, cfg: AppConfig = None) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    if candidate_df.empty:
        return candidate_df.copy()
    frame = candidate_df.copy()
    frame["anchor_confidence"] = (
        frame["confidence"].astype(float)
        * (1.0 - frame["blue_overlap_ratio"].fillna(0.0).astype(float))
        * (1.0 - frame["border_touch_ratio"].fillna(0.0).astype(float))
    )
    final_mask = (
        frame["is_active"].astype(bool)
        & frame["label"].isin(["pupa"])
        & (frame["confidence"].astype(float) >= cfg.counting.min_instance_confidence)
    )
    return frame.loc[final_mask].reset_index(drop=True)


def combine_instances_for_counting(actual_instances_df: pd.DataFrame, synthetic_instances_df: pd.DataFrame) -> pd.DataFrame:
    if actual_instances_df.empty and synthetic_instances_df.empty:
        return pd.DataFrame()
    if actual_instances_df.empty:
        return synthetic_instances_df.copy().reset_index(drop=True)
    if synthetic_instances_df.empty:
        return actual_instances_df.copy().reset_index(drop=True)
    return pd.concat([actual_instances_df.copy(), synthetic_instances_df.copy()], ignore_index=True)


def summarize_counts(
    record: ImageRecord,
    instances_df: pd.DataFrame,
    geometry,
    *,
    config_version: str,
    model_version: Optional[str] = None,
    runtime_ms: Optional[float] = None,
    candidate_df: Optional[pd.DataFrame] = None,
    blue_pixel_ratio: Optional[float] = None,
) -> CountSummary:
    candidate_df = candidate_df if candidate_df is not None else pd.DataFrame()
    unresolved_clusters = 0
    if not candidate_df.empty and "cluster_unresolved" in candidate_df.columns:
        unresolved_clusters = int(
            candidate_df.loc[
                candidate_df["is_active"].astype(bool) & candidate_df["cluster_unresolved"].astype(bool)
            ].shape[0]
        )

    n_top = int((instances_df["band"] == "top").sum()) if "band" in instances_df.columns else 0
    n_middle = int((instances_df["band"] == "middle").sum()) if "band" in instances_df.columns else 0
    n_bottom = int((instances_df["band"] == "bottom").sum()) if "band" in instances_df.columns else 0
    mean_confidence = None if instances_df.empty else float(instances_df["confidence"].mean())

    return CountSummary(
        image_id=record.image_id,
        source_path=str(record.source_path),
        split=record.split,
        n_candidates_raw=int(candidate_df.loc[candidate_df["is_active"].astype(bool)].shape[0]) if not candidate_df.empty else 0,
        n_pupa_final=int(len(instances_df)),
        n_top=n_top,
        n_middle=n_middle,
        n_bottom=n_bottom,
        top_y=None if geometry is None else geometry.top_y,
        bottom_y=None if geometry is None else geometry.bottom_y,
        upper_middle_y=None if geometry is None else geometry.upper_middle_y,
        lower_middle_y=None if geometry is None else geometry.lower_middle_y,
        mean_confidence=mean_confidence,
        unresolved_clusters=unresolved_clusters,
        blue_pixel_ratio=blue_pixel_ratio,
        needs_review=False,
        review_reason="",
        config_version=config_version,
        model_version=model_version,
        runtime_ms=runtime_ms,
    )
