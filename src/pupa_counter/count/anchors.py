"""Anchor selection and band geometry."""

from __future__ import annotations

import pandas as pd

from pupa_counter.types import BandGeometry


def _position_series(instances_df: pd.DataFrame, anchor_mode: str):
    if anchor_mode == "bbox_edge":
        return instances_df["bbox_y0"].astype(float), instances_df["bbox_y1"].astype(float)
    series = instances_df["centroid_y"].astype(float)
    return series, series


def compute_band_geometry(instances_df: pd.DataFrame, anchor_mode: str = "centroid") -> BandGeometry:
    if instances_df.empty:
        raise ValueError("Cannot compute band geometry without instances")
    top_series, bottom_series = _position_series(instances_df, anchor_mode)
    top_y = float(top_series.min())
    bottom_y = float(bottom_series.max())
    span = bottom_y - top_y
    upper_middle_y = top_y + 0.25 * span
    lower_middle_y = top_y + 0.75 * span
    return BandGeometry(
        top_y=top_y,
        bottom_y=bottom_y,
        upper_middle_y=float(upper_middle_y),
        lower_middle_y=float(lower_middle_y),
        anchor_mode=anchor_mode,
    )
