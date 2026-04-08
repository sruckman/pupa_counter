"""Utilities to compare counts between runs."""

from __future__ import annotations

import pandas as pd


def compare_runs(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty or previous_df.empty:
        return pd.DataFrame()
    merged = current_df.merge(
        previous_df[["image_id", "n_middle", "n_pupa_final"]],
        on="image_id",
        suffixes=("_current", "_previous"),
    )
    if merged.empty:
        return merged
    merged["delta_middle"] = merged["n_middle_current"] - merged["n_middle_previous"]
    merged["delta_total"] = merged["n_pupa_final_current"] - merged["n_pupa_final_previous"]
    return merged
