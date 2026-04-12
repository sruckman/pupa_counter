"""Count evaluation metrics."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def evaluate_counts(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> Dict[str, float]:
    if pred_df.empty or gold_df.empty:
        return {}
    join_key = "image_id" if "image_id" in gold_df.columns else "source_path"
    merged = pred_df.merge(gold_df, on=join_key, suffixes=("_pred", "_gold"))
    if merged.empty:
        return {}

    true_middle = merged["true_middle"] if "true_middle" in merged.columns else merged["n_middle_gold"]
    pred_middle = merged["n_middle_pred"] if "n_middle_pred" in merged.columns else merged["n_middle"]
    true_total = merged["true_total"] if "true_total" in merged.columns else merged.get("n_pupa_final_gold", pred_middle)
    pred_total = merged["n_pupa_final_pred"] if "n_pupa_final_pred" in merged.columns else merged.get("n_pupa_final", pred_middle)

    diff_middle = pred_middle.to_numpy(dtype=float) - true_middle.to_numpy(dtype=float)
    diff_total = pred_total.to_numpy(dtype=float) - np.asarray(true_total, dtype=float)

    metrics = {
        "n_images": float(len(merged)),
        "mae_middle": float(np.mean(np.abs(diff_middle))),
        "rmse_middle": float(math.sqrt(np.mean(np.square(diff_middle)))),
        "exact_match_middle": float(np.mean(diff_middle == 0)),
        "signed_error_middle": float(np.mean(diff_middle)),
        "mae_total": float(np.mean(np.abs(diff_total))),
    }
    return metrics
