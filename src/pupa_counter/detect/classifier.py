"""Optional lightweight classifier integration."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig


FEATURE_COLUMNS = [
    "area_px",
    "perimeter_px",
    "major_axis_px",
    "minor_axis_px",
    "eccentricity",
    "solidity",
    "extent",
    "border_touch_ratio",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_h",
    "mean_s",
    "mean_v",
    "mean_lab_l",
    "mean_lab_a",
    "mean_lab_b",
    "blue_overlap_ratio",
    "color_score",
    "local_contrast",
    "gray_std",
    "whitespace_ratio",
    "aspect_ratio",
    "nearest_neighbor_distance",
]


def load_classifier(model_path: Optional[str]):
    if not model_path:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def apply_optional_classifier(features_df: pd.DataFrame, classifier=None, cfg: AppConfig = None) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    if classifier is None or features_df.empty or not cfg.classifier.enabled:
        return features_df.copy()

    frame = features_df.copy()
    available_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if not available_columns:
        return frame

    x_matrix = frame[available_columns].fillna(0.0)
    probabilities = classifier.predict_proba(x_matrix)
    class_names = list(classifier.classes_)
    best_indices = np.argmax(probabilities, axis=1)
    best_scores = probabilities[np.arange(len(frame)), best_indices]
    best_labels = [class_names[index] for index in best_indices]

    updated_labels = []
    updated_confidences = []
    for existing_label, model_label, model_score in zip(
        frame["label"].tolist(), best_labels, best_scores.tolist()
    ):
        if model_score >= cfg.classifier.probability_threshold:
            updated_labels.append(model_label)
        elif cfg.classifier.uncertain_low <= model_score <= cfg.classifier.uncertain_high:
            updated_labels.append("uncertain")
        else:
            updated_labels.append(existing_label)
        updated_confidences.append(float(max(model_score, 0.01)))

    frame["label"] = updated_labels
    frame["confidence"] = np.maximum(frame["confidence"].astype(float).to_numpy(), np.asarray(updated_confidences))
    frame["model_version"] = classifier.__class__.__name__
    return frame
