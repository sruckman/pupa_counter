"""Rule-based component filtering and initial labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def rule_classify_components(features_df: pd.DataFrame, cfg: AppConfig = None) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    if features_df.empty:
        return features_df.copy()

    frame = features_df.copy()
    candidate_areas = frame["area_px"].replace([np.inf, -np.inf], np.nan).dropna()
    median_area = float(candidate_areas.median()) if not candidate_areas.empty else 1.0
    median_major = float(frame["major_axis_px"].median()) if not frame.empty else 1.0
    cluster_area_threshold = max(
        cfg.components.min_area_px * cfg.rule_filter.cluster_area_multiplier,
        median_area * cfg.rule_filter.cluster_area_multiplier,
    )

    # Hard ceiling on what can plausibly be a "cluster of pupae". Anything
    # bigger is almost certainly a paper stain, scan artifact, or torn-edge
    # band. We compute it as the more permissive of the median-relative cap
    # and the absolute pixel cap, then cap "is this a real pupa cluster?"
    # below it.
    plausible_cluster_area_max = max(
        median_area * cfg.rule_filter.max_cluster_area_multiplier,
        cfg.components.max_area_px * cfg.rule_filter.max_cluster_area_multiplier / 4.0,
    )

    image_area = None
    if "image_height" in frame.columns and "image_width" in frame.columns:
        try:
            image_area = float(frame["image_height"].iloc[0]) * float(frame["image_width"].iloc[0])
        except (TypeError, ValueError):
            image_area = None

    labels = []
    confidences = []
    raw_scores = []

    for _, row in frame.iterrows():
        area = float(row["area_px"])
        solidity = float(row["solidity"])
        eccentricity = float(row["eccentricity"])
        aspect_ratio = float(row["aspect_ratio"])
        color_score = float(row["color_score"])
        local_contrast = float(row["local_contrast"])
        mean_s = float(row["mean_s"])
        mean_v = float(row["mean_v"])
        mean_lab_b = float(row.get("mean_lab_b", 0.0))
        blue_overlap_ratio = float(row["blue_overlap_ratio"])
        border_touch_ratio = float(row["border_touch_ratio"])
        major_axis = float(row["major_axis_px"])
        extent = float(row["extent"])

        shape_score = np.mean(
            [
                _clip01((aspect_ratio - cfg.rule_filter.min_aspect_ratio) / 2.0),
                _clip01((eccentricity - cfg.rule_filter.min_eccentricity) / 0.45),
                _clip01((solidity - cfg.rule_filter.min_solidity) / 0.45),
            ]
        )
        # Brown pupae have high saturation; grayscale pupae from cleaned PDFs
        # have nearly none. Use the maximum of (saturation, darkness) so each
        # candidate gets credit for whichever signal makes it look pupa-like,
        # rather than averaging them and penalizing desaturated dark blobs.
        sat_score = _clip01(mean_s / 120.0)
        darkness_score = _clip01((255.0 - mean_v) / 200.0)
        appearance_score = np.mean(
            [
                _clip01((color_score - cfg.rule_filter.min_color_score) / 0.6),
                _clip01(local_contrast / 30.0),
                max(sat_score, darkness_score),
            ]
        )
        confidence = float(np.mean([shape_score, appearance_score]))

        # Cluster sanity checks. The yellow paper stain in scan.pdf and the
        # noisy edge bands in scan0001/scan0005 were both being labeled
        # "cluster", picking up an estimate of 8 fake pupae each. Reject
        # them up front so they never reach the fallback estimator.
        cluster_too_big_relative = area > plausible_cluster_area_max
        cluster_too_big_absolute = (
            image_area is not None and area > image_area * cfg.rule_filter.max_cluster_image_fraction
        )
        cluster_at_border = border_touch_ratio > cfg.rule_filter.cluster_max_border_touch_ratio
        cluster_too_bright = mean_v > cfg.rule_filter.cluster_max_mean_v
        # Yellow imprint: per the user's annotation rule, only brown counts.
        # A component is "yellow imprint" if its mean lab b* is high AND
        # its mean V is high (i.e. bright pale yellow). Dark brown eggs
        # may have moderately high b* but low V, so they pass.
        # Guard: components with strong brown color_score are NOT yellow
        # imprints even if their lab_b / V stats land in the yellow zone
        # (common for CV-generated masks that include some background).
        component_too_yellow = mean_lab_b > 145 and mean_v > 140 and color_score < 0.35

        label = "artifact"
        if area < cfg.components.min_area_px:
            label = "artifact"
            confidence = max(confidence, 0.55)
        elif cluster_too_big_relative or cluster_too_big_absolute or cluster_at_border or cluster_too_bright:
            # Implausible "cluster" — almost certainly a stain or edge band.
            label = "artifact"
            confidence = max(confidence, 0.55)
        elif component_too_yellow:
            # Yellow paper imprint, not a real brown egg.
            label = "artifact"
            confidence = max(confidence, 0.55)
        elif blue_overlap_ratio > cfg.rule_filter.max_blue_overlap_ratio:
            label = "uncertain"
            confidence = min(confidence, 0.55)
        elif area > cfg.components.max_area_px or area > cluster_area_threshold:
            label = "cluster"
            confidence = max(confidence, 0.65)
        elif major_axis > median_major * 1.6 and extent < 0.55 and area > median_area * 1.4:
            label = "cluster"
            confidence = max(confidence, 0.60)
        elif mean_v > 245 and mean_s < 15:
            label = "artifact"
            confidence = max(confidence, 0.60)
        elif color_score >= cfg.rule_filter.min_color_score and local_contrast >= cfg.rule_filter.min_local_contrast:
            if (
                solidity >= cfg.rule_filter.min_solidity
                and eccentricity >= cfg.rule_filter.min_eccentricity
                and aspect_ratio >= cfg.rule_filter.min_aspect_ratio
                and border_touch_ratio <= cfg.components.max_border_touch_ratio
            ):
                label = "pupa"
                confidence = max(confidence, 0.55)
            else:
                label = "uncertain"
        elif confidence >= 0.42:
            label = "uncertain"
        else:
            label = "artifact"

        labels.append(label)
        confidences.append(float(np.clip(confidence, 0.0, 0.99)))
        raw_scores.append(float(np.clip((shape_score + appearance_score) / 2.0, 0.0, 1.0)))

    frame["label"] = labels
    frame["confidence"] = confidences
    frame["rule_score"] = raw_scores
    frame["cluster_area_threshold"] = cluster_area_threshold
    return frame
