"""Watershed-based splitting for merged pupa clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation

from pupa_counter.config import AppConfig
from pupa_counter.detect.components import build_component_row
from pupa_counter.detect.features import featurize_components
from pupa_counter.detect.rule_filter import rule_classify_components


def split_cluster_candidates(
    image: np.ndarray,
    candidate_df: pd.DataFrame,
    blue_mask: np.ndarray = None,
    cfg: AppConfig = None,
) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    if candidate_df.empty or not cfg.split_clusters.enabled:
        return candidate_df.copy()

    rows = []
    child_rows = []
    for _, row in candidate_df.iterrows():
        row_dict = row.to_dict()
        if row_dict.get("label") != "cluster" or not row_dict.get("is_active", True):
            rows.append(row_dict)
            continue

        local_mask = row_dict["mask"].astype(bool)
        distance = ndi.distance_transform_edt(local_mask)
        peaks = feature.peak_local_max(
            distance,
            min_distance=cfg.split_clusters.distance_peak_min_distance,
            threshold_abs=cfg.split_clusters.peak_abs_threshold,
            labels=local_mask.astype(np.uint8),
        )
        if peaks.shape[0] <= 1:
            row_dict["cluster_unresolved"] = True
            rows.append(row_dict)
            continue

        markers = np.zeros(local_mask.shape, dtype=np.int32)
        for marker_index, (peak_row, peak_col) in enumerate(peaks, start=1):
            markers[peak_row, peak_col] = marker_index
        markers, _ = ndi.label(markers > 0)
        labels = segmentation.watershed(-distance, markers, mask=local_mask)

        valid_regions = [
            region
            for region in measure.regionprops(labels)
            if region.area >= cfg.split_clusters.watershed_min_child_area_px
        ]
        if len(valid_regions) < 2 or len(valid_regions) > cfg.split_clusters.max_children_per_cluster:
            row_dict["cluster_unresolved"] = True
            rows.append(row_dict)
            continue

        row_dict["is_active"] = False
        row_dict["cluster_unresolved"] = False
        row_dict["split_children_count"] = len(valid_regions)
        rows.append(row_dict)

        y0 = int(row_dict["bbox_y0"])
        x0 = int(row_dict["bbox_x0"])
        for child_index, region in enumerate(valid_regions, start=1):
            child_mask = labels == region.label
            child_id = "%s_child_%02d" % (row_dict["component_id"], child_index)
            child_rows.append(
                build_component_row(
                    child_mask,
                    y0,
                    x0,
                    image.shape[:2],
                    child_id,
                    parent_component_id=row_dict["component_id"],
                    split_from_cluster=True,
                )
            )

    if child_rows:
        child_df = pd.DataFrame(child_rows)
        child_features = featurize_components(
            image,
            blue_mask if blue_mask is not None else np.zeros(image.shape[:2], dtype=np.uint8),
            child_df,
        )
        child_features = rule_classify_components(child_features, cfg)
        for _, row in child_features.iterrows():
            row_dict = row.to_dict()
            if row_dict.get("label") == "cluster":
                row_dict["label"] = "uncertain"
                row_dict["cluster_unresolved"] = True
            rows.append(row_dict)

    combined = pd.DataFrame(rows)
    if combined.empty:
        return candidate_df.copy()
    return combined.sort_values(["component_id"]).reset_index(drop=True)
