"""Dual-path merge for annotated Cellpose detections.

The clean-image path is still the primary route because it suppresses blue
annotations well, but annotated PNGs sometimes undercount in crowded regions
after cleaning visually merges neighboring pupae. This module supplements the
clean route with a normalized-image Cellpose pass and only merges extra
detections inside dense local patches.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage import measure

from pupa_counter.config import AppConfig
from pupa_counter.preprocess.paper_region import bbox_fraction_inside_paper_bounds, centroid_inside_paper_bounds


def _bbox_iou(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
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


def _centroid_distance(left: pd.Series, right: pd.Series) -> float:
    return float(
        np.hypot(
            float(left["centroid_x"]) - float(right["centroid_x"]),
            float(left["centroid_y"]) - float(right["centroid_y"]),
        )
    )


def _dense_patch_members(
    frame: pd.DataFrame,
    image_shape,
    median_major: float,
    cfg: AppConfig,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    link_radius = int(
        round(
            max(
                cfg.detector.cellpose_dense_patch_dbscan_eps_px / 2.0,
                median_major * cfg.detector.cellpose_dense_patch_dbscan_eps_scale / 2.0,
            )
        )
    )
    link_radius = max(link_radius, 8)
    occupancy = np.zeros(image_shape, dtype=np.uint8)
    for _, row in frame.iterrows():
        y0, y1 = int(row["bbox_y0"]), int(row["bbox_y1"])
        x0, x1 = int(row["bbox_x0"]), int(row["bbox_x1"])
        patch = occupancy[y0:y1, x0:x1]
        patch[row["mask"].astype(bool)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * link_radius + 1, 2 * link_radius + 1))
    linked = cv2.dilate(occupancy, kernel, iterations=1) > 0
    labeled = measure.label(linked.astype(np.uint8), connectivity=2)

    results = []
    for region_id in range(1, int(labeled.max()) + 1):
        ys, xs = np.where(labeled == region_id)
        if ys.size == 0:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
        member_mask = []
        for _, row in frame.iterrows():
            cy = int(round(float(row["centroid_y"])))
            cx = int(round(float(row["centroid_x"])))
            cy = max(0, min(cy, labeled.shape[0] - 1))
            cx = max(0, min(cx, labeled.shape[1] - 1))
            member_mask.append(bool(labeled[cy, cx] == region_id))
        member_indexes = np.flatnonzero(np.asarray(member_mask, dtype=bool))
        if len(member_indexes) >= cfg.detector.cellpose_annotated_dual_path_min_instances:
            results.append((member_indexes, box))
    return results


def _rows_in_patch(frame: pd.DataFrame, patch_box: Tuple[int, int, int, int]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    x0, y0, x1, y1 = patch_box
    centers_inside = (
        (frame["centroid_x"].astype(float) >= x0)
        & (frame["centroid_x"].astype(float) <= x1)
        & (frame["centroid_y"].astype(float) >= y0)
        & (frame["centroid_y"].astype(float) <= y1)
    )
    return frame.loc[centers_inside].copy()


def _matched_to_any(row: pd.Series, reference_df: pd.DataFrame, cfg: AppConfig) -> bool:
    if reference_df.empty:
        return False
    distance_threshold = max(
        8.0,
        float(row.get("major_axis_px", 0.0) or 0.0) * cfg.detector.cellpose_annotated_dual_path_match_distance_scale,
    )
    row_box = (int(row["bbox_x0"]), int(row["bbox_y0"]), int(row["bbox_x1"]), int(row["bbox_y1"]))
    for _, ref_row in reference_df.iterrows():
        ref_box = (
            int(ref_row["bbox_x0"]),
            int(ref_row["bbox_y0"]),
            int(ref_row["bbox_x1"]),
            int(ref_row["bbox_y1"]),
        )
        if _bbox_iou(row_box, ref_box) >= cfg.detector.cellpose_annotated_dual_path_match_iou:
            return True
        if _centroid_distance(row, ref_row) <= distance_threshold:
            return True
    return False


def merge_annotated_detection_paths(
    clean_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    *,
    image_shape,
    cfg: AppConfig,
) -> pd.DataFrame:
    if (
        clean_df.empty
        or normalized_df.empty
        or not cfg.detector.cellpose_annotated_dual_path_enabled
    ):
        return clean_df.copy()

    base = clean_df.copy().reset_index(drop=True)
    alt = normalized_df.copy().reset_index(drop=True)
    alt = alt.loc[
        alt["is_active"].astype(bool)
        & alt["label"].isin(["pupa"])
        & (alt["confidence"].astype(float) >= cfg.detector.cellpose_annotated_dual_path_min_confidence)
        & (alt["blue_overlap_ratio"].fillna(0.0).astype(float) <= cfg.detector.cellpose_annotated_dual_path_max_blue_overlap_ratio)
    ].copy()
    if alt.empty:
        return base

    seed = pd.concat([base, alt], ignore_index=True, sort=False)
    if len(seed) < cfg.detector.cellpose_annotated_dual_path_min_instances:
        return base

    median_major = float(np.median(seed["major_axis_px"].astype(float)))
    if median_major <= 0.0:
        return base

    accepted_boxes: List[Tuple[int, int, int, int]] = []
    replacement_rows: List[pd.DataFrame] = []
    drop_indexes = set()

    dense_regions = _dense_patch_members(seed, image_shape, median_major, cfg)
    for cluster_id, (member_indexes, _linked_box) in enumerate(dense_regions, start=1):
        cluster_df = seed.iloc[member_indexes].copy()
        patch_y0 = max(0, int(np.floor(cluster_df["bbox_y0"].astype(float).min() - median_major * cfg.detector.cellpose_annotated_dual_path_padding_scale)))
        patch_x0 = max(0, int(np.floor(cluster_df["bbox_x0"].astype(float).min() - median_major * cfg.detector.cellpose_annotated_dual_path_padding_scale)))
        patch_y1 = min(image_shape[0], int(np.ceil(cluster_df["bbox_y1"].astype(float).max() + median_major * cfg.detector.cellpose_annotated_dual_path_padding_scale)))
        patch_x1 = min(image_shape[1], int(np.ceil(cluster_df["bbox_x1"].astype(float).max() + median_major * cfg.detector.cellpose_annotated_dual_path_padding_scale)))
        patch_box = (patch_x0, patch_y0, patch_x1, patch_y1)

        if any(_bbox_iou(patch_box, other_box) > 0.30 for other_box in accepted_boxes):
            continue

        clean_patch = _rows_in_patch(base, patch_box)
        alt_patch = _rows_in_patch(alt, patch_box)
        if len(clean_patch) < cfg.detector.cellpose_annotated_dual_path_min_instances:
            continue
        if alt_patch.empty:
            continue

        keep_clean_rows: List[pd.Series] = []
        for _, clean_row in clean_patch.iterrows():
            if not _matched_to_any(clean_row, alt_patch, cfg):
                keep_clean_rows.append(clean_row)

        merged_patch_parts: List[pd.DataFrame] = [alt_patch.copy()]
        if keep_clean_rows:
            merged_patch_parts.append(pd.DataFrame(keep_clean_rows))
        merged_patch = pd.concat(merged_patch_parts, ignore_index=True, sort=False)

        existing_count = len(clean_patch)
        merged_count = len(merged_patch)
        if merged_count < existing_count + cfg.detector.cellpose_annotated_dual_path_min_extra_instances:
            continue
        if merged_count > int(np.ceil(existing_count * cfg.detector.cellpose_annotated_dual_path_max_gain_ratio)):
            continue

        alt_patch_ids = set(alt_patch["component_id"].tolist())
        merged_patch["dual_path_selected"] = merged_patch.get("dual_path_selected", False)
        merged_patch["dual_path_selected"] = merged_patch["dual_path_selected"].fillna(False).astype(bool)
        merged_patch.loc[merged_patch["component_id"].isin(alt_patch_ids), "dual_path_selected"] = True
        merged_patch.loc[merged_patch["component_id"].isin(alt_patch_ids), "dual_path_cluster_id"] = int(cluster_id)
        if "detector_source" in merged_patch.columns:
            merged_patch.loc[merged_patch["component_id"].isin(alt_patch_ids), "detector_source"] = (
                merged_patch.loc[merged_patch["component_id"].isin(alt_patch_ids), "detector_source"]
                .astype(str)
                .map(lambda value: "normalized_%s" % value)
            )

        replacement_rows.append(merged_patch)
        drop_indexes.update(clean_patch.index.tolist())
        accepted_boxes.append(patch_box)

    if not replacement_rows:
        return base

    kept = base.drop(index=sorted(drop_indexes)).copy()
    if "dual_path_selected" not in kept.columns:
        kept["dual_path_selected"] = False
    kept["dual_path_selected"] = kept["dual_path_selected"].fillna(False).astype(bool)
    merged = pd.concat([kept] + replacement_rows, ignore_index=True, sort=False)
    if "dual_path_selected" not in merged.columns:
        merged["dual_path_selected"] = False
    merged["dual_path_selected"] = merged["dual_path_selected"].fillna(False).astype(bool)
    return merged.reset_index(drop=True)


def merge_annotated_pair_rescue(
    primary_df: pd.DataFrame,
    classical_df: pd.DataFrame,
    *,
    image_shape,
    cfg: AppConfig,
    paper_bounds=None,
) -> pd.DataFrame:
    if (
        primary_df.empty and classical_df.empty
        or not cfg.detector.cellpose_annotated_pair_rescue_enabled
    ):
        return primary_df.copy()

    base = primary_df.copy().reset_index(drop=True)
    raw_split_children = classical_df.loc[
        classical_df["is_active"].astype(bool) & classical_df["split_from_cluster"].fillna(False).astype(bool)
    ].copy()
    raw_group_sizes = raw_split_children.groupby("parent_component_id", dropna=True, sort=False).size().to_dict()
    split_children = raw_split_children.loc[
        raw_split_children["label"].isin(["pupa", "uncertain"])
        & (raw_split_children["confidence"].astype(float) >= cfg.detector.cellpose_annotated_pair_rescue_min_confidence)
        & (raw_split_children["blue_overlap_ratio"].fillna(0.0).astype(float) <= cfg.detector.cellpose_annotated_pair_rescue_max_blue_overlap_ratio)
    ].copy()
    if split_children.empty:
        return base

    # Pair-rescue children are expected to be smaller than the merged blob
    # already present in the primary Cellpose output, so using the primary
    # median area here is actively harmful: it filters out the very touching
    # pairs we are trying to recover. Anchor the scale to the split children
    # themselves and only use the primary lower quartile as a gentle upper
    # cap so we still reject tiny fragments and giant stain pieces.
    child_area_median = float(split_children["area_px"].astype(float).median())
    child_major_median = float(split_children["major_axis_px"].astype(float).median())
    if child_area_median <= 0.0 or child_major_median <= 0.0:
        return base
    base_area_q25 = float(base["area_px"].astype(float).quantile(0.25)) if not base.empty else child_area_median
    base_major_q25 = float(base["major_axis_px"].astype(float).quantile(0.25)) if not base.empty else child_major_median
    reference_area = min(child_area_median, base_area_q25) if base_area_q25 > 0.0 else child_area_median
    reference_major = min(child_major_median, base_major_q25) if base_major_q25 > 0.0 else child_major_median

    split_children = split_children.loc[
        (split_children["area_px"].astype(float) >= reference_area * cfg.detector.cellpose_annotated_pair_rescue_min_area_ratio)
        & (split_children["area_px"].astype(float) <= reference_area * cfg.detector.cellpose_annotated_pair_rescue_max_area_ratio)
    ].copy()
    if split_children.empty:
        return base

    accepted_boxes: List[Tuple[int, int, int, int]] = []
    replacement_rows: List[pd.DataFrame] = []
    drop_indexes = set()

    grouped = split_children.groupby("parent_component_id", dropna=True, sort=False)
    for cluster_id, (parent_component_id, child_df) in enumerate(grouped, start=1):
        if not isinstance(parent_component_id, str):
            continue
        raw_group_size = int(raw_group_sizes.get(parent_component_id, len(child_df)))
        if raw_group_size < cfg.detector.cellpose_annotated_pair_rescue_min_children:
            continue
        if raw_group_size > cfg.detector.cellpose_annotated_pair_rescue_max_children:
            continue
        if len(child_df) > 2 and not child_df["label"].isin(["pupa"]).any():
            continue
        if not child_df["label"].isin(["pupa"]).any():
            mean_child_conf = float(child_df["confidence"].astype(float).mean())
            if mean_child_conf < max(
                cfg.detector.cellpose_annotated_pair_rescue_min_confidence + 0.05,
                cfg.counting.min_instance_confidence,
            ):
                continue
        if paper_bounds is not None:
            child_centroids_inside = child_df.apply(
                lambda row: centroid_inside_paper_bounds(
                    row.get("centroid_x", 0.0),
                    row.get("centroid_y", 0.0),
                    paper_bounds,
                ),
                axis=1,
            )
            child_bbox_inside = child_df.apply(
                lambda row: bbox_fraction_inside_paper_bounds(
                    row.get("bbox_x0", 0.0),
                    row.get("bbox_y0", 0.0),
                    row.get("bbox_x1", 0.0),
                    row.get("bbox_y1", 0.0),
                    paper_bounds,
                ),
                axis=1,
            )
            if not child_centroids_inside.all():
                continue
            if (child_bbox_inside.astype(float) < cfg.preprocess.paper_min_bbox_inside_fraction).any():
                continue

        patch_y0 = max(0, int(np.floor(child_df["bbox_y0"].astype(float).min() - reference_major * cfg.detector.cellpose_annotated_pair_rescue_padding_scale)))
        patch_x0 = max(0, int(np.floor(child_df["bbox_x0"].astype(float).min() - reference_major * cfg.detector.cellpose_annotated_pair_rescue_padding_scale)))
        patch_y1 = min(image_shape[0], int(np.ceil(child_df["bbox_y1"].astype(float).max() + reference_major * cfg.detector.cellpose_annotated_pair_rescue_padding_scale)))
        patch_x1 = min(image_shape[1], int(np.ceil(child_df["bbox_x1"].astype(float).max() + reference_major * cfg.detector.cellpose_annotated_pair_rescue_padding_scale)))
        patch_box = (patch_x0, patch_y0, patch_x1, patch_y1)

        if any(_bbox_iou(patch_box, other_box) > 0.30 for other_box in accepted_boxes):
            continue

        base_patch = _rows_in_patch(base, patch_box)
        matched_base_rows: List[pd.Series] = []
        keep_base_rows: List[pd.Series] = []
        for _, base_row in base_patch.iterrows():
            if _matched_to_any(base_row, child_df, cfg):
                matched_base_rows.append(base_row)
            else:
                keep_base_rows.append(base_row)
        matched_base_count = len(matched_base_rows)
        if matched_base_count not in {0, 1}:
            continue
        if matched_base_count == 0:
            mean_child_conf = float(child_df["confidence"].astype(float).mean())
            if mean_child_conf < max(
                cfg.detector.cellpose_annotated_pair_rescue_min_confidence + 0.08,
                cfg.counting.min_instance_confidence,
            ):
                continue

        merged_parts: List[pd.DataFrame] = [child_df.copy()]
        if keep_base_rows:
            merged_parts.append(pd.DataFrame(keep_base_rows))
        merged_patch = pd.concat(merged_parts, ignore_index=True, sort=False)

        existing_count = len(base_patch)
        merged_count = len(merged_patch)
        if existing_count > 0:
            if merged_count <= existing_count:
                continue
            if merged_count > int(np.ceil(existing_count * cfg.detector.cellpose_annotated_pair_rescue_max_gain_ratio)):
                continue

        child_ids = set(child_df["component_id"].tolist())
        merged_patch["pair_rescue_selected"] = merged_patch.get("pair_rescue_selected", False)
        merged_patch["pair_rescue_selected"] = merged_patch["pair_rescue_selected"].fillna(False).astype(bool)
        merged_patch.loc[merged_patch["component_id"].isin(child_ids), "pair_rescue_selected"] = True
        merged_patch.loc[merged_patch["component_id"].isin(child_ids), "pair_rescue_cluster_id"] = int(cluster_id)
        merged_patch.loc[merged_patch["component_id"].isin(child_ids), "label"] = "pupa"
        merged_patch.loc[merged_patch["component_id"].isin(child_ids), "confidence"] = np.maximum(
            merged_patch.loc[merged_patch["component_id"].isin(child_ids), "confidence"].astype(float),
            cfg.counting.min_instance_confidence,
        )
        if "detector_source" not in merged_patch.columns:
            merged_patch["detector_source"] = ""
        child_source = merged_patch.loc[merged_patch["component_id"].isin(child_ids), "detector_source"]
        merged_patch.loc[merged_patch["component_id"].isin(child_ids), "detector_source"] = (
            child_source.where(child_source.astype(str) != "", "classical")
            .fillna("classical")
            .astype(str)
            .map(lambda value: "pair_%s" % value)
        )

        replacement_rows.append(merged_patch)
        drop_indexes.update(base_patch.index.tolist())
        accepted_boxes.append(patch_box)

    if not replacement_rows:
        return base

    kept = base.drop(index=sorted(drop_indexes)).copy()
    if "pair_rescue_selected" not in kept.columns:
        kept["pair_rescue_selected"] = False
    kept["pair_rescue_selected"] = kept["pair_rescue_selected"].fillna(False).astype(bool)
    merged = pd.concat([kept] + replacement_rows, ignore_index=True, sort=False)
    if "pair_rescue_selected" not in merged.columns:
        merged["pair_rescue_selected"] = False
    merged["pair_rescue_selected"] = merged["pair_rescue_selected"].fillna(False).astype(bool)
    return merged.reset_index(drop=True)
