"""Dense-patch refinement for Cellpose detections.

The main full-image Cellpose pass works well for isolated pupae, but some
annotated PNGs contain very dense local colonies where touching pupae lose
their separating gaps after normalization / cleaning. For those local regions
we re-run Cellpose on the higher-fidelity normalized crop with a slightly
smaller diameter and only accept the refined patch when it adds a modest
number of plausible instances.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage import measure

from pupa_counter.config import AppConfig


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
        if len(member_indexes) >= cfg.detector.cellpose_dense_patch_min_instances:
            results.append((member_indexes, box))
    return results


def refine_dense_cellpose_patches(
    image_rgb: np.ndarray,
    components_df: pd.DataFrame,
    *,
    source_type: str,
    cfg: AppConfig,
    detect_fn: Optional[Callable[..., pd.DataFrame]] = None,
) -> pd.DataFrame:
    if (
        components_df.empty
        or not cfg.detector.cellpose_dense_patch_refine_enabled
        or source_type != "annotated_png"
    ):
        return components_df.copy()

    if detect_fn is None:
        from pupa_counter.detect.cellpose_backend import detect_instances as detect_fn  # noqa: WPS433

    frame = components_df.copy().reset_index(drop=True)
    if len(frame) < cfg.detector.cellpose_dense_patch_min_instances:
        return frame

    median_major = float(np.median(frame["major_axis_px"].astype(float)))
    median_area = float(np.median(frame["area_px"].astype(float)))
    if median_major <= 0.0 or median_area <= 0.0:
        return frame

    accepted_boxes: List[Tuple[int, int, int, int]] = []
    replacement_rows: List[pd.DataFrame] = []
    drop_indexes = set()

    dense_regions = _dense_patch_members(frame, image_rgb.shape[:2], median_major, cfg)
    for cluster_id, (member_indexes, _linked_box) in enumerate(dense_regions, start=1):
        cluster_df = frame.iloc[member_indexes].copy()
        patch_y0 = max(0, int(np.floor(cluster_df["bbox_y0"].astype(float).min() - median_major * cfg.detector.cellpose_dense_patch_padding_scale)))
        patch_x0 = max(0, int(np.floor(cluster_df["bbox_x0"].astype(float).min() - median_major * cfg.detector.cellpose_dense_patch_padding_scale)))
        patch_y1 = min(image_rgb.shape[0], int(np.ceil(cluster_df["bbox_y1"].astype(float).max() + median_major * cfg.detector.cellpose_dense_patch_padding_scale)))
        patch_x1 = min(image_rgb.shape[1], int(np.ceil(cluster_df["bbox_x1"].astype(float).max() + median_major * cfg.detector.cellpose_dense_patch_padding_scale)))
        patch_box = (patch_x0, patch_y0, patch_x1, patch_y1)

        if any(_bbox_iou(patch_box, other_box) > 0.30 for other_box in accepted_boxes):
            continue

        patch_area = max((patch_y1 - patch_y0) * (patch_x1 - patch_x0), 1)
        fill_ratio = float(cluster_df["area_px"].astype(float).sum() / patch_area)
        if fill_ratio < cfg.detector.cellpose_dense_patch_min_fill_ratio:
            continue

        patch = image_rgb[patch_y0:patch_y1, patch_x0:patch_x1]
        if patch.size == 0:
            continue

        dense_diameter = max(
            cfg.detector.cellpose_dense_patch_min_diameter_px,
            float(np.median(cluster_df["major_axis_px"].astype(float))) * cfg.detector.cellpose_dense_patch_diameter_scale,
        )
        refined_df = detect_fn(
            patch,
            cfg,
            diameter=dense_diameter,
            max_side_px=int(max(patch.shape[:2])),
            flow_threshold=cfg.detector.cellpose_dense_patch_flow_threshold,
            cellprob_threshold=cfg.detector.cellpose_dense_patch_cellprob_threshold,
            component_prefix="cpd%02d" % cluster_id,
            offset_row=patch_y0,
            offset_col=patch_x0,
            global_image_shape=image_rgb.shape[:2],
        )
        if refined_df.empty:
            continue

        refined_df = refined_df.loc[
            (refined_df["area_px"].astype(float) >= median_area * cfg.detector.cellpose_dense_patch_min_area_ratio)
            & (refined_df["area_px"].astype(float) <= median_area * cfg.detector.cellpose_dense_patch_max_area_ratio)
        ].copy()
        if refined_df.empty:
            continue

        existing_count = len(cluster_df)
        refined_count = len(refined_df)
        if refined_count < existing_count + cfg.detector.cellpose_dense_patch_min_extra_instances:
            continue
        if refined_count > int(np.ceil(existing_count * cfg.detector.cellpose_dense_patch_max_gain_ratio)):
            continue

        refined_df["dense_patch_refined"] = True
        refined_df["dense_patch_cluster_id"] = int(cluster_id)
        replacement_rows.append(refined_df)
        drop_indexes.update(cluster_df.index.tolist())
        accepted_boxes.append(patch_box)

    if not replacement_rows:
        return frame

    kept = frame.drop(index=sorted(drop_indexes)).copy()
    kept["dense_patch_refined"] = kept.get("dense_patch_refined", False)
    refined = pd.concat(replacement_rows, ignore_index=True)
    merged = pd.concat([kept, refined], ignore_index=True, sort=False)
    if "dense_patch_refined" not in merged.columns:
        merged["dense_patch_refined"] = False
    merged["dense_patch_refined"] = merged["dense_patch_refined"].fillna(False).astype(bool)
    return merged.reset_index(drop=True)
