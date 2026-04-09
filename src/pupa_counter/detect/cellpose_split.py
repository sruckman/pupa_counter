"""Conservative post-split for large Cellpose masks.

Cellpose handles most single pupae well, but on high-quality annotated sheets
it still occasionally returns one fat mask for two touching pupae. We only
touch a very small subset of detections:

- clearly larger than the image-level median pupa area
- too round / too low-eccentricity to look like a single pupa
- showing multiple distance-map peaks inside the mask

If any of those conditions fail, the original Cellpose mask is kept as-is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation

from pupa_counter.config import AppConfig
from pupa_counter.detect.components import build_component_row


def split_large_cellpose_instances(
    components_df: pd.DataFrame,
    image_shape,
    *,
    source_type: str,
    cfg: AppConfig,
    guide_image: np.ndarray | None = None,
    restrict_to_dense_patch: bool = False,
) -> pd.DataFrame:
    if components_df.empty or not cfg.detector.cellpose_overlap_split_enabled:
        return components_df.copy()
    if source_type not in {"annotated_png", "clean_png"}:
        return components_df.copy()

    median_area = float(components_df["area_px"].median()) if "area_px" in components_df.columns else 0.0
    if median_area <= 0:
        return components_df.copy()

    rows = []
    child_rows = []
    min_child_area_px = max(
        float(cfg.components.min_area_px),
        median_area * cfg.detector.cellpose_overlap_split_min_child_area_ratio,
    )

    def _normalized(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        masked = values[mask]
        if masked.size == 0:
            return np.zeros_like(values, dtype=np.float32)
        low = float(masked.min())
        high = float(masked.max())
        if high <= low:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values.astype(np.float32) - low) / (high - low), 0.0, 1.0)

    def _combo_regions(local_mask: np.ndarray, row_dict: dict):
        if (
            source_type != "annotated_png"
            or guide_image is None
            or not cfg.detector.cellpose_overlap_split_combo_enabled
        ):
            return []

        area_px = float(row_dict.get("area_px", 0.0) or 0.0)
        if area_px < median_area * float(cfg.detector.cellpose_overlap_split_combo_area_ratio):
            return []
        if bool(row_dict.get("touches_image_border", False)):
            return []
        if float(row_dict.get("border_touch_ratio", 0.0) or 0.0) > float(
            cfg.detector.cellpose_overlap_split_combo_max_border_touch_ratio
        ):
            return []

        y0 = int(row_dict["bbox_y0"])
        x0 = int(row_dict["bbox_x0"])
        y1 = int(row_dict["bbox_y1"])
        x1 = int(row_dict["bbox_x1"])
        if y1 <= y0 or x1 <= x0:
            return []
        local_patch = guide_image[y0:y1, x0:x1]
        if local_patch.shape[:2] != local_mask.shape:
            return []

        rgb = local_patch.astype(np.float32)
        brownness = rgb[:, :, 0] - 0.6 * rgb[:, :, 2] - 0.4 * rgb[:, :, 1]
        brownness = _normalized(brownness, local_mask)
        distance = ndi.distance_transform_edt(local_mask)
        distance = _normalized(distance, local_mask)
        brown_weight = float(cfg.detector.cellpose_overlap_split_combo_brown_weight)
        combo = ((1.0 - brown_weight) * distance + brown_weight * brownness) * local_mask.astype(np.float32)

        peaks = feature.peak_local_max(
            combo,
            min_distance=cfg.detector.cellpose_overlap_split_combo_peak_min_distance,
            threshold_abs=cfg.detector.cellpose_overlap_split_combo_peak_abs_threshold,
            labels=local_mask.astype(np.uint8),
        )
        combo_peak_cap = max(cfg.detector.cellpose_overlap_split_max_children * 3, 8)
        if peaks.shape[0] < 2:
            return []
        if peaks.shape[0] > combo_peak_cap:
            return []
        if peaks.shape[0] > cfg.detector.cellpose_overlap_split_max_children:
            peaks = peaks[:combo_peak_cap]

        markers = np.zeros(local_mask.shape, dtype=np.int32)
        for marker_index, (peak_row, peak_col) in enumerate(peaks, start=1):
            markers[peak_row, peak_col] = marker_index
        markers, _ = ndi.label(markers > 0)
        labels = segmentation.watershed(-combo, markers, mask=local_mask)
        combo_min_child_area_px = max(
            float(cfg.components.min_area_px),
            area_px * float(cfg.detector.cellpose_overlap_split_combo_min_child_area_ratio),
        )
        regions = [
            region
            for region in measure.regionprops(labels)
            if float(region.area) >= combo_min_child_area_px
        ]
        if len(regions) < 2 or len(regions) > cfg.detector.cellpose_overlap_split_max_children:
            return []
        child_areas = [float(region.area) for region in regions]
        if min(child_areas) <= 0.0:
            return []
        if max(child_areas) / min(child_areas) > float(
            cfg.detector.cellpose_overlap_split_combo_max_child_area_ratio
        ):
            return []
        return regions

    for _, row in components_df.iterrows():
        row_dict = row.to_dict()
        if restrict_to_dense_patch and not bool(row_dict.get("dense_patch_refined", False)):
            rows.append(row_dict)
            continue
        area_px = float(row_dict.get("area_px", 0.0) or 0.0)
        aspect_ratio = float(row_dict.get("aspect_ratio", 0.0) or 0.0)
        eccentricity = float(row_dict.get("eccentricity", 1.0) or 1.0)

        split_area_ratio = float(cfg.detector.cellpose_overlap_split_area_ratio)
        if source_type == "annotated_png":
            split_area_ratio = min(split_area_ratio, 1.45)

        area_gate = area_px >= median_area * split_area_ratio
        if source_type == "annotated_png" and cfg.detector.cellpose_overlap_split_annotated_ignore_shape:
            eligible = area_gate
        else:
            eligible = (
                area_gate
                and aspect_ratio <= cfg.detector.cellpose_overlap_split_max_aspect_ratio
                and eccentricity <= cfg.detector.cellpose_overlap_split_max_eccentricity
            )

        local_mask = row_dict["mask"].astype(bool)
        valid_regions = []
        split_labels = None
        if eligible:
            distance = ndi.distance_transform_edt(local_mask)
            peaks = feature.peak_local_max(
                distance,
                min_distance=cfg.detector.cellpose_overlap_split_peak_min_distance,
                threshold_abs=cfg.detector.cellpose_overlap_split_peak_abs_threshold,
                labels=local_mask.astype(np.uint8),
            )
            if 2 <= peaks.shape[0] <= cfg.detector.cellpose_overlap_split_max_children:
                markers = np.zeros(local_mask.shape, dtype=np.int32)
                for marker_index, (peak_row, peak_col) in enumerate(peaks, start=1):
                    markers[peak_row, peak_col] = marker_index
                markers, _ = ndi.label(markers > 0)
                split_labels = segmentation.watershed(-distance, markers, mask=local_mask)
                valid_regions = [
                    region
                    for region in measure.regionprops(split_labels)
                    if float(region.area) >= min_child_area_px
                ]

        if len(valid_regions) < 2 or len(valid_regions) > cfg.detector.cellpose_overlap_split_max_children:
            valid_regions = _combo_regions(local_mask, row_dict)
            if valid_regions:
                rgb = guide_image[int(row_dict["bbox_y0"]):int(row_dict["bbox_y1"]), int(row_dict["bbox_x0"]):int(row_dict["bbox_x1"])].astype(np.float32)
                brownness = rgb[:, :, 0] - 0.6 * rgb[:, :, 2] - 0.4 * rgb[:, :, 1]
                brownness = _normalized(brownness, local_mask)
                distance = _normalized(ndi.distance_transform_edt(local_mask), local_mask)
                brown_weight = float(cfg.detector.cellpose_overlap_split_combo_brown_weight)
                combo = ((1.0 - brown_weight) * distance + brown_weight * brownness) * local_mask.astype(np.float32)
                combo_peaks = feature.peak_local_max(
                    combo,
                    min_distance=cfg.detector.cellpose_overlap_split_combo_peak_min_distance,
                    threshold_abs=cfg.detector.cellpose_overlap_split_combo_peak_abs_threshold,
                    labels=local_mask.astype(np.uint8),
                )
                combo_peak_cap = max(cfg.detector.cellpose_overlap_split_max_children * 3, 8)
                if combo_peaks.shape[0] > combo_peak_cap:
                    combo_peaks = combo_peaks[:combo_peak_cap]
                markers = np.zeros(local_mask.shape, dtype=np.int32)
                for marker_index, (peak_row, peak_col) in enumerate(combo_peaks, start=1):
                    markers[peak_row, peak_col] = marker_index
                markers, _ = ndi.label(markers > 0)
                split_labels = segmentation.watershed(-combo, markers, mask=local_mask)
        if len(valid_regions) < 2 or len(valid_regions) > cfg.detector.cellpose_overlap_split_max_children:
            rows.append(row_dict)
            continue

        row_dict["is_active"] = False
        row_dict["cellpose_overlap_split_applied"] = True
        row_dict["split_children_count"] = len(valid_regions)
        rows.append(row_dict)

        y0 = int(row_dict["bbox_y0"])
        x0 = int(row_dict["bbox_x0"])
        for child_index, region in enumerate(valid_regions, start=1):
            child_mask = split_labels == region.label
            child_id = "%s_split_%02d" % (row_dict["component_id"], child_index)
            child_row = build_component_row(
                child_mask,
                y0,
                x0,
                image_shape,
                child_id,
                parent_component_id=row_dict["component_id"],
                split_from_cluster=False,
            )
            child_row["cellpose_overlap_child"] = True
            child_rows.append(child_row)

    if child_rows:
        combined = pd.concat([pd.DataFrame(rows), pd.DataFrame(child_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(rows)
    if combined.empty:
        return components_df.copy()
    return combined.sort_values(["component_id"]).reset_index(drop=True)
