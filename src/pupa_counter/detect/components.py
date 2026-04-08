"""Connected component extraction."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from skimage import measure

from pupa_counter.config import AppConfig


def build_component_row(
    local_mask: np.ndarray,
    offset_row: int,
    offset_col: int,
    image_shape,
    component_id: str,
    parent_component_id: Optional[str] = None,
    split_from_cluster: bool = False,
) -> Dict[str, object]:
    labeled = measure.label(local_mask.astype(bool), connectivity=2)
    regions = measure.regionprops(labeled)
    if not regions:
        raise ValueError("local_mask does not contain a component")
    region = max(regions, key=lambda item: item.area)
    min_row, min_col, max_row, max_col = region.bbox
    component_mask = labeled[min_row:max_row, min_col:max_col] == region.label
    abs_min_row = offset_row + min_row
    abs_min_col = offset_col + min_col
    abs_max_row = offset_row + max_row
    abs_max_col = offset_col + max_col
    area = float(region.area)
    border_hits = 0.0
    coords = region.coords
    if coords.size:
        border_hits = float(
            np.sum(coords[:, 0] == 0)
            + np.sum(coords[:, 1] == 0)
            + np.sum(coords[:, 0] == local_mask.shape[0] - 1)
            + np.sum(coords[:, 1] == local_mask.shape[1] - 1)
        )
    return {
        "component_id": component_id,
        "parent_component_id": parent_component_id,
        "split_from_cluster": split_from_cluster,
        "is_active": True,
        "cluster_unresolved": False,
        "bbox_y0": abs_min_row,
        "bbox_x0": abs_min_col,
        "bbox_y1": abs_max_row,
        "bbox_x1": abs_max_col,
        "centroid_y": float(offset_row + region.centroid[0]),
        "centroid_x": float(offset_col + region.centroid[1]),
        "area_px": area,
        "bbox_area_px": float(component_mask.shape[0] * component_mask.shape[1]),
        "perimeter_px": float(region.perimeter),
        "major_axis_px": float(region.major_axis_length),
        "minor_axis_px": float(region.minor_axis_length),
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "extent": float(region.extent),
        "orientation_rad": float(region.orientation),
        "border_touch_ratio": float(border_hits / area) if area else 0.0,
        "touches_image_border": bool(
            abs_min_row <= 0
            or abs_min_col <= 0
            or abs_max_row >= image_shape[0]
            or abs_max_col >= image_shape[1]
        ),
        "mask": component_mask.astype(bool),
    }


def extract_components(mask: np.ndarray, cfg: AppConfig = None) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    labeled = measure.label(mask > 0, connectivity=2)
    rows = []
    for region in measure.regionprops(labeled):
        component_id = "cc_%05d" % region.label
        min_row, min_col, max_row, max_col = region.bbox
        local_mask = labeled[min_row:max_row, min_col:max_col] == region.label
        row = build_component_row(local_mask, min_row, min_col, mask.shape, component_id)
        # Stamp image dimensions on every row so downstream filters can apply
        # absolute (image-relative) thresholds without re-plumbing the shape.
        row["image_height"] = int(mask.shape[0])
        row["image_width"] = int(mask.shape[1])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["component_id", "mask", "image_height", "image_width"])
    frame = pd.DataFrame(rows)
    return frame.sort_values("component_id").reset_index(drop=True)
