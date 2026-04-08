"""Overlay rendering for QA."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from pupa_counter.types import BandGeometry, ReviewFlag


BAND_COLORS = {
    "top": (255, 140, 0),
    "middle": (40, 170, 40),
    "bottom": (220, 60, 60),
}


def _draw_component_contour(canvas: np.ndarray, row: pd.Series, color) -> None:
    if bool(row.get("synthetic_instance", False)):
        return
    local_mask = row["mask"].astype(np.uint8) * 255
    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x0 = int(row["bbox_x0"])
    y0 = int(row["bbox_y0"])
    for contour in contours:
        shifted = contour + np.array([[[x0, y0]]], dtype=contour.dtype)
        cv2.drawContours(canvas, [shifted], -1, color, 2)


def build_overlay(
    image: np.ndarray,
    instances_df: pd.DataFrame,
    geometry: BandGeometry = None,
    flags = None,
    candidate_df: pd.DataFrame = None,
) -> np.ndarray:
    overlay = image.copy()
    if instances_df is not None and not instances_df.empty:
        for _, row in instances_df.iterrows():
            band = row.get("band", "middle")
            color = BAND_COLORS.get(band, (255, 255, 0))
            _draw_component_contour(overlay, row, color)
            if not bool(row.get("synthetic_instance", False)):
                center = (int(round(row["centroid_x"])), int(round(row["centroid_y"])))
                cv2.circle(overlay, center, 4, color, -1)

    if candidate_df is not None and not candidate_df.empty:
        hard_clusters = candidate_df.loc[
            candidate_df["is_active"].astype(bool) & candidate_df["cluster_unresolved"].astype(bool)
        ]
        for _, row in hard_clusters.iterrows():
            x0 = int(row["bbox_x0"])
            y0 = int(row["bbox_y0"])
            x1 = int(row["bbox_x1"])
            y1 = int(row["bbox_y1"])
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
            if int(row.get("estimated_cluster_count", 0) or 0) > 0:
                label = "%s x%d" % (
                    row.get("cluster_count_source", "est") or "est",
                    int(row.get("estimated_cluster_count", 0)),
                )
                cv2.putText(
                    overlay,
                    label,
                    (x0, max(20, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

    if geometry is not None:
        width = overlay.shape[1]
        cv2.line(overlay, (0, int(round(geometry.upper_middle_y))), (width, int(round(geometry.upper_middle_y))), (0, 180, 255), 3)
        cv2.line(overlay, (0, int(round(geometry.lower_middle_y))), (width, int(round(geometry.lower_middle_y))), (0, 180, 255), 3)
        cv2.line(overlay, (0, int(round(geometry.top_y))), (width, int(round(geometry.top_y))), (255, 0, 255), 1)
        cv2.line(overlay, (0, int(round(geometry.bottom_y))), (width, int(round(geometry.bottom_y))), (255, 0, 255), 1)

    counts_text = "top=%s middle=%s bottom=%s total=%s" % (
        int((instances_df["band"] == "top").sum()) if instances_df is not None and "band" in instances_df.columns else 0,
        int((instances_df["band"] == "middle").sum()) if instances_df is not None and "band" in instances_df.columns else 0,
        int((instances_df["band"] == "bottom").sum()) if instances_df is not None and "band" in instances_df.columns else 0,
        0 if instances_df is None else len(instances_df),
    )
    cv2.rectangle(overlay, (10, 10), (min(overlay.shape[1] - 10, 900), 120), (255, 255, 255), -1)
    cv2.putText(overlay, counts_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    if flags:
        flag_text = "; ".join(flag.code for flag in flags[:4])
        cv2.putText(overlay, flag_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return overlay
