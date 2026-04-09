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


def _draw_instance_label(canvas: np.ndarray, center: tuple[int, int], label: str, color) -> None:
    stats_box_bottom = 125
    x = int(center[0] + 8)
    y = int(center[1] - 8)
    x = max(5, min(x, canvas.shape[1] - 40))
    if y < stats_box_bottom:
        y = int(center[1] + 18)
    y = max(20, min(y, canvas.shape[0] - 5))
    cv2.putText(
        canvas,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def build_overlay(
    image: np.ndarray,
    instances_df: pd.DataFrame,
    geometry: BandGeometry = None,
    flags = None,
    candidate_df: pd.DataFrame = None,
    *,
    show_middle_labels: bool = False,
) -> np.ndarray:
    overlay = image.copy()
    n_top_5pct = 0
    if instances_df is not None and not instances_df.empty:
        if "is_top_5pct" in instances_df.columns:
            n_top_5pct = int(instances_df["is_top_5pct"].astype(bool).sum())
        elif geometry is not None:
            n_top_5pct = int((instances_df["centroid_y"].astype(float) <= float(geometry.upper_five_pct_y)).sum())
        middle_labels = {}
        if show_middle_labels:
            middle_df = instances_df.loc[
                ~instances_df.get("synthetic_instance", pd.Series(False, index=instances_df.index)).astype(bool)
                & (instances_df.get("band", pd.Series("", index=instances_df.index)) == "middle")
            ].copy()
            if not middle_df.empty:
                middle_df = middle_df.sort_values(["centroid_y", "centroid_x"], kind="mergesort").reset_index()
                middle_labels = {
                    int(orig_idx): str(display_idx)
                    for display_idx, orig_idx in enumerate(middle_df["index"].tolist(), start=1)
                }

        for row_idx, row in instances_df.iterrows():
            band = row.get("band", "middle")
            color = BAND_COLORS.get(band, (255, 255, 0))
            _draw_component_contour(overlay, row, color)
            if not bool(row.get("synthetic_instance", False)):
                center = (int(round(row["centroid_x"])), int(round(row["centroid_y"])))
                cv2.circle(overlay, center, 4, color, -1)
                label = middle_labels.get(int(row_idx))
                if label is not None:
                    _draw_instance_label(overlay, center, label, color)

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
        cv2.line(
            overlay,
            (0, int(round(geometry.upper_five_pct_y))),
            (width, int(round(geometry.upper_five_pct_y))),
            (255, 215, 0),
            2,
        )
        cv2.line(overlay, (0, int(round(geometry.upper_middle_y))), (width, int(round(geometry.upper_middle_y))), (0, 180, 255), 3)
        cv2.line(overlay, (0, int(round(geometry.lower_middle_y))), (width, int(round(geometry.lower_middle_y))), (0, 180, 255), 3)
        cv2.line(overlay, (0, int(round(geometry.top_y))), (width, int(round(geometry.top_y))), (255, 0, 255), 1)
        cv2.line(overlay, (0, int(round(geometry.bottom_y))), (width, int(round(geometry.bottom_y))), (255, 0, 255), 1)
        cv2.putText(
            overlay,
            "5%",
            (12, max(20, int(round(geometry.upper_five_pct_y)) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )

    counts_text = "top5%%=%s top=%s middle=%s bottom=%s total=%s" % (
        n_top_5pct,
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
