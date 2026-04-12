"""Component feature extraction."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pupa_counter.detect.brown_mask import compute_brown_score


def _ring_mask(mask: np.ndarray, padding: int = 2) -> np.ndarray:
    padded = cv2.copyMakeBorder(
        mask.astype(np.uint8),
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(padded, kernel, iterations=1)
    ring = (dilated > 0) & ~(padded > 0)
    return ring[padding:-padding, padding:-padding]


def featurize_components(image: np.ndarray, blue_mask: np.ndarray, components_df: pd.DataFrame) -> pd.DataFrame:
    if components_df.empty:
        return components_df.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brown_score = compute_brown_score(image)

    features = components_df.copy()
    mean_rows = []
    centroids = []
    for _, row in features.iterrows():
        y0, y1 = int(row["bbox_y0"]), int(row["bbox_y1"])
        x0, x1 = int(row["bbox_x0"]), int(row["bbox_x1"])
        local_mask = row["mask"]

        patch_rgb = image[y0:y1, x0:x1]
        patch_hsv = hsv[y0:y1, x0:x1]
        patch_lab = lab[y0:y1, x0:x1]
        patch_gray = gray[y0:y1, x0:x1]
        patch_blue = blue_mask[y0:y1, x0:x1] if blue_mask is not None else np.zeros_like(local_mask, dtype=np.uint8)
        patch_brown_score = brown_score[y0:y1, x0:x1]

        pixels_rgb = patch_rgb[local_mask]
        pixels_hsv = patch_hsv[local_mask]
        pixels_lab = patch_lab[local_mask]
        pixels_gray = patch_gray[local_mask]
        pixels_blue = patch_blue[local_mask] > 0
        pixels_brown_score = patch_brown_score[local_mask]

        if pixels_rgb.size == 0:
            mean_rows.append(
                {
                    "mean_r": 0.0,
                    "mean_g": 0.0,
                    "mean_b": 0.0,
                    "mean_h": 0.0,
                    "mean_s": 0.0,
                    "mean_v": 0.0,
                    "mean_lab_l": 0.0,
                    "mean_lab_a": 0.0,
                    "mean_lab_b": 0.0,
                    "blue_overlap_ratio": 0.0,
                    "color_score": 0.0,
                    "local_contrast": 0.0,
                    "gray_std": 0.0,
                    "whitespace_ratio": 1.0,
                    "aspect_ratio": 0.0,
                }
            )
            centroids.append((float(row["centroid_y"]), float(row["centroid_x"])))
            continue

        ring = _ring_mask(local_mask)
        if np.any(ring):
            bg_mean = float(np.mean(patch_gray[ring]))
        else:
            bg_mean = 255.0

        major_axis = float(row["major_axis_px"]) if row["major_axis_px"] else 0.0
        minor_axis = float(row["minor_axis_px"]) if row["minor_axis_px"] else 0.0
        aspect_ratio = major_axis / max(minor_axis, 1.0)

        mean_rows.append(
            {
                "mean_r": float(np.mean(pixels_rgb[:, 0])),
                "mean_g": float(np.mean(pixels_rgb[:, 1])),
                "mean_b": float(np.mean(pixels_rgb[:, 2])),
                "mean_h": float(np.mean(pixels_hsv[:, 0])),
                "mean_s": float(np.mean(pixels_hsv[:, 1])),
                "mean_v": float(np.mean(pixels_hsv[:, 2])),
                "mean_lab_l": float(np.mean(pixels_lab[:, 0])),
                "mean_lab_a": float(np.mean(pixels_lab[:, 1])),
                "mean_lab_b": float(np.mean(pixels_lab[:, 2])),
                "blue_overlap_ratio": float(np.mean(pixels_blue.astype(np.float32))),
                "color_score": float(np.mean(pixels_brown_score)),
                "local_contrast": float(bg_mean - np.mean(pixels_gray)),
                "gray_std": float(np.std(pixels_gray)),
                "whitespace_ratio": float(1.0 - (row["area_px"] / max(row["bbox_area_px"], 1.0))),
                "aspect_ratio": float(aspect_ratio),
            }
        )
        centroids.append((float(row["centroid_y"]), float(row["centroid_x"])))

    mean_df = pd.DataFrame(mean_rows)
    for column in mean_df.columns:
        features[column] = mean_df[column].values

    if len(centroids) > 1:
        points = np.asarray(centroids, dtype=np.float32)
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        features["nearest_neighbor_distance"] = distances[:, 1]
    else:
        features["nearest_neighbor_distance"] = math.inf
    return features
