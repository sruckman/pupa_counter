"""Fast counting-first pure CV backend with local dense deblend."""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from pupa_counter.config import AppConfig
from pupa_counter.detect.components import build_component_row, extract_components
from pupa_counter.detect.features import featurize_components
from pupa_counter.detect.rule_filter import rule_classify_components
from pupa_counter.preprocess.paper_region import PaperBounds, bbox_fraction_inside_paper_bounds


_KERNEL_CACHE: dict[tuple[int, int, tuple[int, ...]], list[tuple[int, np.ndarray]]] = {}


def _paper_mask(shape: tuple[int, int], bounds: PaperBounds | None) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if bounds is None:
        return mask
    left, top, right, bottom = bounds
    mask[:] = False
    mask[max(0, top):max(0, bottom), max(0, left):max(0, right)] = True
    return mask


def _scale_bounds(bounds: PaperBounds | None, scale: float, shape: tuple[int, int]) -> PaperBounds | None:
    if bounds is None or scale == 1.0:
        return bounds
    left, top, right, bottom = bounds
    height, width = shape
    return (
        max(0, int(round(left * scale))),
        max(0, int(round(top * scale))),
        min(width, int(round(right * scale))),
        min(height, int(round(bottom * scale))),
    )


def _localize_paper_bounds(
    bounds: PaperBounds | None,
    *,
    patch_y0: int,
    patch_x0: int,
    patch_shape: tuple[int, int],
) -> PaperBounds | None:
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    patch_h, patch_w = patch_shape
    local_left = max(0, int(round(left - patch_x0)))
    local_top = max(0, int(round(top - patch_y0)))
    local_right = min(patch_w, int(round(right - patch_x0)))
    local_bottom = min(patch_h, int(round(bottom - patch_y0)))
    if local_right <= local_left or local_bottom <= local_top:
        return None
    return (local_left, local_top, local_right, local_bottom)


def _maybe_downscale(
    image: np.ndarray,
    blue_mask: np.ndarray | None,
    paper_bounds: PaperBounds | None,
    max_side_px: int,
) -> tuple[np.ndarray, np.ndarray | None, PaperBounds | None, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side_px:
        return image, blue_mask, paper_bounds, 1.0
    scale = max_side_px / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    resized_blue = None
    if blue_mask is not None and blue_mask.size:
        resized_blue = cv2.resize(
            blue_mask.astype(np.uint8),
            new_size,
            interpolation=cv2.INTER_NEAREST,
        )
    resized_bounds = _scale_bounds(paper_bounds, scale, resized_image.shape[:2])
    return resized_image, resized_blue, resized_bounds, scale


def compute_fast_brown_score(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    red = image_float[:, :, 0]
    green = image_float[:, :, 1]
    blue = image_float[:, :, 2]
    value = image_float.max(axis=2)
    score = np.clip(red - 0.55 * green - 0.45 * blue, 0.0, 1.0)
    score += 0.35 * np.clip(0.95 - value, 0.0, 1.0)
    score += 0.10 * np.clip(red - 0.8 * green + 0.05, 0.0, 1.0)
    return np.clip(score / 1.45, 0.0, 1.0).astype(np.float32)


def _binary_foreground(
    image: np.ndarray,
    blue_mask: np.ndarray | None,
    paper_bounds: PaperBounds | None,
    cfg: AppConfig,
) -> tuple[np.ndarray, np.ndarray]:
    score = compute_fast_brown_score(image)
    mask = score >= cfg.detector.cv_fg_brown_thresh
    if blue_mask is not None and blue_mask.size:
        mask &= blue_mask == 0
    mask &= _paper_mask(image.shape[:2], paper_bounds)
    raw = (mask.astype(np.uint8) * 255)
    open_kernel = np.ones((cfg.detector.cv_open_kernel, cfg.detector.cv_open_kernel), dtype=np.uint8)
    close_kernel = np.ones((cfg.detector.cv_close_kernel, cfg.detector.cv_close_kernel), dtype=np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_kernel)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
    return score, raw > 0


def build_foreground_mask(
    image: np.ndarray,
    cfg: AppConfig,
    *,
    blue_mask: np.ndarray | None = None,
    paper_bounds: PaperBounds | None = None,
) -> np.ndarray:
    """Return the fast-CV foreground mask for debugging / saved intermediates."""
    _score, mask = _binary_foreground(image, blue_mask, paper_bounds, cfg)
    return (mask.astype(np.uint8) * 255)


def _estimate_single_scale(components_df: pd.DataFrame, cfg: AppConfig) -> tuple[float, float, float]:
    if components_df.empty:
        return 120.0, 18.0, 8.0

    frame = components_df.copy()
    frame = frame.loc[
        (frame["area_px"].astype(float) >= cfg.detector.cv_min_area_px)
        & (~frame["touches_image_border"].astype(bool))
    ].copy()
    if frame.empty:
        return 120.0, 18.0, 8.0

    robust = frame.loc[
        (frame["solidity"].astype(float) >= 0.78)
        & (frame["eccentricity"].astype(float) >= 0.72)
    ].copy()
    pool = robust if len(robust) >= 4 else frame
    if len(pool) >= 6:
        low = pool["area_px"].astype(float).quantile(0.20)
        high = pool["area_px"].astype(float).quantile(0.80)
        trimmed = pool.loc[(pool["area_px"].astype(float) >= low) & (pool["area_px"].astype(float) <= high)]
        if len(trimmed) >= 4:
            pool = trimmed

    single_area = max(float(pool["area_px"].astype(float).median()), float(cfg.detector.cv_min_area_px))
    single_major = max(float(pool["major_axis_px"].astype(float).median()), 10.0)
    single_minor = max(float(pool["minor_axis_px"].astype(float).median()), 5.0)
    return single_area, single_major, single_minor


def _oriented_filter_bank(
    major_axis_px: float,
    minor_axis_px: float,
    angles_deg: Sequence[int],
) -> list[tuple[int, np.ndarray]]:
    key = (int(round(major_axis_px)), int(round(minor_axis_px)), tuple(int(angle) for angle in angles_deg))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    half_size = max(5, int(math.ceil(max(major_axis_px, minor_axis_px) * 1.25)))
    ys, xs = np.mgrid[-half_size : half_size + 1, -half_size : half_size + 1].astype(np.float32)
    sigma_major = max(1.5, major_axis_px / 4.0)
    sigma_minor = max(1.0, minor_axis_px / 4.0)
    filters: list[tuple[int, np.ndarray]] = []
    for angle in angles_deg:
        theta = np.deg2rad(float(angle))
        xr = xs * np.cos(theta) + ys * np.sin(theta)
        yr = -xs * np.sin(theta) + ys * np.cos(theta)
        kernel = np.exp(-0.5 * ((xr / sigma_major) ** 2 + (yr / sigma_minor) ** 2))
        kernel = kernel - (kernel.mean() * 0.90)
        norm = float(np.sum(np.abs(kernel)))
        if norm > 0.0:
            kernel = kernel / norm
        filters.append((int(angle), kernel.astype(np.float32)))
    _KERNEL_CACHE[key] = filters
    return filters


def _matched_response(
    score_patch: np.ndarray,
    major_axis_px: float,
    minor_axis_px: float,
    angles_deg: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    filters = _oriented_filter_bank(major_axis_px, minor_axis_px, angles_deg)
    responses = []
    for _, kernel in filters:
        responses.append(
            cv2.filter2D(
                score_patch.astype(np.float32),
                ddepth=cv2.CV_32F,
                kernel=kernel,
                borderType=cv2.BORDER_REPLICATE,
            )
        )
    stacked = np.stack(responses, axis=0)
    angle_indexes = np.argmax(stacked, axis=0).astype(np.int32)
    return np.max(stacked, axis=0), angle_indexes


def _ranked_peaks(
    response: np.ndarray,
    allowed_mask: np.ndarray,
    single_minor: float,
    cfg: AppConfig,
) -> np.ndarray:
    if not np.any(allowed_mask):
        return np.empty((0, 2), dtype=int)
    values = response[allowed_mask]
    if values.size == 0:
        return np.empty((0, 2), dtype=int)
    threshold_abs = float(np.percentile(values, cfg.detector.cv_local_peak_threshold_percentile))
    min_distance = max(2, int(round(single_minor * cfg.detector.cv_peak_min_distance_minor_scale)))
    peaks = feature.peak_local_max(
        response,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        labels=allowed_mask.astype(np.uint8),
    )
    if peaks.size == 0:
        return peaks
    scores = response[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(scores)[::-1]
    return peaks[order]


def _multithresh_peaks(
    response: np.ndarray,
    allowed_mask: np.ndarray,
    single_minor: float,
    cfg: AppConfig,
) -> np.ndarray:
    if not cfg.detector.cv_enable_multithresh_fallback or not np.any(allowed_mask):
        return np.empty((0, 2), dtype=int)
    values = response[allowed_mask]
    if values.size == 0:
        return np.empty((0, 2), dtype=int)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.empty((0, 2), dtype=int)
    min_distance = max(2, int(round(single_minor * cfg.detector.cv_peak_min_distance_minor_scale)))
    contrast_floor = lo + cfg.detector.cv_multithresh_min_contrast * (hi - lo)
    seeds: list[tuple[int, int, float]] = []
    thresholds = np.linspace(contrast_floor, hi, cfg.detector.cv_multithresh_levels)
    for threshold in thresholds:
        binary = (response >= threshold) & allowed_mask
        labeled = measure.label(binary, connectivity=2)
        for region in measure.regionprops(labeled, intensity_image=response):
            coords = region.coords
            if coords.size == 0:
                continue
            coord_scores = response[coords[:, 0], coords[:, 1]]
            peak_index = int(np.argmax(coord_scores))
            global_row = int(coords[peak_index, 0])
            global_col = int(coords[peak_index, 1])
            seeds.append((global_row, global_col, float(response[global_row, global_col])))
    if not seeds:
        return np.empty((0, 2), dtype=int)
    seeds.sort(key=lambda item: item[2], reverse=True)
    accepted: list[tuple[int, int]] = []
    for row, col, _score in seeds:
        if any((row - prev_row) ** 2 + (col - prev_col) ** 2 < min_distance**2 for prev_row, prev_col in accepted):
            continue
        accepted.append((row, col))
    if not accepted:
        return np.empty((0, 2), dtype=int)
    return np.asarray(accepted, dtype=int)


def _distance_peaks(
    allowed_mask: np.ndarray,
    single_minor: float,
    cfg: AppConfig,
) -> np.ndarray:
    if not np.any(allowed_mask):
        return np.empty((0, 2), dtype=int)
    distance = ndi.distance_transform_edt(allowed_mask)
    if float(distance.max()) <= 1.0:
        return np.empty((0, 2), dtype=int)
    min_distance = max(2, int(round(single_minor * cfg.detector.cv_peak_min_distance_minor_scale)))
    threshold_abs = max(1.0, float(np.percentile(distance[allowed_mask], 70)))
    peaks = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        labels=allowed_mask.astype(np.uint8),
    )
    if peaks.size == 0:
        return peaks
    scores = distance[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(scores)[::-1]
    return peaks[order]


def _merge_peak_sets(
    peak_sets: Sequence[np.ndarray],
    *,
    score_image: np.ndarray,
    min_distance: int,
) -> np.ndarray:
    candidates: list[tuple[float, int, int]] = []
    for peaks in peak_sets:
        if peaks is None or len(peaks) == 0:
            continue
        for peak_row, peak_col in np.asarray(peaks, dtype=int):
            candidates.append((float(score_image[int(peak_row), int(peak_col)]), int(peak_row), int(peak_col)))
    if not candidates:
        return np.empty((0, 2), dtype=int)
    candidates.sort(key=lambda item: item[0], reverse=True)
    accepted: list[tuple[int, int]] = []
    for _score, row, col in candidates:
        if any((row - prev_row) ** 2 + (col - prev_col) ** 2 < min_distance**2 for prev_row, prev_col in accepted):
            continue
        accepted.append((row, col))
    if not accepted:
        return np.empty((0, 2), dtype=int)
    return np.asarray(accepted, dtype=int)


def _ellipse_mask(
    shape: tuple[int, int],
    center_row: float,
    center_col: float,
    major_axis_px: float,
    minor_axis_px: float,
    angle_deg: float,
) -> np.ndarray:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    rows = rows - float(center_row)
    cols = cols - float(center_col)
    theta = np.deg2rad(float(angle_deg))
    xr = cols * np.cos(theta) + rows * np.sin(theta)
    yr = -cols * np.sin(theta) + rows * np.cos(theta)
    semi_major = max(2.0, major_axis_px / 2.0)
    semi_minor = max(1.5, minor_axis_px / 2.0)
    return ((xr / semi_major) ** 2 + (yr / semi_minor) ** 2) <= 1.0


def _is_suspicious(row: pd.Series, single_area: float, single_major: float, cfg: AppConfig) -> bool:
    return (
        float(row["area_px"]) > single_area * cfg.detector.cv_suspicious_area_ratio
        or float(row["major_axis_px"]) > single_major * cfg.detector.cv_suspicious_major_ratio
        or float(row["solidity"]) < cfg.detector.cv_suspicious_solidity_max
    )


def _rescale_component_row(
    row_dict: dict[str, object],
    *,
    scale: float,
    original_shape: tuple[int, int],
) -> dict[str, object]:
    if scale == 1.0:
        row_dict["image_height"] = int(original_shape[0])
        row_dict["image_width"] = int(original_shape[1])
        return row_dict

    y0 = int(round(float(row_dict["bbox_y0"]) / scale))
    x0 = int(round(float(row_dict["bbox_x0"]) / scale))
    y1 = int(round(float(row_dict["bbox_y1"]) / scale))
    x1 = int(round(float(row_dict["bbox_x1"]) / scale))
    y1 = max(y0 + 1, min(original_shape[0], y1))
    x1 = max(x0 + 1, min(original_shape[1], x1))
    resized_mask = cv2.resize(
        row_dict["mask"].astype(np.uint8),
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    rebuilt = build_component_row(
        resized_mask,
        y0,
        x0,
        original_shape,
        str(row_dict["component_id"]),
        parent_component_id=row_dict.get("parent_component_id"),
        split_from_cluster=bool(row_dict.get("split_from_cluster", False)),
    )
    for key, value in row_dict.items():
        if key in rebuilt or key == "mask":
            continue
        rebuilt[key] = value
    rebuilt["image_height"] = int(original_shape[0])
    rebuilt["image_width"] = int(original_shape[1])
    return rebuilt


def _build_child_row(
    local_mask: np.ndarray,
    *,
    patch_y0: int,
    patch_x0: int,
    image_shape: tuple[int, int],
    component_id: str,
    parent_component_id: str,
    seed_count: int,
    dense_deblend: bool,
    center_only: bool,
) -> dict[str, object]:
    row = build_component_row(
        local_mask,
        patch_y0,
        patch_x0,
        image_shape,
        component_id,
        parent_component_id=parent_component_id,
        split_from_cluster=False,
    )
    row["cv_seed_count"] = int(seed_count)
    row["cv_dense_deblend"] = bool(dense_deblend)
    row["cv_center_only"] = bool(center_only)
    row["detector_backend"] = "cv_peak_deblend"
    row["detector_source"] = "cv_peak_deblend"
    return row


def _component_patch_bounds(row: pd.Series, image_shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    y0 = max(0, int(math.floor(float(row["bbox_y0"]) - padding)))
    x0 = max(0, int(math.floor(float(row["bbox_x0"]) - padding)))
    y1 = min(image_shape[0], int(math.ceil(float(row["bbox_y1"]) + padding)))
    x1 = min(image_shape[1], int(math.ceil(float(row["bbox_x1"]) + padding)))
    return y0, x0, y1, x1


def _deblend_component(
    row: pd.Series,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    single_area: float,
    single_major: float,
    single_minor: float,
    cfg: AppConfig,
    *,
    dense_deblend: bool = False,
    loose_mode: bool = False,
) -> list[dict[str, object]] | None:
    padding = max(4, int(round(single_minor * (1.25 if loose_mode else 1.0))))
    patch_y0, patch_x0, patch_y1, patch_x1 = _component_patch_bounds(row, score_image.shape[:2], padding)
    patch_score = score_image[patch_y0:patch_y1, patch_x0:patch_x1]
    patch_fg = foreground_mask[patch_y0:patch_y1, patch_x0:patch_x1]
    local_mask = np.zeros_like(patch_fg, dtype=bool)
    y0 = int(row["bbox_y0"]) - patch_y0
    y1 = int(row["bbox_y1"]) - patch_y0
    x0 = int(row["bbox_x0"]) - patch_x0
    x1 = int(row["bbox_x1"]) - patch_x0
    local_mask[y0:y1, x0:x1] = row["mask"].astype(bool)
    dilation_kernel = np.ones((5, 5), dtype=np.uint8) if loose_mode else np.ones((3, 3), dtype=np.uint8)
    dilation_iters = 2 if loose_mode else 1
    allowed_mask = cv2.dilate(local_mask.astype(np.uint8), dilation_kernel, iterations=dilation_iters) > 0
    if loose_mode:
        if np.any(allowed_mask):
            score_floor = float(np.percentile(patch_score[allowed_mask], 35))
            allowed_mask &= patch_score >= max(score_floor, float(cfg.detector.cv_fg_brown_thresh) * 0.75)
        if int(np.sum(allowed_mask)) < cfg.detector.cv_min_area_px:
            allowed_mask = cv2.dilate(local_mask.astype(np.uint8), dilation_kernel, iterations=dilation_iters) > 0
    else:
        allowed_mask &= patch_fg
    if int(np.sum(allowed_mask)) < cfg.detector.cv_min_area_px:
        return None

    response, angle_indexes = _matched_response(
        patch_score,
        single_major * cfg.detector.cv_filter_major_px_scale,
        single_minor * cfg.detector.cv_filter_minor_px_scale,
        cfg.detector.cv_filter_angles_deg,
    )
    watershed_score = response
    peaks = _ranked_peaks(response, allowed_mask, single_minor, cfg)
    if len(peaks) <= 1:
        peaks = _multithresh_peaks(response, allowed_mask, single_minor, cfg)
    if len(peaks) <= 1:
        peaks = _distance_peaks(allowed_mask, single_minor, cfg)
        if len(peaks) <= 1:
            return None

    max_seeds = max(2, int(round(float(row["area_px"]) / max(single_area, 1.0) * cfg.detector.cv_max_seed_area_ratio)))
    if len(peaks) > max_seeds:
        peaks = peaks[:max_seeds]
    if len(peaks) <= 1:
        return None

    markers = np.zeros_like(allowed_mask, dtype=np.int32)
    for marker_index, (peak_row, peak_col) in enumerate(peaks, start=1):
        markers[int(peak_row), int(peak_col)] = marker_index

    labels = segmentation.watershed(-watershed_score, markers, mask=allowed_mask)
    min_child_area = max(float(cfg.detector.cv_min_area_px) * 0.50, single_area * 0.30)
    max_child_area = max(single_area * 2.3, min_child_area + 1.0)

    child_rows: list[dict[str, object]] = []
    valid_regions = [
        region
        for region in measure.regionprops(labels)
        if region.area >= min_child_area
    ]
    if 2 <= len(valid_regions) <= len(peaks):
        kept_regions = [region for region in valid_regions if float(region.area) <= max_child_area]
        if 2 <= len(kept_regions) <= len(peaks):
            for child_index, region in enumerate(kept_regions, start=1):
                child_mask = labels == region.label
                child_rows.append(
                    _build_child_row(
                        child_mask,
                        patch_y0=patch_y0,
                        patch_x0=patch_x0,
                        image_shape=score_image.shape[:2],
                        component_id=f"{row['component_id']}_cv_{child_index:02d}",
                        parent_component_id=str(row["component_id"]),
                        seed_count=len(peaks),
                        dense_deblend=dense_deblend,
                        center_only=False,
                    )
                )
    if len(child_rows) >= 2:
        return child_rows

    if not cfg.detector.cv_enable_center_only_multi_emit:
        return None

    ellipse_rows: list[dict[str, object]] = []
    for child_index, (peak_row, peak_col) in enumerate(peaks, start=1):
        angle_deg = cfg.detector.cv_filter_angles_deg[int(angle_indexes[int(peak_row), int(peak_col)])]
        child_mask = _ellipse_mask(
            patch_score.shape,
            float(peak_row),
            float(peak_col),
            single_major * cfg.detector.cv_filter_major_px_scale,
            single_minor * cfg.detector.cv_filter_minor_px_scale,
            angle_deg,
        )
        child_mask &= allowed_mask
        if float(np.sum(child_mask)) < min_child_area:
            continue
        ellipse_rows.append(
            _build_child_row(
                child_mask,
                patch_y0=patch_y0,
                patch_x0=patch_x0,
                image_shape=score_image.shape[:2],
                component_id=f"{row['component_id']}_cv_{child_index:02d}",
                parent_component_id=str(row["component_id"]),
                seed_count=len(peaks),
                dense_deblend=dense_deblend,
                center_only=True,
            )
        )
    return ellipse_rows if len(ellipse_rows) >= 2 else None


def _dense_patch_supplement(
    rows: list[dict[str, object]],
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    blue_mask: np.ndarray | None,
    paper_bounds: PaperBounds | None,
    single_area: float,
    single_major: float,
    single_minor: float,
    cfg: AppConfig,
) -> list[dict[str, object]]:
    if len(rows) < 6:
        return []
    frame = pd.DataFrame(rows)
    points = frame[["centroid_y", "centroid_x"]].astype(float).to_numpy()
    if len(points) < 6:
        return []
    eps = max(8.0, single_major * 1.8)
    labels = DBSCAN(eps=eps, min_samples=4).fit_predict(points)
    extras: list[dict[str, object]] = []
    for cluster_id in sorted(set(int(label) for label in labels if int(label) >= 0)):
        cluster_df = frame.loc[labels == cluster_id].copy()
        patch_y0, patch_x0, patch_y1, patch_x1 = _component_patch_bounds(
            pd.Series(
                {
                    "bbox_y0": cluster_df["bbox_y0"].astype(float).min(),
                    "bbox_x0": cluster_df["bbox_x0"].astype(float).min(),
                    "bbox_y1": cluster_df["bbox_y1"].astype(float).max(),
                    "bbox_x1": cluster_df["bbox_x1"].astype(float).max(),
                }
            ),
            score_image.shape[:2],
            max(6, int(round(single_major * 0.8))),
        )
        patch_score = score_image[patch_y0:patch_y1, patch_x0:patch_x1]
        if patch_score.size == 0:
            continue
        patch_fg = foreground_mask[patch_y0:patch_y1, patch_x0:patch_x1].copy()
        if blue_mask is not None and blue_mask.size:
            patch_fg &= blue_mask[patch_y0:patch_y1, patch_x0:patch_x1] == 0
        if paper_bounds is not None:
            local_bounds = _localize_paper_bounds(
                paper_bounds,
                patch_y0=patch_y0,
                patch_x0=patch_x0,
                patch_shape=patch_score.shape,
            )
            if local_bounds is not None:
                patch_fg &= _paper_mask(patch_score.shape, local_bounds)
        response, angle_indexes = _matched_response(
            patch_score,
            single_major * cfg.detector.cv_filter_major_px_scale,
            single_minor * cfg.detector.cv_filter_minor_px_scale,
            cfg.detector.cv_filter_angles_deg,
        )
        peaks = _ranked_peaks(response, patch_fg, single_minor, cfg)
        if len(peaks) <= len(cluster_df):
            continue
        existing = cluster_df[["centroid_y", "centroid_x"]].astype(float).to_numpy()
        max_extra = max(1, int(math.ceil(len(cluster_df) * 0.30)))
        added = 0
        for peak_row, peak_col in peaks:
            global_row = float(patch_y0 + peak_row)
            global_col = float(patch_x0 + peak_col)
            if existing.size and np.min(np.sum((existing - np.array([global_row, global_col])) ** 2, axis=1)) < (single_minor * 0.75) ** 2:
                continue
            angle_deg = cfg.detector.cv_filter_angles_deg[int(angle_indexes[int(peak_row), int(peak_col)])]
            child_mask = _ellipse_mask(
                patch_score.shape,
                float(peak_row),
                float(peak_col),
                single_major * cfg.detector.cv_filter_major_px_scale,
                single_minor * cfg.detector.cv_filter_minor_px_scale,
                angle_deg,
            )
            child_mask &= patch_fg
            if float(np.sum(child_mask)) < max(float(cfg.detector.cv_min_area_px) * 0.4, single_area * 0.18):
                continue
            row = _build_child_row(
                child_mask,
                patch_y0=patch_y0,
                patch_x0=patch_x0,
                image_shape=score_image.shape[:2],
                component_id=f"cv_dense_{cluster_id:02d}_{added + 1:02d}",
                parent_component_id=f"cv_dense_{cluster_id:02d}",
                seed_count=len(peaks),
                dense_deblend=True,
                center_only=True,
            )
            inside_fraction = bbox_fraction_inside_paper_bounds(
                row["bbox_x0"],
                row["bbox_y0"],
                row["bbox_x1"],
                row["bbox_y1"],
                paper_bounds,
            )
            if inside_fraction < 0.75:
                continue
            extras.append(row)
            existing = np.vstack([existing, np.array([global_row, global_col])]) if existing.size else np.asarray([[global_row, global_col]])
            added += 1
            if added >= max_extra:
                break
    return extras


def _finalize_component_rows(
    rows: list[dict[str, object]],
    *,
    scale: float,
    original_shape: tuple[int, int],
    component_prefix: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    scaled_rows = [
        _rescale_component_row(row_dict, scale=scale, original_shape=original_shape)
        for row_dict in rows
    ]
    frame = pd.DataFrame(scaled_rows)
    if frame.empty:
        return frame
    frame["component_id"] = frame["component_id"].astype(str).map(lambda value: f"{component_prefix}_{value}")
    if "parent_component_id" in frame.columns:
        frame["parent_component_id"] = frame["parent_component_id"].apply(
            lambda value: None if value is None or (isinstance(value, float) and np.isnan(value)) else f"{component_prefix}_{value}"
        )
    frame["image_height"] = int(original_shape[0])
    frame["image_width"] = int(original_shape[1])
    return frame.sort_values("component_id").reset_index(drop=True)


def _refine_component_dataframe(
    component_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    blue_mask: np.ndarray | None,
    paper_bounds: PaperBounds | None,
    cfg: AppConfig,
    component_prefix: str,
    scale: float = 1.0,
    original_shape: tuple[int, int] | None = None,
) -> pd.DataFrame:
    if component_df.empty:
        return component_df.copy()
    original_shape = original_shape or score_image.shape[:2]
    component_df = component_df.loc[
        component_df["area_px"].astype(float) >= cfg.detector.cv_min_area_px
    ].reset_index(drop=True)
    if component_df.empty:
        return component_df.copy()

    single_area, single_major, single_minor = _estimate_single_scale(component_df, cfg)

    rows: list[dict[str, object]] = []
    for _, row in component_df.iterrows():
        row_dict = row.to_dict()
        row_dict["cv_seed_count"] = 1
        row_dict["cv_dense_deblend"] = False
        row_dict["cv_center_only"] = False
        row_dict["detector_backend"] = "cv_peak_deblend"
        row_dict["detector_source"] = "cv_peak_deblend"
        if _is_suspicious(row, single_area, single_major, cfg):
            child_rows = _deblend_component(
                row,
                score_image,
                foreground_mask,
                single_area,
                single_major,
                single_minor,
                cfg,
                dense_deblend=False,
            )
            if child_rows:
                rows.extend(child_rows)
                continue
        rows.append(row_dict)

    rows.extend(
        _dense_patch_supplement(
            rows,
            score_image,
            foreground_mask,
            blue_mask,
            paper_bounds,
            single_area,
            single_major,
            single_minor,
            cfg,
        )
    )

    return _finalize_component_rows(
        rows,
        scale=scale,
        original_shape=original_shape,
        component_prefix=component_prefix,
    )


def refine_component_candidates(
    component_df: pd.DataFrame,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    cfg: AppConfig,
    *,
    blue_mask: np.ndarray | None = None,
    paper_bounds: PaperBounds | None = None,
    component_prefix: str = "cv",
) -> pd.DataFrame:
    """Refine a fast CV backbone with local deblend on suspicious components."""
    return _refine_component_dataframe(
        component_df,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        blue_mask=blue_mask,
        paper_bounds=paper_bounds,
        cfg=cfg,
        component_prefix=component_prefix,
        scale=1.0,
        original_shape=score_image.shape[:2],
    )


def _reclassify_component_frame(
    component_df: pd.DataFrame,
    *,
    feature_image: np.ndarray,
    blue_mask: np.ndarray | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    if component_df.empty:
        return component_df.copy()
    features_df = featurize_components(
        feature_image,
        blue_mask if blue_mask is not None else np.zeros(feature_image.shape[:2], dtype=np.uint8),
        component_df,
    )
    return rule_classify_components(features_df, cfg)


def _ensure_cv_debug_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    updated = frame.copy()
    defaults = {
        "cv_seed_count": 1,
        "cv_dense_deblend": False,
        "cv_center_only": False,
        "cv_patch_supplement": False,
        "detector_backend": "cv_peak_deblend",
        "detector_source": "cv_peak_deblend",
    }
    for column, default in defaults.items():
        if column not in updated.columns:
            updated[column] = default
        else:
            series = updated[column]
            updated[column] = series.mask(series.isna(), default)
    return updated


def _estimate_labeled_scale(labeled_df: pd.DataFrame, cfg: AppConfig) -> tuple[float, float, float]:
    if labeled_df.empty:
        return 120.0, 18.0, 8.0
    preferred = labeled_df.loc[labeled_df["label"].astype(str) == "pupa"].copy()
    if len(preferred) < 3:
        preferred = labeled_df.loc[labeled_df["label"].astype(str) != "artifact"].copy()
    if preferred.empty:
        preferred = labeled_df.copy()
    return _estimate_single_scale(preferred, cfg)


def _promote_uncertain_rows(
    labeled_df: pd.DataFrame,
    *,
    confidence_min: float,
    color_score_min: float,
    max_mean_v: float,
    max_mean_lab_b: float,
    min_seed_count: int | None = None,
    restrict_to_patch_supplement: bool = False,
) -> pd.DataFrame:
    if labeled_df.empty:
        return labeled_df.copy()
    frame = labeled_df.copy()
    mask = frame["label"].astype(str) == "uncertain"
    mask &= frame.get("is_active", True).astype(bool)
    mask &= frame["confidence"].astype(float) >= confidence_min
    mask &= frame["color_score"].astype(float) >= color_score_min
    mask &= frame["blue_overlap_ratio"].astype(float) <= 0.06
    mask &= frame["border_touch_ratio"].astype(float) <= 0.08
    mask &= frame["mean_v"].astype(float) <= max_mean_v
    mask &= frame["mean_lab_b"].astype(float) <= max_mean_lab_b
    if min_seed_count is not None and "cv_seed_count" in frame.columns:
        mask &= frame["cv_seed_count"].astype(float) >= float(min_seed_count)
    if restrict_to_patch_supplement:
        mask &= frame.get("cv_patch_supplement", False).astype(bool)
    if mask.any():
        frame.loc[mask, "label"] = "pupa"
        frame.loc[mask, "confidence"] = np.maximum(
            frame.loc[mask, "confidence"].astype(float).to_numpy(),
            float(confidence_min),
        )
    return frame


def _promote_pairlike_artifact_rows(
    labeled_df: pd.DataFrame,
    *,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_promote_pairlike_artifacts_enabled:
        return labeled_df.copy()
    frame = labeled_df.copy()
    mask = frame["label"].astype(str) == "artifact"
    mask &= frame.get("is_active", True).astype(bool)
    mask &= frame["cv_seed_count"].astype(float) >= 2.0
    mask &= frame["color_score"].astype(float) >= cfg.detector.cv_promote_pairlike_artifacts_color_score_min
    mask &= frame["blue_overlap_ratio"].astype(float) <= 0.03
    mask &= frame["border_touch_ratio"].astype(float) <= 0.05
    mask &= frame["mean_v"].astype(float) <= cfg.detector.cv_promote_pairlike_artifacts_max_mean_v
    mask &= frame["mean_lab_b"].astype(float) <= cfg.detector.cv_promote_pairlike_artifacts_max_mean_lab_b
    mask &= frame["area_px"].astype(float) >= cfg.detector.cv_promote_pairlike_artifacts_min_area_px
    if mask.any():
        frame.loc[mask, "label"] = "pupa"
        frame.loc[mask, "confidence"] = np.maximum(
            frame.loc[mask, "confidence"].astype(float).to_numpy(),
            float(cfg.detector.cv_promote_pairlike_artifacts_color_score_min),
        )
    return frame


def _erosion_split_component(
    row: pd.Series,
    foreground_mask: np.ndarray,
    single_area: float,
    single_minor: float,
    image_shape: tuple[int, int],
) -> list[dict[str, object]] | None:
    """Split a large component using distance-transform peaks + watershed."""
    padding = max(4, int(round(single_minor)))
    y0 = max(0, int(math.floor(float(row["bbox_y0"]) - padding)))
    x0 = max(0, int(math.floor(float(row["bbox_x0"]) - padding)))
    y1 = min(image_shape[0], int(math.ceil(float(row["bbox_y1"]) + padding)))
    x1 = min(image_shape[1], int(math.ceil(float(row["bbox_x1"]) + padding)))

    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    my0 = int(row["bbox_y0"]) - y0
    my1 = int(row["bbox_y1"]) - y0
    mx0 = int(row["bbox_x0"]) - x0
    mx1 = int(row["bbox_x1"]) - x0
    local_mask[my0:my1, mx0:mx1] = row["mask"].astype(np.uint8)
    local_mask &= foreground_mask[y0:y1, x0:x1].astype(np.uint8)

    parent_area = float(np.sum(local_mask))
    if parent_area < single_area * 1.2:
        return None
    expected_n = max(2, min(6, int(round(parent_area / max(single_area, 1.0)))))
    min_child_area = max(20.0, single_area * 0.18)

    # Strategy 1: Distance transform peaks
    distance = ndi.distance_transform_edt(local_mask > 0)
    if float(distance.max()) <= 1.0:
        return None

    # Use a low min_distance to find close peaks
    min_dist = max(2, int(round(single_minor * 0.40)))
    thresh = max(1.0, float(np.percentile(distance[local_mask > 0], 50)))
    peaks = feature.peak_local_max(
        distance,
        min_distance=min_dist,
        threshold_abs=thresh,
        labels=local_mask,
    )

    # Strategy 2: If distance peaks fail, try progressive erosion
    if len(peaks) <= 1:
        for erosion_iter in range(1, 10):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(local_mask, kernel, iterations=erosion_iter)
            if eroded.sum() == 0:
                break
            n_labels, labels = cv2.connectedComponents(eroded, connectivity=8)
            if n_labels - 1 >= 2:
                # Use eroded component centroids as peaks
                peaks_list = []
                for lbl in range(1, n_labels):
                    ys, xs = np.where(labels == lbl)
                    if len(ys) >= min_child_area * 0.3:
                        # Use distance maximum within this component
                        comp_mask = labels == lbl
                        comp_dist = distance * comp_mask
                        max_pos = np.unravel_index(np.argmax(comp_dist), comp_dist.shape)
                        peaks_list.append([max_pos[0], max_pos[1]])
                if len(peaks_list) >= 2:
                    peaks = np.array(peaks_list, dtype=int)
                    break

    if len(peaks) <= 1:
        return None
    if len(peaks) > expected_n:
        # Keep only the strongest peaks
        scores = distance[peaks[:, 0], peaks[:, 1]]
        order = np.argsort(scores)[::-1][:expected_n]
        peaks = peaks[order]

    # Watershed with peaks as markers
    markers = np.zeros_like(local_mask, dtype=np.int32)
    for i, (pr, pc) in enumerate(peaks, start=1):
        markers[int(pr), int(pc)] = i

    ws_labels = segmentation.watershed(-distance, markers, mask=local_mask > 0)

    child_rows: list[dict[str, object]] = []
    for i in range(1, len(peaks) + 1):
        child_mask = (ws_labels == i).astype(np.uint8)
        child_area = float(np.sum(child_mask))
        if child_area < min_child_area:
            continue
        child_rows.append(
            _build_child_row(
                child_mask.astype(bool),
                patch_y0=y0,
                patch_x0=x0,
                image_shape=image_shape,
                component_id="{}_esplit_{:02d}".format(row["component_id"], i),
                parent_component_id=str(row["component_id"]),
                seed_count=len(peaks),
                dense_deblend=True,
                center_only=False,
            )
        )

    return child_rows if len(child_rows) >= 2 else None


def _pairlike_split_component(
    row: pd.Series,
    foreground_mask: np.ndarray,
    single_area: float,
    single_minor: float,
    image_shape: tuple[int, int],
) -> list[dict[str, object]] | None:
    """Split a pupa-like component into exactly 2 children using concavity/erosion cues."""
    padding = max(4, int(round(single_minor * 0.8)))
    y0 = max(0, int(math.floor(float(row["bbox_y0"]) - padding)))
    x0 = max(0, int(math.floor(float(row["bbox_x0"]) - padding)))
    y1 = min(image_shape[0], int(math.ceil(float(row["bbox_y1"]) + padding)))
    x1 = min(image_shape[1], int(math.ceil(float(row["bbox_x1"]) + padding)))

    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    my0 = int(row["bbox_y0"]) - y0
    my1 = int(row["bbox_y1"]) - y0
    mx0 = int(row["bbox_x0"]) - x0
    mx1 = int(row["bbox_x1"]) - x0
    local_mask[my0:my1, mx0:mx1] = row["mask"].astype(np.uint8)
    local_mask &= foreground_mask[y0:y1, x0:x1].astype(np.uint8)

    parent_area = float(np.sum(local_mask))
    if parent_area < single_area * 1.2:
        return None

    min_child_area = max(30.0, single_area * 0.30)
    max_child_balance = 2.5  # max ratio of larger/smaller child

    # Strategy 1: progressive erosion to find exactly 2 lobes
    best_split = None
    for erosion_iter in range(1, 8):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(local_mask, kernel, iterations=erosion_iter)
        if eroded.sum() == 0:
            break
        n_labels, labels = cv2.connectedComponents(eroded, connectivity=8)
        if n_labels - 1 < 2:
            continue
        # Only take the 2 largest lobes
        areas = [(np.sum(labels == lbl), lbl) for lbl in range(1, n_labels)]
        areas.sort(reverse=True)
        if len(areas) < 2:
            continue
        top2 = areas[:2]
        if top2[0][0] < min_child_area or top2[1][0] < min_child_area:
            continue
        balance = float(top2[0][0]) / max(float(top2[1][0]), 1.0)
        if balance > max_child_balance:
            continue
        best_split = (erosion_iter, top2)
        break  # Take first valid split (minimal erosion)

    if best_split is None:
        # Strategy 2: distance transform with aggressive peak finding
        distance = ndi.distance_transform_edt(local_mask > 0)
        if float(distance.max()) <= 1.5:
            return None
        min_dist = max(2, int(round(single_minor * 0.30)))
        thresh = max(1.0, float(np.percentile(distance[local_mask > 0], 40)))
        peaks = feature.peak_local_max(
            distance, min_distance=min_dist, threshold_abs=thresh, labels=local_mask,
        )
        if len(peaks) < 2:
            return None
        # Take top 2 peaks
        peak_scores = distance[peaks[:, 0], peaks[:, 1]]
        top2_idx = np.argsort(peak_scores)[::-1][:2]
        peaks = peaks[top2_idx]
        markers = np.zeros_like(local_mask, dtype=np.int32)
        markers[peaks[0][0], peaks[0][1]] = 1
        markers[peaks[1][0], peaks[1][1]] = 2
        ws_labels = segmentation.watershed(-distance, markers, mask=local_mask > 0)
        c1_area = float(np.sum(ws_labels == 1))
        c2_area = float(np.sum(ws_labels == 2))
        if c1_area < min_child_area or c2_area < min_child_area:
            return None
        balance = max(c1_area, c2_area) / max(min(c1_area, c2_area), 1.0)
        if balance > max_child_balance:
            return None
        # Build children directly from watershed
        child_rows = []
        for ci in [1, 2]:
            child_mask = (ws_labels == ci).astype(bool)
            child_rows.append(_build_child_row(
                child_mask, patch_y0=y0, patch_x0=x0, image_shape=image_shape,
                component_id="{}_pair_{:02d}".format(row["component_id"], ci),
                parent_component_id=str(row["component_id"]),
                seed_count=2, dense_deblend=False, center_only=False,
            ))
        return child_rows if len(child_rows) == 2 else None

    # Use erosion lobes as seeds for watershed
    erosion_iter, top2 = best_split
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(local_mask, kernel, iterations=erosion_iter)
    _, labels = cv2.connectedComponents(eroded, connectivity=8)

    distance = ndi.distance_transform_edt(local_mask > 0)
    markers = np.zeros_like(local_mask, dtype=np.int32)
    for i, (_, lbl) in enumerate(top2, start=1):
        markers[labels == lbl] = i

    ws_labels = segmentation.watershed(-distance, markers, mask=local_mask > 0)
    child_rows = []
    for ci in [1, 2]:
        child_mask = (ws_labels == ci).astype(bool)
        child_area = float(np.sum(child_mask))
        if child_area < min_child_area:
            return None
        child_rows.append(_build_child_row(
            child_mask, patch_y0=y0, patch_x0=x0, image_shape=image_shape,
            component_id="{}_pair_{:02d}".format(row["component_id"], ci),
            parent_component_id=str(row["component_id"]),
            seed_count=2, dense_deblend=False, center_only=False,
        ))
    # Verify area conservation
    child_total = sum(float(np.sum(ws_labels == ci)) for ci in [1, 2])
    if child_total < parent_area * 0.5 or child_total > parent_area * 1.2:
        return None
    return child_rows if len(child_rows) == 2 else None


def _pairlike_resplit_pupa_rows(
    labeled_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    feature_image: np.ndarray,
    blue_mask: np.ndarray | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Split accepted pupa rows that look like two touching pupae."""
    if labeled_df.empty or not cfg.detector.cv_pairlike_resplit_enabled:
        return labeled_df.copy()

    single_area, single_major, single_minor = _estimate_labeled_scale(labeled_df, cfg)
    kept_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    split_parent_rows: dict[str, dict[str, object]] = {}
    if (
        "image_height" in labeled_df.columns
        and "image_width" in labeled_df.columns
        and pd.notna(labeled_df["image_height"].iloc[0])
        and pd.notna(labeled_df["image_width"].iloc[0])
    ):
        image_shape = (
            int(labeled_df["image_height"].iloc[0]),
            int(labeled_df["image_width"].iloc[0]),
        )
    else:
        image_shape = foreground_mask.shape[:2]
    pupa_nbr_60 = (
        labeled_df["pupa_nbr_60"].astype(float)
        if "pupa_nbr_60" in labeled_df.columns
        else pd.Series(0.0, index=labeled_df.index, dtype=float)
    )

    for df_idx, row in labeled_df.iterrows():
        row_dict = row.to_dict()
        label = str(row_dict.get("label", ""))
        if not row_dict.get("is_active", True) or label != "pupa":
            kept_rows.append(row_dict)
            continue

        area_px = float(row_dict.get("area_px", 0.0))
        if bool(row_dict.get("parent_component_id")):
            kept_rows.append(row_dict)
            continue
        if bool(row_dict.get("cluster_unresolved", False)):
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("blue_overlap_ratio", 1.0)) > cfg.detector.cv_pairlike_resplit_max_blue_overlap_ratio:
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("border_touch_ratio", 1.0)) > cfg.detector.cv_pairlike_resplit_max_border_touch_ratio:
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("minor_axis_px", 0.0)) < cfg.detector.cv_pairlike_resplit_min_minor_px:
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("eccentricity", 1.0)) > cfg.detector.cv_pairlike_resplit_max_eccentricity:
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("color_score", 0.0)) < cfg.detector.cv_pairlike_resplit_min_color_score:
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("local_contrast", 0.0)) < cfg.detector.cv_pairlike_resplit_min_local_contrast:
            kept_rows.append(row_dict)
            continue
        if pupa_nbr_60.loc[df_idx] > float(cfg.detector.cv_pairlike_resplit_max_neighbors_60):
            kept_rows.append(row_dict)
            continue
        if area_px < single_area * cfg.detector.cv_pairlike_resplit_min_area_ratio:
            kept_rows.append(row_dict)
            continue
        if area_px > single_area * cfg.detector.cv_pairlike_resplit_max_area_ratio:
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("solidity", 1.0)) > 0.92:
            kept_rows.append(row_dict)
            continue
        if bool(row_dict.get("cv_center_only", False)):
            kept_rows.append(row_dict)
            continue

        children = _pairlike_split_component(
            row, foreground_mask, single_area, single_minor, image_shape,
        )
        if children and len(children) == 2:
            split_parent_rows[str(row_dict.get("component_id", ""))] = row_dict
            child_rows.extend(children)
            continue
        kept_rows.append(row_dict)

    if not child_rows:
        return labeled_df.copy()

    kept_df = pd.DataFrame(kept_rows)
    child_df = pd.DataFrame(child_rows)
    child_labeled = _reclassify_component_frame(
        child_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg,
    )
    # Only accept the split if both children are non-artifact
    final_kept = kept_df.copy()
    final_children = []
    pairs = {}
    for _, cr in child_labeled.iterrows():
        parent = str(cr.get("parent_component_id", ""))
        pairs.setdefault(parent, []).append(cr.to_dict())
    for parent_id, pair in pairs.items():
        if len(pair) == 2:
            non_art = sum(1 for c in pair if str(c.get("label", "")) in {"pupa", "uncertain"})
            child_areas = np.array([float(c.get("area_px", 0.0)) for c in pair], dtype=float)
            child_area_total = float(child_areas.sum())
            parent_row = split_parent_rows.get(parent_id)
            parent_area = float(parent_row.get("area_px", 0.0)) if parent_row else 0.0
            child_balance = float(child_areas.max() / max(child_areas.min(), 1.0)) if len(child_areas) == 2 else 999.0
            if (
                non_art >= cfg.detector.cv_pairlike_resplit_min_non_artifact_children
                and child_area_total >= parent_area * cfg.detector.cv_pairlike_resplit_min_child_area_fraction
                and child_area_total <= parent_area * cfg.detector.cv_pairlike_resplit_max_child_area_fraction
                and child_balance <= cfg.detector.cv_pairlike_resplit_max_child_balance_ratio
            ):
                for child in pair:
                    child["cv_pairlike_resplit"] = True
                final_children.extend(pair)
                continue
        parent_row = split_parent_rows.get(parent_id)
        if parent_row is not None:
            final_kept = pd.concat([final_kept, pd.DataFrame([parent_row])], ignore_index=True)

    if not final_children:
        return _ensure_cv_debug_columns(final_kept).sort_values("component_id").reset_index(drop=True)

    child_final = pd.DataFrame(final_children)
    combined = pd.concat([final_kept, child_final], ignore_index=True)
    return _ensure_cv_debug_columns(combined).sort_values("component_id").reset_index(drop=True)


def _erosion_resplit_large_rows(
    labeled_df: pd.DataFrame,
    *,
    foreground_mask: np.ndarray,
    feature_image: np.ndarray,
    blue_mask: np.ndarray | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Split large cluster/uncertain/artifact components using erosion."""
    if labeled_df.empty:
        return labeled_df.copy()

    single_area, single_major, single_minor = _estimate_labeled_scale(labeled_df, cfg)
    kept_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    image_shape = (
        int(labeled_df["image_height"].iloc[0]),
        int(labeled_df["image_width"].iloc[0]),
    ) if "image_height" in labeled_df.columns else foreground_mask.shape[:2]

    for _, row in labeled_df.iterrows():
        row_dict = row.to_dict()
        label = str(row_dict.get("label", ""))
        if not row_dict.get("is_active", True):
            kept_rows.append(row_dict)
            continue

        area_px = float(row_dict.get("area_px", 0.0))
        # Target large pupa (likely multi-pupa merge) AND non-pupa labels
        area_threshold = single_area * 2.5 if label == "pupa" else single_area * 2.0
        if area_px < area_threshold:
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("blue_overlap_ratio", 1.0)) > 0.10:
            kept_rows.append(row_dict)
            continue
        if float(row_dict.get("border_touch_ratio", 1.0)) > 0.30:
            kept_rows.append(row_dict)
            continue

        children = _erosion_split_component(
            row, foreground_mask, single_area, single_minor, image_shape,
        )
        if children and len(children) >= 2:
            child_rows.extend(children)
            continue
        kept_rows.append(row_dict)

    if not child_rows:
        return labeled_df.copy()

    kept_df = pd.DataFrame(kept_rows)
    child_df = pd.DataFrame(child_rows)
    child_labeled = _reclassify_component_frame(
        child_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg,
    )
    combined = pd.concat([kept_df, child_labeled], ignore_index=True)
    return _ensure_cv_debug_columns(combined).sort_values("component_id").reset_index(drop=True)


def _resplit_strong_artifact_rows(
    labeled_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    feature_image: np.ndarray,
    blue_mask: np.ndarray | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_artifact_resplit_enabled:
        return labeled_df.copy()

    single_area, single_major, single_minor = _estimate_labeled_scale(labeled_df, cfg)
    kept_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []

    for _, row in labeled_df.iterrows():
        row_dict = row.to_dict()
        if not row_dict.get("is_active", True) or str(row_dict.get("label", "")) != "artifact":
            kept_rows.append(row_dict)
            continue

        seed_count = float(row_dict.get("cv_seed_count", 1.0) or 1.0)
        size_trigger = (
            float(row_dict.get("area_px", 0.0)) >= single_area * cfg.detector.cv_artifact_resplit_min_area_ratio
            or float(row_dict.get("major_axis_px", 0.0)) >= single_major * cfg.detector.cv_artifact_resplit_min_major_ratio
            or seed_count >= 2.0
        )
        strong_artifact = (
            size_trigger
            and float(row_dict.get("blue_overlap_ratio", 1.0)) <= 0.03
            and float(row_dict.get("border_touch_ratio", 1.0)) <= cfg.detector.cv_artifact_resplit_max_border_touch_ratio
            and float(row_dict.get("color_score", 0.0)) >= cfg.detector.cv_artifact_resplit_color_score_min
            and float(row_dict.get("local_contrast", 0.0)) >= cfg.detector.cv_artifact_resplit_min_local_contrast
            and float(row_dict.get("mean_v", 255.0)) <= cfg.detector.cv_artifact_resplit_max_mean_v
            and float(row_dict.get("mean_lab_b", 255.0)) <= cfg.detector.cv_artifact_resplit_max_mean_lab_b
        )
        if not strong_artifact:
            kept_rows.append(row_dict)
            continue

        child_components = _deblend_component(
            row,
            score_image,
            foreground_mask,
            single_area,
            single_major,
            single_minor,
            cfg,
            dense_deblend=True,
        )
        if not child_components or len(child_components) < 2:
            child_components = _deblend_component(
                row,
                score_image,
                foreground_mask,
                single_area,
                single_major,
                single_minor,
                cfg,
                dense_deblend=True,
                loose_mode=True,
            )
        if child_components and len(child_components) >= 2:
            child_rows.extend(child_components)
            continue
        kept_rows.append(row_dict)

    if not child_rows:
        return labeled_df.copy()

    kept_df = pd.DataFrame(kept_rows)
    child_df = pd.DataFrame(child_rows)
    child_labeled = _reclassify_component_frame(child_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg)
    combined = pd.concat([kept_df, child_labeled], ignore_index=True)
    return _ensure_cv_debug_columns(combined).sort_values("component_id").reset_index(drop=True)


def _resplit_large_cluster_rows(
    labeled_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    feature_image: np.ndarray,
    blue_mask: np.ndarray | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_large_cluster_resplit_enabled:
        return labeled_df.copy()

    single_area, single_major, single_minor = _estimate_labeled_scale(labeled_df, cfg)
    kept_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []

    for _, row in labeled_df.iterrows():
        row_dict = row.to_dict()
        label = str(row_dict.get("label", ""))
        if not row_dict.get("is_active", True) or label not in {"cluster", "uncertain"}:
            kept_rows.append(row_dict)
            continue

        size_trigger = (
            float(row_dict.get("area_px", 0.0)) >= single_area * cfg.detector.cv_large_cluster_resplit_min_area_ratio
            or float(row_dict.get("major_axis_px", 0.0)) >= single_major * cfg.detector.cv_large_cluster_resplit_min_major_ratio
            or bool(row_dict.get("cluster_unresolved", False))
        )
        strong_cluster = (
            size_trigger
            and float(row_dict.get("blue_overlap_ratio", 1.0)) <= 0.03
            and float(row_dict.get("border_touch_ratio", 1.0)) <= cfg.detector.cv_large_cluster_resplit_max_border_touch_ratio
            and float(row_dict.get("color_score", 0.0)) >= cfg.detector.cv_large_cluster_resplit_color_score_min
            and float(row_dict.get("local_contrast", 0.0)) >= cfg.detector.cv_large_cluster_resplit_min_local_contrast
        )
        if not strong_cluster:
            kept_rows.append(row_dict)
            continue

        child_components = _deblend_component(
            row,
            score_image,
            foreground_mask,
            single_area,
            single_major,
            single_minor,
            cfg,
            dense_deblend=True,
        )
        if not child_components or len(child_components) < 2:
            child_components = _deblend_component(
                row,
                score_image,
                foreground_mask,
                single_area,
                single_major,
                single_minor,
                cfg,
                dense_deblend=True,
                loose_mode=True,
            )
        if child_components and len(child_components) >= 2:
            child_rows.extend(child_components)
            continue
        kept_rows.append(row_dict)

    if not child_rows:
        return labeled_df.copy()

    kept_df = pd.DataFrame(kept_rows)
    child_df = pd.DataFrame(child_rows)
    child_labeled = _reclassify_component_frame(child_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg)
    combined = pd.concat([kept_df, child_labeled], ignore_index=True)
    return _ensure_cv_debug_columns(combined).sort_values("component_id").reset_index(drop=True)


def _resplit_large_pupa_rows(
    labeled_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    feature_image: np.ndarray,
    blue_mask: np.ndarray | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_large_pupa_resplit_enabled:
        return labeled_df.copy()

    single_area, single_major, single_minor = _estimate_labeled_scale(labeled_df, cfg)
    kept_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []

    for _, row in labeled_df.iterrows():
        row_dict = row.to_dict()
        if not row_dict.get("is_active", True) or str(row_dict.get("label", "")) != "pupa":
            kept_rows.append(row_dict)
            continue

        seed_count = float(row_dict.get("cv_seed_count", 1.0) or 1.0)
        area_px = float(row_dict.get("area_px", 0.0))
        major_px = float(row_dict.get("major_axis_px", 0.0))
        size_trigger = (
            area_px >= single_area * cfg.detector.cv_large_pupa_resplit_min_area_ratio
            or major_px >= single_major * cfg.detector.cv_large_pupa_resplit_min_major_ratio
            or seed_count >= float(cfg.detector.cv_large_pupa_resplit_min_seed_count)
        )
        strong_pupa = (
            size_trigger
            and not bool(row_dict.get("cluster_unresolved", False))
            and not bool(row_dict.get("parent_component_id"))
            and float(row_dict.get("blue_overlap_ratio", 1.0)) <= 0.03
            and float(row_dict.get("border_touch_ratio", 1.0)) <= cfg.detector.cv_large_pupa_resplit_max_border_touch_ratio
            and float(row_dict.get("color_score", 0.0)) >= cfg.detector.cv_large_pupa_resplit_color_score_min
            and float(row_dict.get("local_contrast", 0.0)) >= cfg.detector.cv_large_pupa_resplit_min_local_contrast
            and area_px <= single_area * cfg.detector.cv_large_pupa_resplit_max_area_ratio
            and seed_count <= float(cfg.detector.cv_large_pupa_resplit_max_seed_count)
            and float(row_dict.get("aspect_ratio", 0.0)) >= cfg.detector.cv_large_pupa_resplit_min_aspect_ratio
            and float(row_dict.get("eccentricity", 0.0)) >= cfg.detector.cv_large_pupa_resplit_min_eccentricity
        )
        if not strong_pupa:
            kept_rows.append(row_dict)
            continue

        child_components = _deblend_component(
            row,
            score_image,
            foreground_mask,
            single_area,
            single_major,
            single_minor,
            cfg,
            dense_deblend=False,
        )
        if not child_components or len(child_components) < 2:
            child_components = _deblend_component(
                row,
                score_image,
                foreground_mask,
                single_area,
                single_major,
                single_minor,
                cfg,
                dense_deblend=False,
                loose_mode=True,
            )
        if not child_components or len(child_components) < 2:
            kept_rows.append(row_dict)
            continue

        child_df = pd.DataFrame(child_components)
        if len(child_df) != 2:
            kept_rows.append(row_dict)
            continue
        child_labeled = _reclassify_component_frame(child_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg)
        child_non_artifact = child_labeled["label"].astype(str).isin(["pupa", "uncertain"]).sum()
        child_area = float(child_labeled["area_px"].astype(float).sum())
        parent_area = float(row_dict.get("area_px", 0.0))
        child_areas = child_labeled["area_px"].astype(float).to_numpy()
        child_balance = float(child_areas.max() / max(child_areas.min(), 1.0))
        if (
            int(child_non_artifact) >= cfg.detector.cv_large_pupa_resplit_min_child_non_artifact_count
            and child_area >= parent_area * cfg.detector.cv_large_pupa_resplit_min_child_area_fraction
            and child_area <= parent_area * cfg.detector.cv_large_pupa_resplit_max_child_area_fraction
            and child_balance <= cfg.detector.cv_large_pupa_resplit_max_child_balance_ratio
        ):
            for child in child_components:
                child["cv_large_pupa_resplit"] = True
            child_rows.extend(child_components)
            continue

        kept_rows.append(row_dict)

    if not child_rows:
        return labeled_df.copy()

    kept_df = pd.DataFrame(kept_rows)
    child_df = pd.DataFrame(child_rows)
    child_labeled = _reclassify_component_frame(child_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg)
    combined = pd.concat([kept_df, child_labeled], ignore_index=True)
    return _ensure_cv_debug_columns(combined).sort_values("component_id").reset_index(drop=True)


def _promote_strong_single_artifact_rows(
    labeled_df: pd.DataFrame,
    *,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_promote_strong_single_artifacts_enabled:
        return labeled_df.copy()

    frame = labeled_df.copy()
    single_area, _, _ = _estimate_labeled_scale(frame, cfg)
    area_min = max(float(cfg.detector.cv_min_area_px), single_area * 0.28)
    area_max = max(area_min + 1.0, min(single_area * 2.8, 360.0))

    mask = frame["label"].astype(str) == "artifact"
    mask &= frame.get("is_active", True).astype(bool)
    mask &= ~frame.get("cluster_unresolved", False).astype(bool)
    mask &= frame["blue_overlap_ratio"].astype(float) <= 0.02
    mask &= frame["border_touch_ratio"].astype(float) <= cfg.detector.cv_promote_strong_single_artifacts_max_border_touch_ratio
    mask &= frame["area_px"].astype(float).between(area_min, area_max)
    mask &= frame["color_score"].astype(float) >= cfg.detector.cv_promote_strong_single_artifacts_color_score_min
    mask &= frame["local_contrast"].astype(float) >= cfg.detector.cv_promote_strong_single_artifacts_min_local_contrast
    mask &= frame["mean_v"].astype(float) <= cfg.detector.cv_promote_strong_single_artifacts_max_mean_v
    mask &= frame["mean_lab_b"].astype(float) <= cfg.detector.cv_promote_strong_single_artifacts_max_mean_lab_b
    mask &= frame["aspect_ratio"].astype(float) >= cfg.detector.cv_promote_strong_single_artifacts_min_aspect_ratio
    mask &= frame["eccentricity"].astype(float) >= cfg.detector.cv_promote_strong_single_artifacts_min_eccentricity
    if "cv_seed_count" in frame.columns:
        mask &= frame["cv_seed_count"].fillna(1.0).astype(float) <= 1.5

    if mask.any():
        frame.loc[mask, "label"] = "pupa"
        frame.loc[mask, "confidence"] = np.maximum(
            frame.loc[mask, "confidence"].astype(float).to_numpy(),
            0.60,
        )
    return frame


def _promote_large_single_artifact_rows(
    labeled_df: pd.DataFrame,
    *,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_promote_large_single_artifacts_enabled:
        return labeled_df.copy()

    frame = labeled_df.copy()
    whitespace = (
        frame["whitespace_ratio"].astype(float)
        if "whitespace_ratio" in frame.columns
        else pd.Series(0.0, index=frame.index, dtype=float)
    )
    mask = frame["label"].astype(str) == "artifact"
    mask &= frame.get("is_active", True).astype(bool)
    mask &= ~frame.get("cluster_unresolved", False).astype(bool)
    mask &= frame["blue_overlap_ratio"].astype(float) <= 0.02
    mask &= frame["border_touch_ratio"].astype(float) <= cfg.detector.cv_promote_large_single_artifacts_max_border_touch_ratio
    mask &= whitespace <= cfg.detector.cv_promote_large_single_artifacts_max_whitespace_ratio
    mask &= frame["area_px"].astype(float).between(
        cfg.detector.cv_promote_large_single_artifacts_min_area_px,
        cfg.detector.cv_promote_large_single_artifacts_max_area_px,
    )
    mask &= frame["color_score"].astype(float) >= cfg.detector.cv_promote_large_single_artifacts_color_score_min
    mask &= frame["local_contrast"].astype(float) >= cfg.detector.cv_promote_large_single_artifacts_min_local_contrast
    mask &= frame["mean_v"].astype(float) <= cfg.detector.cv_promote_large_single_artifacts_max_mean_v
    mask &= frame["mean_lab_b"].astype(float) <= cfg.detector.cv_promote_large_single_artifacts_max_mean_lab_b
    mask &= frame["aspect_ratio"].astype(float) >= cfg.detector.cv_promote_large_single_artifacts_min_aspect_ratio
    mask &= frame["eccentricity"].astype(float) >= cfg.detector.cv_promote_large_single_artifacts_min_eccentricity
    if "cv_seed_count" in frame.columns:
        mask &= frame["cv_seed_count"].fillna(1.0).astype(float) <= 1.5

    if mask.any():
        frame.loc[mask, "label"] = "pupa"
        frame.loc[mask, "confidence"] = np.maximum(
            frame.loc[mask, "confidence"].astype(float).to_numpy(),
            0.58,
        )
    return frame


def _suppress_border_split_rows(
    labeled_df: pd.DataFrame,
    *,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_suppress_border_split_pupae_enabled:
        return labeled_df.copy()

    frame = labeled_df.copy()
    parent_ids = (
        frame["parent_component_id"].astype(str)
        if "parent_component_id" in frame.columns
        else pd.Series("", index=frame.index, dtype=str)
    )
    mask = frame["label"].astype(str) == "pupa"
    mask &= frame.get("is_active", True).astype(bool)
    mask &= frame.get("touches_image_border", False).astype(bool)
    mask &= parent_ids.ne("")
    mask &= frame["cv_seed_count"].astype(float) >= float(cfg.detector.cv_suppress_border_split_pupae_min_seed_count)
    mask &= frame["border_touch_ratio"].astype(float) >= cfg.detector.cv_suppress_border_split_pupae_min_border_touch_ratio
    mask &= frame["color_score"].astype(float) <= cfg.detector.cv_suppress_border_split_pupae_max_color_score
    if mask.any():
        frame.loc[mask, "label"] = "artifact"
        frame.loc[mask, "confidence"] = np.minimum(
            frame.loc[mask, "confidence"].astype(float).to_numpy(),
            0.49,
        )
    return frame


def _post_resplit_labeled_rows(
    labeled_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    feature_image: np.ndarray,
    blue_mask: np.ndarray | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_post_resplit_enabled:
        return labeled_df.copy()

    single_area, single_major, single_minor = _estimate_labeled_scale(labeled_df, cfg)
    kept_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []

    for _, row in labeled_df.iterrows():
        row_dict = row.to_dict()
        label = str(row_dict.get("label", ""))
        seed_count = int(row_dict.get("cv_seed_count", 1) or 1)
        if (
            not row_dict.get("is_active", True)
            or label not in {"cluster", "uncertain"}
            or seed_count < cfg.detector.cv_post_resplit_min_seed_count
        ):
            kept_rows.append(row_dict)
            continue

        child_components = _deblend_component(
            row,
            score_image,
            foreground_mask,
            single_area,
            single_major,
            single_minor,
            cfg,
            dense_deblend=bool(row_dict.get("cv_dense_deblend", False)),
        )
        if child_components and len(child_components) >= 2:
            child_rows.extend(child_components)
            continue
        kept_rows.append(row_dict)

    if not child_rows:
        return labeled_df.copy()

    kept_df = pd.DataFrame(kept_rows)
    child_df = pd.DataFrame(child_rows)
    child_labeled = _reclassify_component_frame(child_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg)
    combined = pd.concat([kept_df, child_labeled], ignore_index=True)
    return _ensure_cv_debug_columns(combined).sort_values("component_id").reset_index(drop=True)


def _dense_patch_supplement_from_labeled(
    labeled_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    blue_mask: np.ndarray | None,
    paper_bounds: PaperBounds | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    if labeled_df.empty or not cfg.detector.cv_patch_supplement_enabled:
        return pd.DataFrame()

    pupa_df = labeled_df.loc[
        labeled_df["label"].astype(str).eq("pupa") & labeled_df.get("is_active", True).astype(bool)
    ].copy()
    if len(pupa_df) < 6:
        return pd.DataFrame()

    single_area, single_major, single_minor = _estimate_labeled_scale(labeled_df, cfg)
    points = pupa_df[["centroid_y", "centroid_x"]].astype(float).to_numpy()
    eps = max(8.0, single_major * 1.8)
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(points)

    extras: list[dict[str, object]] = []
    for cluster_id in sorted(set(int(label) for label in labels if int(label) >= 0)):
        cluster_df = pupa_df.loc[labels == cluster_id].copy()
        if len(cluster_df) < 5:
            continue
        patch_y0, patch_x0, patch_y1, patch_x1 = _component_patch_bounds(
            pd.Series(
                {
                    "bbox_y0": cluster_df["bbox_y0"].astype(float).min(),
                    "bbox_x0": cluster_df["bbox_x0"].astype(float).min(),
                    "bbox_y1": cluster_df["bbox_y1"].astype(float).max(),
                    "bbox_x1": cluster_df["bbox_x1"].astype(float).max(),
                }
            ),
            score_image.shape[:2],
            max(6, int(round(single_major * 0.8))),
        )
        patch_score = score_image[patch_y0:patch_y1, patch_x0:patch_x1]
        if patch_score.size == 0:
            continue
        patch_fg = foreground_mask[patch_y0:patch_y1, patch_x0:patch_x1].copy()
        if blue_mask is not None and blue_mask.size:
            patch_fg &= blue_mask[patch_y0:patch_y1, patch_x0:patch_x1] == 0
        if paper_bounds is not None:
            local_bounds = _localize_paper_bounds(
                paper_bounds,
                patch_y0=patch_y0,
                patch_x0=patch_x0,
                patch_shape=patch_score.shape,
            )
            if local_bounds is not None:
                patch_fg &= _paper_mask(patch_score.shape, local_bounds)
        if not np.any(patch_fg):
            continue

        response, angle_indexes = _matched_response(
            patch_score,
            single_major * cfg.detector.cv_filter_major_px_scale,
            single_minor * cfg.detector.cv_filter_minor_px_scale,
            cfg.detector.cv_filter_angles_deg,
        )
        min_distance = max(2, int(round(single_minor * cfg.detector.cv_peak_min_distance_minor_scale)))
        peaks = _merge_peak_sets(
            [
                _ranked_peaks(response, patch_fg, single_minor, cfg),
                _multithresh_peaks(response, patch_fg, single_minor, cfg),
                _distance_peaks(patch_fg, single_minor, cfg),
            ],
            score_image=response,
            min_distance=min_distance,
        )
        if len(peaks) == 0:
            continue

        patch_existing = cluster_df.loc[
            (cluster_df["centroid_y"].astype(float) >= patch_y0)
            & (cluster_df["centroid_y"].astype(float) < patch_y1)
            & (cluster_df["centroid_x"].astype(float) >= patch_x0)
            & (cluster_df["centroid_x"].astype(float) < patch_x1)
        ]
        existing = (
            patch_existing[["centroid_y", "centroid_x"]].astype(float).to_numpy()
            if not patch_existing.empty
            else np.empty((0, 2), dtype=float)
        )
        max_extra = max(1, int(math.ceil(len(cluster_df) * cfg.detector.cv_patch_supplement_max_extra_ratio)))
        added = 0
        for peak_row, peak_col in peaks:
            global_row = float(patch_y0 + peak_row)
            global_col = float(patch_x0 + peak_col)
            if existing.size:
                nearest_sq = np.min(np.sum((existing - np.array([global_row, global_col])) ** 2, axis=1))
                if nearest_sq < (single_minor * 0.75) ** 2:
                    continue
            angle_deg = cfg.detector.cv_filter_angles_deg[int(angle_indexes[int(peak_row), int(peak_col)])]
            child_mask = _ellipse_mask(
                patch_score.shape,
                float(peak_row),
                float(peak_col),
                single_major * cfg.detector.cv_filter_major_px_scale,
                single_minor * cfg.detector.cv_filter_minor_px_scale,
                angle_deg,
            )
            child_mask &= patch_fg
            if float(np.sum(child_mask)) < max(float(cfg.detector.cv_min_area_px) * 0.4, single_area * 0.18):
                continue
            row = _build_child_row(
                child_mask,
                patch_y0=patch_y0,
                patch_x0=patch_x0,
                image_shape=score_image.shape[:2],
                component_id=f"cv_patch_{cluster_id:02d}_{added + 1:02d}",
                parent_component_id=f"cv_patch_{cluster_id:02d}",
                seed_count=len(peaks),
                dense_deblend=True,
                center_only=True,
            )
            row["cv_patch_supplement"] = True
            inside_fraction = bbox_fraction_inside_paper_bounds(
                row["bbox_x0"],
                row["bbox_y0"],
                row["bbox_x1"],
                row["bbox_y1"],
                paper_bounds,
            )
            if inside_fraction < 0.75:
                continue
            extras.append(row)
            existing = (
                np.vstack([existing, np.array([global_row, global_col])])
                if existing.size
                else np.asarray([[global_row, global_col]], dtype=float)
            )
            added += 1
            if added >= max_extra:
                break

    if not extras:
        return pd.DataFrame()
    return pd.DataFrame(extras)


def _log_blob_supplement(
    labeled_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    blue_mask: np.ndarray | None,
    paper_bounds: PaperBounds | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Find pupa-like blobs via LoG not covered by any accepted pupa."""
    if labeled_df.empty or not cfg.detector.cv_global_peak_supplement_enabled:
        return pd.DataFrame()

    accepted = labeled_df.loc[
        labeled_df["label"].astype(str).eq("pupa") & labeled_df.get("is_active", True).astype(bool)
    ]
    if accepted.empty:
        return pd.DataFrame()

    single_area, single_major, single_minor = _estimate_labeled_scale(labeled_df, cfg)

    # Run LoG blob detection on the brown score image
    min_sigma = max(1.5, single_minor / 5.0)
    max_sigma = max(min_sigma + 1.0, single_minor / 1.8)
    blobs = feature.blob_log(
        score_image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=6,
        threshold=0.012,
        overlap=0.4,
    )
    if len(blobs) == 0:
        return pd.DataFrame()

    # Build masks
    fg_mask = foreground_mask.copy()
    if blue_mask is not None and blue_mask.size:
        fg_mask &= blue_mask == 0
    if paper_bounds is not None:
        fg_mask &= _paper_mask(score_image.shape[:2], paper_bounds)

    accepted_pts = accepted[["centroid_y", "centroid_x"]].astype(float).to_numpy()
    exclusion_dist = single_minor * cfg.detector.cv_global_peak_exclusion_scale

    # Compute brown_score threshold from accepted pupae centroids
    accepted_scores = np.array([
        float(score_image[int(round(float(y))), int(round(float(x)))])
        for y, x in accepted_pts
        if 0 <= int(round(float(y))) < score_image.shape[0]
        and 0 <= int(round(float(x))) < score_image.shape[1]
    ])
    score_floor = float(np.percentile(accepted_scores, 25)) if len(accepted_scores) >= 4 else 0.12

    # Sort blobs by brown score at center (strongest first)
    blob_scores = np.array([
        float(score_image[int(round(float(b[0]))), int(round(float(b[1])))])
        for b in blobs
    ])
    order = np.argsort(blob_scores)[::-1]
    blobs = blobs[order]
    blob_scores = blob_scores[order]

    extras: list[dict[str, object]] = []
    max_extras = max(1, int(round(len(accepted) * cfg.detector.cv_global_peak_max_extra_ratio)))

    for blob, bs in zip(blobs, blob_scores):
        by, bx, bsigma = float(blob[0]), float(blob[1]), float(blob[2])
        iby, ibx = int(round(by)), int(round(bx))

        # Quality checks
        if bs < max(score_floor, cfg.detector.cv_global_peak_min_brown_score):
            continue
        if iby < 0 or iby >= score_image.shape[0] or ibx < 0 or ibx >= score_image.shape[1]:
            continue
        if not fg_mask[iby, ibx]:
            continue

        # Exclusion distance from existing pupae
        if accepted_pts.size > 0:
            nearest = float(np.min(np.sqrt(np.sum(
                (accepted_pts - np.array([by, bx])) ** 2, axis=1
            ))))
            if nearest < exclusion_dist:
                continue

        # Create circular mask (LoG blobs are isotropic)
        blob_radius = bsigma * math.sqrt(2)
        child_mask = _ellipse_mask(
            score_image.shape,
            by, bx,
            blob_radius * 2.5,
            blob_radius * 2.5,
            0.0,
        )
        child_mask &= fg_mask
        if float(np.sum(child_mask)) < max(float(cfg.detector.cv_min_area_px) * 0.35, single_area * 0.15):
            continue

        inside_fraction = bbox_fraction_inside_paper_bounds(
            int(np.min(np.where(child_mask)[1])),
            int(np.min(np.where(child_mask)[0])),
            int(np.max(np.where(child_mask)[1])) + 1,
            int(np.max(np.where(child_mask)[0])) + 1,
            paper_bounds,
        )
        if inside_fraction < 0.75:
            continue

        row = _build_child_row(
            child_mask,
            patch_y0=0,
            patch_x0=0,
            image_shape=score_image.shape[:2],
            component_id="cv_log_{:04d}".format(len(extras) + 1),
            parent_component_id="cv_log",
            seed_count=1,
            dense_deblend=False,
            center_only=True,
        )
        row["cv_global_peak_supplement"] = True
        extras.append(row)
        accepted_pts = np.vstack([accepted_pts, np.array([by, bx])])
        if len(extras) >= max_extras:
            break

    if not extras:
        return pd.DataFrame()
    return pd.DataFrame(extras)


def _deduplicate_close_pupae(
    labeled_df: pd.DataFrame,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Remove tiny duplicate pupae that are too close to a larger accepted pupa."""
    if labeled_df.empty:
        return labeled_df.copy()

    pupa_mask = (
        labeled_df["label"].astype(str).eq("pupa")
        & labeled_df.get("is_active", True).astype(bool)
    )
    if pupa_mask.sum() < 2:
        return labeled_df.copy()

    single_area, _, single_minor = _estimate_labeled_scale(labeled_df, cfg)
    # Only remove tiny fragments (<15px area) that are very close to a real pupa
    fragment_area_max = 15.0
    dedup_dist = single_minor * 0.5

    pupa_df = labeled_df.loc[pupa_mask].copy()
    pts = pupa_df[["centroid_y", "centroid_x"]].astype(float).to_numpy()
    areas = pupa_df["area_px"].astype(float).to_numpy()

    drop_indices: set[int] = set()
    for i in range(len(pts)):
        if areas[i] > fragment_area_max:
            continue  # Only consider tiny fragments
        if pupa_df.index[i] in drop_indices:
            continue
        # Check if there's a larger pupa nearby
        for j in range(len(pts)):
            if i == j or pupa_df.index[j] in drop_indices:
                continue
            if areas[j] <= areas[i]:
                continue
            dist = float(np.sqrt(np.sum((pts[i] - pts[j]) ** 2)))
            if dist < dedup_dist:
                drop_indices.add(pupa_df.index[i])
                break

    if not drop_indices:
        return labeled_df.copy()

    frame = labeled_df.copy()
    frame.loc[list(drop_indices), "label"] = "artifact"
    frame.loc[list(drop_indices), "is_active"] = False
    return frame


def _voronoi_reassign_masks(
    labeled_df: pd.DataFrame,
    foreground_mask: np.ndarray,
    score_image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Expand accepted pupa masks to cover the full foreground region via Voronoi.

    Uses the brown score image (if provided) for a more generous foreground
    that better captures pupa edges.
    """
    if labeled_df.empty:
        return labeled_df.copy()

    accepted_mask = (
        labeled_df["label"].astype(str).eq("pupa")
        & labeled_df.get("is_active", True).astype(bool)
    )
    if not accepted_mask.any():
        return labeled_df.copy()

    # Build a generous foreground: union of brown_mask AND soft brown_score
    generous_fg = foreground_mask.copy()
    if score_image is not None:
        # Lower threshold to capture pupa edges the strict mask missed
        generous_fg = generous_fg | (score_image >= 0.04)
        # Smooth to fill small gaps within pupae
        kernel = np.ones((3, 3), dtype=np.uint8)
        generous_fg = cv2.morphologyEx(
            generous_fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2
        ).astype(bool)

    h, w = generous_fg.shape[:2]
    markers = np.zeros((h, w), dtype=np.int32)
    idx_to_iloc: dict[int, int] = {}
    centroids: list[tuple[float, float]] = []

    for marker_idx, (df_idx, row) in enumerate(
        labeled_df.loc[accepted_mask].iterrows(), start=1
    ):
        cy = int(round(float(row["centroid_y"])))
        cx = int(round(float(row["centroid_x"])))
        cy = max(0, min(cy, h - 1))
        cx = max(0, min(cx, w - 1))
        markers[cy, cx] = marker_idx
        idx_to_iloc[marker_idx] = df_idx
        centroids.append((float(row["centroid_y"]), float(row["centroid_x"])))

    # Estimate max radius from pupa scale
    single_area, single_major, _ = _estimate_labeled_scale(labeled_df, AppConfig())
    max_radius = max(single_major * 1.5, 25.0)

    distance = ndi.distance_transform_edt(generous_fg)
    ws_labels = segmentation.watershed(-distance, markers, mask=generous_fg)

    frame = labeled_df.copy()
    for marker_idx, df_idx in idx_to_iloc.items():
        region = ws_labels == marker_idx
        if not np.any(region):
            continue

        # Clip to max_radius from centroid to prevent elongated masks
        cy, cx = centroids[marker_idx - 1]
        ys_all, xs_all = np.where(region)
        dist_from_center = np.sqrt((ys_all - cy) ** 2 + (xs_all - cx) ** 2)
        far = dist_from_center > max_radius
        if far.any():
            region[ys_all[far], xs_all[far]] = False

        if not np.any(region):
            continue

        # Erode by 1px to create visible gaps between adjacent pupa masks,
        # then smooth the contour for cleaner outlines.
        region_u8 = region.astype(np.uint8) * 255
        region_u8 = cv2.erode(region_u8, np.ones((3, 3), np.uint8), iterations=1)
        region_u8 = cv2.morphologyEx(region_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        region = region_u8 > 0
        if not np.any(region):
            continue

        ys, xs = np.where(region)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        local_mask = region[y0:y1, x0:x1]

        frame.at[df_idx, "mask"] = local_mask.astype(bool)
        frame.at[df_idx, "bbox_y0"] = y0
        frame.at[df_idx, "bbox_y1"] = y1
        frame.at[df_idx, "bbox_x0"] = x0
        frame.at[df_idx, "bbox_x1"] = x1
        frame.at[df_idx, "area_px"] = float(np.sum(local_mask))
        frame.at[df_idx, "cv_center_only"] = False

    return frame


def _attach_local_pupa_density_columns(
    labeled_df: pd.DataFrame,
    *,
    radii_px: tuple[int, int, int] = (40, 60, 80),
) -> pd.DataFrame:
    """Add pupa_nbr_40, pupa_nbr_60, pupa_nbr_80 columns via KDTree on accepted pupa centroids."""
    frame = labeled_df.copy()
    col_names = [f"pupa_nbr_{r}" for r in radii_px]
    for col in col_names:
        frame[col] = 0

    accepted_mask = (
        frame["label"].astype(str).eq("pupa")
        & frame.get("is_active", True).astype(bool)
    )
    if accepted_mask.sum() < 1:
        return frame

    accepted_pts = frame.loc[accepted_mask, ["centroid_y", "centroid_x"]].astype(float).to_numpy()
    tree = cKDTree(accepted_pts)

    all_pts = frame[["centroid_y", "centroid_x"]].astype(float).to_numpy()
    for radius, col in zip(radii_px, col_names):
        counts = tree.query_ball_point(all_pts, r=float(radius))
        raw_counts = np.array([len(c) for c in counts], dtype=int)
        # Subtract self for rows that are themselves accepted pupae
        self_correction = accepted_mask.to_numpy().astype(int)
        frame[col] = np.maximum(raw_counts - self_correction, 0)

    return frame


def _suppress_weak_split_children_rows(
    labeled_df: pd.DataFrame,
    *,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Relabel tiny weak split children as artifact."""
    if labeled_df.empty or not cfg.detector.cv_weak_child_suppress_enabled:
        return labeled_df.copy()

    frame = labeled_df.copy()
    parent_ids = (
        frame["parent_component_id"].astype(str)
        if "parent_component_id" in frame.columns
        else pd.Series("", index=frame.index, dtype=str)
    )
    has_parent = parent_ids.ne("") & parent_ids.ne("None") & parent_ids.ne("nan") & parent_ids.notna()

    pupa_nbr_60 = (
        frame["pupa_nbr_60"].astype(float)
        if "pupa_nbr_60" in frame.columns
        else pd.Series(0.0, index=frame.index, dtype=float)
    )

    mask = frame["label"].astype(str).eq("pupa")
    mask &= frame.get("is_active", True).astype(bool)
    mask &= has_parent
    mask &= frame["area_px"].astype(float) <= cfg.detector.cv_weak_child_suppress_max_area_px
    mask &= frame["color_score"].astype(float) <= cfg.detector.cv_weak_child_suppress_max_color_score
    mask &= frame["local_contrast"].astype(float) <= cfg.detector.cv_weak_child_suppress_max_local_contrast
    mask &= pupa_nbr_60 >= float(cfg.detector.cv_weak_child_suppress_min_neighbors_60)

    if mask.any():
        frame.loc[mask, "label"] = "artifact"
        frame.loc[mask, "confidence"] = np.minimum(
            frame.loc[mask, "confidence"].astype(float).to_numpy(),
            0.49,
        )
    return frame


def _score_global_supplement_candidate_row(row: pd.Series) -> float:
    """Compute a lightweight supplement score for a non-pupa candidate row."""
    score = 0.0
    color = float(row.get("color_score", 0.0))
    conf = float(row.get("confidence", 0.0))
    area = float(row.get("area_px", 0.0))
    solidity = float(row.get("solidity", 0.0))
    ecc = float(row.get("eccentricity", 0.0))
    aspect = float(row.get("aspect_ratio", 0.0))
    whitespace = float(row.get("whitespace_ratio", 0.5))
    local_contrast = float(row.get("local_contrast", 0.0))
    mean_lab_b = float(row.get("mean_lab_b", 150.0))

    # Color contribution: strong brown signature
    if color >= 0.40:
        score += min((color - 0.30) * 3.0, 1.5)
    # Confidence contribution
    if conf >= 0.45:
        score += min((conf - 0.35) * 2.0, 1.0)
    # Area contribution: favor reasonable single-pupa sized components
    if 50.0 <= area <= 400.0:
        score += 0.5
    elif 30.0 <= area <= 500.0:
        score += 0.25
    # Shape: pupa-like solidity and eccentricity
    if solidity >= 0.70:
        score += 0.3
    if ecc >= 0.55:
        score += 0.2
    if aspect >= 1.2:
        score += 0.2
    # Low whitespace is good
    if whitespace <= 0.30:
        score += 0.3
    # Contrast bonus
    if local_contrast >= 15.0:
        score += 0.3
    # Lab b penalty for yellow-ish blobs
    if mean_lab_b <= 155.0:
        score += 0.2

    return score


def _supplement_global_candidates(
    labeled_df: pd.DataFrame,
    *,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Promote existing artifact/uncertain/cluster rows that look like standalone pupae."""
    if labeled_df.empty or not cfg.detector.cv_global_candidate_supplement_enabled:
        return labeled_df.copy()

    frame = labeled_df.copy()

    # Identify accepted pupa centroids for distance and NMS checks
    accepted_mask = (
        frame["label"].astype(str).eq("pupa")
        & frame.get("is_active", True).astype(bool)
    )
    accepted_pts = (
        frame.loc[accepted_mask, ["centroid_y", "centroid_x"]].astype(float).to_numpy()
        if accepted_mask.any()
        else np.empty((0, 2), dtype=float)
    )

    pupa_nbr_60 = (
        frame["pupa_nbr_60"].astype(float)
        if "pupa_nbr_60" in frame.columns
        else pd.Series(0.0, index=frame.index, dtype=float)
    )

    parent_ids = (
        frame["parent_component_id"].astype(str)
        if "parent_component_id" in frame.columns
        else pd.Series("", index=frame.index, dtype=str)
    )
    has_parent = parent_ids.ne("") & parent_ids.ne("None") & parent_ids.ne("nan") & parent_ids.notna()
    center_only = (
        frame["cv_center_only"].astype(bool)
        if "cv_center_only" in frame.columns
        else pd.Series(False, index=frame.index, dtype=bool)
    )

    # Candidate pool: top-level artifact/uncertain rows only.
    # Child/center-only rows were a major source of overcount in 20/60/65.
    pool_mask = frame["label"].astype(str).isin(["artifact", "uncertain"])
    pool_mask &= frame.get("is_active", True).astype(bool)
    pool_mask &= frame["blue_overlap_ratio"].astype(float) <= 0.05
    pool_mask &= frame["border_touch_ratio"].astype(float) <= 0.20
    pool_mask &= pupa_nbr_60 <= float(cfg.detector.cv_global_candidate_supplement_max_neighbors_60)
    pool_mask &= ~has_parent
    pool_mask &= ~center_only

    candidate_indices = frame.index[pool_mask].tolist()
    if not candidate_indices:
        return frame

    # Score each candidate
    scored: list[tuple[int, float]] = []
    min_dist_px = cfg.detector.cv_global_candidate_supplement_min_distance_px
    for idx in candidate_indices:
        row = frame.loc[idx]
        # Distance from nearest accepted pupa
        if accepted_pts.size > 0:
            pt = np.array([float(row["centroid_y"]), float(row["centroid_x"])])
            nearest_dist = float(np.min(np.sqrt(np.sum((accepted_pts - pt) ** 2, axis=1))))
            if nearest_dist < min_dist_px:
                continue
        s = _score_global_supplement_candidate_row(row)
        if s >= cfg.detector.cv_global_candidate_supplement_score_min:
            scored.append((idx, s))

    if not scored:
        return frame

    # Sort by score descending and greedily accept with NMS
    scored.sort(key=lambda item: item[1], reverse=True)
    nms_px = cfg.detector.cv_global_candidate_supplement_nms_px
    promoted_pts: list[np.ndarray] = []
    if accepted_pts.size > 0:
        for pt in accepted_pts:
            promoted_pts.append(pt)

    promote_indices: list[int] = []
    for idx, _s in scored:
        row = frame.loc[idx]
        pt = np.array([float(row["centroid_y"]), float(row["centroid_x"])])
        # NMS: skip if too close to already-promoted or accepted
        too_close = False
        for prev_pt in promoted_pts:
            if float(np.sqrt(np.sum((pt - prev_pt) ** 2))) < nms_px:
                too_close = True
                break
        if too_close:
            continue
        promote_indices.append(idx)
        promoted_pts.append(pt)

    if promote_indices:
        frame.loc[promote_indices, "label"] = "pupa"
        frame.loc[promote_indices, "confidence"] = np.maximum(
            frame.loc[promote_indices, "confidence"].astype(float).to_numpy(),
            0.60,
        )

    return frame


def refine_labeled_candidates(
    labeled_df: pd.DataFrame,
    *,
    score_image: np.ndarray,
    foreground_mask: np.ndarray,
    feature_image: np.ndarray,
    blue_mask: np.ndarray | None,
    paper_bounds: PaperBounds | None,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Refine labeled CV candidates with second-pass split and dense-patch extras."""
    if labeled_df.empty:
        return labeled_df.copy()

    frame = _ensure_cv_debug_columns(labeled_df)
    frame = _post_resplit_labeled_rows(
        frame,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        feature_image=feature_image,
        blue_mask=blue_mask,
        cfg=cfg,
    )
    frame = _resplit_strong_artifact_rows(
        frame,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        feature_image=feature_image,
        blue_mask=blue_mask,
        cfg=cfg,
    )
    frame = _resplit_large_cluster_rows(
        frame,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        feature_image=feature_image,
        blue_mask=blue_mask,
        cfg=cfg,
    )
    frame = _resplit_large_pupa_rows(
        frame,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        feature_image=feature_image,
        blue_mask=blue_mask,
        cfg=cfg,
    )
    # --- New modules: density, weak child suppression, global supplement ---
    frame = _attach_local_pupa_density_columns(frame)
    frame = _pairlike_resplit_pupa_rows(
        frame,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        feature_image=feature_image,
        blue_mask=blue_mask,
        cfg=cfg,
    )
    frame = _attach_local_pupa_density_columns(frame)
    frame = _suppress_weak_split_children_rows(frame, cfg=cfg)
    frame = _supplement_global_candidates(frame, cfg=cfg)
    # --- End new modules ---
    frame = _promote_uncertain_rows(
        frame,
        confidence_min=cfg.detector.cv_promote_uncertain_confidence_min,
        color_score_min=cfg.detector.cv_promote_uncertain_color_score_min,
        max_mean_v=cfg.detector.cv_promote_uncertain_max_mean_v,
        max_mean_lab_b=cfg.detector.cv_promote_uncertain_max_mean_lab_b,
        min_seed_count=cfg.detector.cv_post_resplit_min_seed_count,
    )

    patch_df = _dense_patch_supplement_from_labeled(
        frame,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        blue_mask=blue_mask,
        paper_bounds=paper_bounds,
        cfg=cfg,
    )
    if not patch_df.empty:
        patch_labeled = _reclassify_component_frame(patch_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg)
        patch_labeled = _ensure_cv_debug_columns(patch_labeled)
        frame = pd.concat([frame, patch_labeled], ignore_index=True)
        frame = _promote_uncertain_rows(
            frame,
            confidence_min=cfg.detector.cv_patch_supplement_promote_uncertain_confidence_min,
            color_score_min=cfg.detector.cv_patch_supplement_promote_uncertain_color_score_min,
            max_mean_v=cfg.detector.cv_patch_supplement_max_mean_v,
            max_mean_lab_b=cfg.detector.cv_patch_supplement_max_mean_lab_b,
            restrict_to_patch_supplement=True,
        )
    frame = _promote_pairlike_artifact_rows(frame, cfg=cfg)
    frame = _promote_strong_single_artifact_rows(frame, cfg=cfg)
    frame = _promote_large_single_artifact_rows(frame, cfg=cfg)
    # General uncertain promotion (no seed_count restriction) rescues
    # single-component uncertains with strong pupa-like features.
    frame = _promote_uncertain_rows(
        frame,
        confidence_min=cfg.detector.cv_promote_general_uncertain_confidence_min,
        color_score_min=cfg.detector.cv_promote_general_uncertain_color_score_min,
        max_mean_v=cfg.detector.cv_promote_general_uncertain_max_mean_v,
        max_mean_lab_b=cfg.detector.cv_promote_general_uncertain_max_mean_lab_b,
        min_seed_count=None,
    )
    # LoG blob supplement: find pupa-like blobs via Laplacian of Gaussian
    # not covered by any accepted pupa. Better than matched filter for
    # detecting individual blobs within dense clusters.
    gpeak_df = _log_blob_supplement(
        frame,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        blue_mask=blue_mask,
        paper_bounds=paper_bounds,
        cfg=cfg,
    )
    if not gpeak_df.empty:
        gpeak_labeled = _reclassify_component_frame(
            gpeak_df, feature_image=feature_image, blue_mask=blue_mask, cfg=cfg,
        )
        gpeak_labeled = _ensure_cv_debug_columns(gpeak_labeled)
        frame = pd.concat([frame, gpeak_labeled], ignore_index=True)
        # Promote uncertain global peaks with strong features
        frame = _promote_uncertain_rows(
            frame,
            confidence_min=0.50,
            color_score_min=0.48,
            max_mean_v=175.0,
            max_mean_lab_b=160.0,
            min_seed_count=None,
        )
    frame = _suppress_border_split_rows(frame, cfg=cfg)
    # Deduplicate close pupa pairs: remove tiny fragments from over-splitting
    frame = _deduplicate_close_pupae(frame, cfg)
    return frame.sort_values("component_id").reset_index(drop=True)


def detect_instances(
    image: np.ndarray,
    cfg: AppConfig,
    *,
    source_type: str,
    blue_mask: np.ndarray | None = None,
    paper_bounds: PaperBounds | None = None,
    component_prefix: str = "cv",
) -> pd.DataFrame:
    """Detect pupa-like instances with a fast pure-CV backbone."""
    original_shape = image.shape[:2]
    work_image, work_blue_mask, work_paper_bounds, scale = _maybe_downscale(
        image,
        blue_mask,
        paper_bounds,
        cfg.detector.cv_max_side_px,
    )
    score_image, foreground_mask = _binary_foreground(work_image, work_blue_mask, work_paper_bounds, cfg)
    component_df = extract_components((foreground_mask.astype(np.uint8) * 255), cfg)
    if component_df.empty:
        return component_df

    return _refine_component_dataframe(
        component_df,
        score_image=score_image,
        foreground_mask=foreground_mask.astype(bool),
        blue_mask=work_blue_mask,
        paper_bounds=work_paper_bounds,
        cfg=cfg,
        component_prefix=component_prefix,
        scale=scale,
        original_shape=original_shape,
    )
