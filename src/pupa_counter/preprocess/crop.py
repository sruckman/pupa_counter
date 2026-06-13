"""Scanner border cropping."""

from __future__ import annotations

import numpy as np

from pupa_counter.config import AppConfig


def _scan_edge(dark_ratio: np.ndarray, threshold_ratio: float, max_crop: int, reverse: bool = False) -> int:
    run = 0
    if reverse:
        for index in range(len(dark_ratio) - 1, max(len(dark_ratio) - max_crop - 1, -1), -1):
            if dark_ratio[index] >= threshold_ratio:
                run += 1
            else:
                break
        return run
    for index in range(min(max_crop, len(dark_ratio))):
        if dark_ratio[index] >= threshold_ratio:
            run += 1
        else:
            break
    return run


def crop_scanner_border(image: np.ndarray, cfg: AppConfig = None) -> np.ndarray:
    cfg = cfg or AppConfig()
    if not cfg.preprocess.auto_crop_black_border:
        return image
    gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)
    dark_threshold = cfg.preprocess.crop_dark_threshold
    threshold_ratio = cfg.preprocess.crop_min_run_ratio
    max_row_crop = max(1, int(image.shape[0] * cfg.preprocess.max_border_crop_fraction))
    max_col_crop = max(1, int(image.shape[1] * cfg.preprocess.max_border_crop_fraction))

    row_dark_ratio = (gray < dark_threshold).mean(axis=1)
    col_dark_ratio = (gray < dark_threshold).mean(axis=0)

    top = _scan_edge(row_dark_ratio, threshold_ratio, max_row_crop, reverse=False)
    bottom = _scan_edge(row_dark_ratio, threshold_ratio, max_row_crop, reverse=True)
    left = _scan_edge(col_dark_ratio, threshold_ratio, max_col_crop, reverse=False)
    right = _scan_edge(col_dark_ratio, threshold_ratio, max_col_crop, reverse=True)

    end_row = image.shape[0] - bottom if bottom > 0 else image.shape[0]
    end_col = image.shape[1] - right if right > 0 else image.shape[1]
    cropped = image[top:end_row, left:end_col]
    if cropped.size == 0:
        return image
    return cropped
