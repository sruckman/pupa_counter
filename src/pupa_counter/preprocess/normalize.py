"""Background normalization."""

from __future__ import annotations

import cv2
import numpy as np

from pupa_counter.config import AppConfig


def normalize_background(image: np.ndarray, cfg: AppConfig = None) -> np.ndarray:
    cfg = cfg or AppConfig()
    normalized = image.astype(np.float32).copy()
    low = cfg.preprocess.background_percentile_low
    high = cfg.preprocess.background_percentile_high

    for channel_index in range(3):
        channel = normalized[:, :, channel_index]
        lo = np.percentile(channel, low)
        hi = np.percentile(channel, high)
        if hi <= lo:
            continue
        channel = (channel - lo) / (hi - lo)
        normalized[:, :, channel_index] = np.clip(channel * 255.0, 0.0, 255.0)

    normalized_uint8 = normalized.astype(np.uint8)
    lab = cv2.cvtColor(normalized_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cfg.preprocess.clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    normalized_lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
