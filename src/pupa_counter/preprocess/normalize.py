"""Background normalization."""

from __future__ import annotations

import cv2
import numpy as np

from pupa_counter.config import AppConfig


def build_reference_view(image: np.ndarray, cfg: AppConfig = None) -> np.ndarray:
    """Build a color-preserving reference view for secondary detection paths.

    The main normalized view intentionally boosts local contrast to help the
    primary detector, but that same boost can visually thicken touching pupae
    until narrow gaps disappear. The reference view stays much closer to the
    original crop and only applies a very gentle white balancing so rescue
    paths can preserve narrow gaps without darkening the pupae themselves.
    """
    cfg = cfg or AppConfig()
    reference = image.astype(np.float32).copy()

    for channel_index in range(3):
        channel = reference[:, :, channel_index]
        high = np.percentile(channel, 99.5)
        if high <= 1.0:
            continue
        scale = min(1.08, 250.0 / float(high))
        reference[:, :, channel_index] = np.clip(channel * scale, 0.0, 255.0)

    return reference.astype(np.uint8)


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
