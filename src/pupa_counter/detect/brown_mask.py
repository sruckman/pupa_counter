"""Brown candidate detection.

Two detection paths:

1. **Color path** (default for annotated PNGs): combines a brown-likelihood
   score with HSV saturation, value, and LAB a* gates. Tuned for the
   characteristic warm brown of pupae photographed on white paper.

2. **Grayscale path** (auto-enabled for desaturated inputs such as cleaned
   PDF scans): drops the saturation gate because dark grayscale pupae have
   near-zero saturation. Relies on darkness + adaptive threshold + a stricter
   value cap to suppress paper noise.

The grayscale path activates automatically when the median image saturation
is below ``brown_detection.grayscale_sat_median_threshold``.
"""

from __future__ import annotations

import cv2
import numpy as np

from pupa_counter.config import AppConfig


def compute_brown_score(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lab_a = np.clip((lab[:, :, 1] - 128.0) / 60.0, 0.0, 1.0)
    lab_b = np.clip((lab[:, :, 2] - 118.0) / 80.0, 0.0, 1.0)
    red_minus_blue = np.clip((image_float[:, :, 0] - image_float[:, :, 2]) / 0.55, 0.0, 1.0)
    red_minus_green = np.clip((image_float[:, :, 0] - image_float[:, :, 1] + 0.08) / 0.35, 0.0, 1.0)
    darkness = np.clip((0.92 - value) / 0.92, 0.0, 1.0)
    # Weights rebalanced (2026-04-07): darkness contribution boosted from 0.20
    # to 0.30 so that desaturated pupae on cleaned scans score above the
    # rule_filter min_color_score threshold of 0.20. Brown pupae still score
    # comfortably above 0.45 — verified by hand on representative pixels.
    return (
        0.20 * saturation
        + 0.30 * darkness
        + 0.18 * lab_a
        + 0.12 * lab_b
        + 0.12 * red_minus_blue
        + 0.08 * red_minus_green
    )


def is_grayscale_image(image: np.ndarray, cfg: AppConfig = None) -> bool:
    """Return True if the image is desaturated enough to need the grayscale path.

    The decision is made on *dark* pixels only. A page-wide median saturation
    is dominated by white background and therefore stays near zero even when
    the (small) ink content is fully colored — it would misclassify any
    paper scan as grayscale. We instead look at the saturation distribution
    of pixels darker than ~150 in V, which is where the actual subject
    (pupae, ink) lives.
    """
    cfg = cfg or AppConfig()
    if not cfg.brown_detection.auto_grayscale_mode:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    dark = hsv[:, :, 2] <= 150
    if int(dark.sum()) < 100:
        # Essentially blank page; treat as not-grayscale so the legacy
        # color path runs. Either path would return an empty mask anyway.
        return False
    dark_sat_median = float(np.median(hsv[:, :, 1][dark]))
    return dark_sat_median < cfg.brown_detection.grayscale_sat_median_threshold


def detect_brown_candidates(image: np.ndarray, blue_mask: np.ndarray = None, cfg: AppConfig = None) -> np.ndarray:
    cfg = cfg or AppConfig()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    brown_score = compute_brown_score(image)

    grayscale_mode = is_grayscale_image(image, cfg)

    score_mask = brown_score >= cfg.brown_detection.brown_score_threshold
    sat_mask = hsv[:, :, 1] >= cfg.brown_detection.min_saturation
    value_mask = hsv[:, :, 2] <= cfg.brown_detection.max_value
    a_mask = lab[:, :, 1] >= cfg.brown_detection.lab_a_min

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    block_size = cfg.brown_detection.adaptive_block_size
    if block_size % 2 == 0:
        block_size += 1
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        cfg.brown_detection.adaptive_c,
    )
    adaptive_mask = adaptive > 0

    # Yellow imprint rejection: brown pupae have lab b* ~140-155, light
    # yellow paper imprints score higher (~160-180) AND have high V
    # (they're light). The user's annotation rule is "棕色的才是卵，浅
    # 黄色的只是印记" (only brown counts; light yellow are just marks).
    # The conjunctive (high b*) AND (high V) keeps dark brown pixels in
    # while removing only the bright yellow ones.
    is_yellow_imprint = (lab[:, :, 2] > cfg.brown_detection.max_lab_b) & (hsv[:, :, 2] > 150)
    not_yellow_mask = ~is_yellow_imprint

    color_path = score_mask & sat_mask & value_mask & (a_mask | adaptive_mask) & not_yellow_mask
    if grayscale_mode:
        # Grayscale supplement: add dark-blob + adaptive-contrast detections
        # without removing anything the color path already found. This keeps
        # mildly-brown pupae (which pass via lab_a) while also catching
        # near-black pupae on cleaned PDFs (which fail the saturation gate).
        # The stricter grayscale_max_value (~180) keeps the supplement from
        # picking up light scan noise.
        dark_value_mask = hsv[:, :, 2] <= cfg.brown_detection.grayscale_max_value
        grayscale_path = dark_value_mask & adaptive_mask & score_mask & not_yellow_mask
        combined = color_path | grayscale_path
    else:
        combined = color_path

    if blue_mask is not None and blue_mask.size > 0:
        combined &= blue_mask == 0

    raw = combined.astype(np.uint8) * 255
    open_kernel = np.ones(
        (cfg.brown_detection.morphology_open_kernel, cfg.brown_detection.morphology_open_kernel), dtype=np.uint8
    )
    close_kernel = np.ones(
        (cfg.brown_detection.morphology_close_kernel, cfg.brown_detection.morphology_close_kernel), dtype=np.uint8
    )
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_kernel)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
    return raw
