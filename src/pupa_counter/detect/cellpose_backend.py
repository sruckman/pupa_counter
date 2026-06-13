"""Cellpose-based instance segmentation backend.

This is an alternative detection path to the classical brown_mask + watershed
flow. Cellpose is a learned biological-object instance segmenter trained on
diverse cells; the cyto3 model transfers well to drosophila pupae which are
oval, dark, similar in size, and frequently touching.

Activation
----------
Switching is config-driven, no global state. Set in YAML:

    detector:
      backend: cellpose
      cellpose_diameter: null   # null = auto, otherwise integer pixels
      cellpose_max_side_px: 1400  # downscale before inference for speed
      cellpose_flow_threshold: 0.4
      cellpose_cellprob_threshold: 0.0

The classical pipeline is unchanged when ``backend == "classical"``.

Performance notes (CPU, M1)
---------------------------
- Loading cyto3 weights: ~3 s, 25 MB on disk
- Inference on 1400-side image: ~15-25 s
- Inference on full 3300-side image: ~75-95 s

Downscaling is on by default. The diameter estimate scales with the image,
so the model still finds the same instances.
"""

from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.components import build_component_row


def _lazy_model(diam_mean: Optional[float] = None):
    """Import and load Cellpose only when actually requested. Keeps the
    classical pipeline import-light when cellpose is not used."""
    from cellpose import models  # noqa: WPS433

    return models.Cellpose(gpu=False, model_type="cyto3")


_MODEL_CACHE = {}


def _get_model():
    if "model" not in _MODEL_CACHE:
        _MODEL_CACHE["model"] = _lazy_model()
    return _MODEL_CACHE["model"]


def _maybe_downscale(image: np.ndarray, max_side_px: int):
    """Downscale large images so cellpose runs in reasonable time. Returns
    (resized_image, scale_factor) where scale_factor maps pixels back to
    the original coordinate system."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side_px:
        return image, 1.0
    scale = max_side_px / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def detect_instances(
    image: np.ndarray,
    cfg: AppConfig,
    *,
    diameter: Optional[float] = None,
    max_side_px: Optional[int] = None,
    flow_threshold: Optional[float] = None,
    cellprob_threshold: Optional[float] = None,
    component_prefix: str = "cp",
    offset_row: int = 0,
    offset_col: int = 0,
    global_image_shape = None,
) -> pd.DataFrame:
    """Run cellpose and return a components-shaped dataframe.

    The output schema matches what ``extract_components`` produces, so the
    rest of the pipeline (featurize -> rule_filter -> band assignment) can
    consume it without modification.
    """
    cellpose_cfg = cfg.detector
    image_rgb = image
    resized, scale = _maybe_downscale(image_rgb, max_side_px or cellpose_cfg.cellpose_max_side_px)

    diameter = cellpose_cfg.cellpose_diameter if diameter is None else diameter
    if diameter is None or diameter <= 0:
        diameter = None  # let cellpose auto-estimate

    flow_threshold = cellpose_cfg.cellpose_flow_threshold if flow_threshold is None else flow_threshold
    cellprob_threshold = cellpose_cfg.cellpose_cellprob_threshold if cellprob_threshold is None else cellprob_threshold

    model = _get_model()
    masks, _flows, _styles, _diams = model.eval(
        resized,
        diameter=diameter,
        channels=[0, 0],  # grayscale single-channel mode (R only)
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    # masks is a 2-D int array where each integer label > 0 is one instance.
    n_instances = int(masks.max())
    if n_instances == 0:
        return pd.DataFrame()

    # If we downscaled before inference, scale the masks back up so the
    # downstream pipeline operates in the original image coordinate system.
    if scale != 1.0:
        masks = cv2.resize(
            masks.astype(np.int32),
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    rows: List[dict] = []
    full_shape = image_rgb.shape[:2] if global_image_shape is None else global_image_shape
    for instance_id in range(1, n_instances + 1):
        local_mask = masks == instance_id
        if not local_mask.any():
            continue
        ys, xs = np.where(local_mask)
        min_row, max_row = int(ys.min()), int(ys.max() + 1)
        min_col, max_col = int(xs.min()), int(xs.max() + 1)
        cropped = local_mask[min_row:max_row, min_col:max_col]
        component_id = "%s_%05d" % (component_prefix, instance_id)
        try:
            row = build_component_row(
                cropped,
                offset_row=offset_row + min_row,
                offset_col=offset_col + min_col,
                image_shape=full_shape,
                component_id=component_id,
            )
        except ValueError:
            continue
        row["image_height"] = int(full_shape[0])
        row["image_width"] = int(full_shape[1])
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values("component_id").reset_index(drop=True)
