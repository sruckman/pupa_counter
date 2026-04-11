"""Top-level fresh peak-first detector.

Glues together ``preprocess`` → ``response`` → ``peaks`` → ``resolver_cv`` →
``geometry`` into a single ``run_detector`` entry point.

Rules that live here, not in submodules:

* No ``v8`` / no ``cellpose`` ever imported.
* Output is a pandas frame with the columns the audit harness needs, plus a
  side dictionary of debug maps for overlay generation.
* All coordinates in the returned frame are in *native* image pixels even
  when internal processing happened at a downscaled resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .preprocess import build_blue_mask, downscale, load_image_rgb
from .response import build_allowed_mask, compute_response_map
from .peaks import PeakConfig, detect_peaks
from .resolver_cv import (
    ComponentSplitConfig,
    detect_peaks_by_component,
    tag_resolver_type,
)
from .geometry import assign_bands


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    # Runtime / scale
    work_scale: float = 0.67
    """Downscale factor applied before the response map. The fresh-start
    experiment hit ``recall ≈ 0.795`` at ``precision ≈ 0.929`` at 0.67x with
    ~330 ms/image, which is the reason this is the v0 default."""

    # Response / allowed mask
    smooth_sigma: float = 1.2
    allowed_abs_threshold: float = 0.12
    """Permissive foreground gate. Must be *lower* than ``peak_abs_score_threshold``
    because the allowed mask is the gate the peak detector looks inside; peaks
    with response between ``allowed_abs_threshold`` and ``peak_abs_score_threshold``
    are intentionally reachable but rejected."""
    allowed_min_percentile: float = 0.0
    """Disabled by default. The absolute threshold is already permissive
    enough for every scan in the 20-image benchmark."""

    # Peak NMS
    peak_min_distance_px: int = 10
    """In *detector-resolution* pixels (i.e. already accounting for
    ``work_scale``). ``10`` at 0.67x ≈ 15 native px, roughly the observed
    teacher nearest-neighbor p25; anything larger silently drops ~20%
    of teacher instances on touching-pupae images."""
    peak_abs_score_threshold: float = 0.22
    """At teacher centroids the response map has ``p10 ≈ 0.27`` / ``median ≈ 0.31``;
    ``0.22`` is a few percent below p10 to leave margin for borderline pupae."""
    peak_plateau_erosion_radius: int = 1
    peak_edge_margin_px: int = 4

    # v1 component split resolver
    use_component_split: bool = False
    """When True, replace the single global ``detect_peaks`` call with a
    per-component ``detect_peaks_by_component`` call so dense blobs can
    produce multiple peaks. This is the v1 watershed-lite resolver."""
    component_single_pupa_area_px: float = 230.0
    component_area_ratio_threshold: float = 1.35
    component_min_peak_distance_px: int = 5
    component_abs_score_threshold: float = 0.20
    component_min_component_area_px: int = 60
    component_max_peaks: int = 20

    # v2 — distance transform assisted splitting
    component_use_distance_transform: bool = False
    component_dt_weight: float = 0.60
    component_dt_min_distance_px: int = 4
    component_dt_min_edt_px: float = 3.0
    component_dt_expected_k_slack: float = 1.25
    component_dedup_edt_min_distance_px: int = 6

    # v2 — erosion core counting
    component_use_erosion_core_count: bool = False
    component_erosion_radius_px: int = 2
    component_erosion_min_core_area_px: int = 20
    component_erosion_k_margin: int = 0

    # v2 — response core mask
    component_use_response_core_mask: bool = False
    component_response_core_threshold: float = 0.26
    component_response_core_min_area_px: int = 15
    component_response_core_k_margin: int = 0

    # Export book-keeping
    instance_source: str = "fresh_peak_v0"
    detector_backend: str = "fresh_peak_v0"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class DetectorOutput:
    instances: pd.DataFrame
    """Per-pupa rows in *native* image coordinates."""

    debug: Dict[str, np.ndarray] = field(default_factory=dict)
    """Intermediate images keyed by name: ``rgb_native``, ``rgb_work``,
    ``blue_mask``, ``response_map``, ``allowed_mask``, ``peak_map``."""

    runtime_ms: float = 0.0
    """Wall-clock time spent inside ``run_detector`` for this image."""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_INSTANCE_COLUMNS = [
    "image_id",
    "source_path",
    "instance_source",
    "component_id",
    "centroid_x",
    "centroid_y",
    "score",
    "status",
    "resolver_type",
    "band",
    "is_top_5pct",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "major_axis_px",
    "minor_axis_px",
    "detector_backend",
]


def _build_instance_frame(
    peaks_native: pd.DataFrame,
    *,
    image_id: str,
    source_path: str,
    cfg: DetectorConfig,
    pseudo_radius_px: float,
) -> pd.DataFrame:
    if peaks_native.empty:
        return pd.DataFrame(columns=_INSTANCE_COLUMNS)

    radius = float(pseudo_radius_px)
    frame = peaks_native.copy()
    frame.rename(columns={"x": "centroid_x", "y": "centroid_y"}, inplace=True)
    frame["image_id"] = image_id
    frame["source_path"] = source_path
    frame["instance_source"] = cfg.instance_source
    frame["component_id"] = [f"peak_{idx:05d}" for idx in range(len(frame))]

    frame["bbox_x0"] = frame["centroid_x"] - radius
    frame["bbox_y0"] = frame["centroid_y"] - radius
    frame["bbox_x1"] = frame["centroid_x"] + radius
    frame["bbox_y1"] = frame["centroid_y"] + radius
    # Peaks don't give us a real axis — use the nominal teacher-scale value
    # so the audit matcher's ``major_axis_scale`` gate has something sane.
    frame["major_axis_px"] = 2.0 * radius
    frame["minor_axis_px"] = radius
    frame["detector_backend"] = cfg.detector_backend

    frame = tag_resolver_type(frame)
    frame = assign_bands(frame)

    for col in _INSTANCE_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame.loc[:, _INSTANCE_COLUMNS].copy()


def run_detector(
    image_path: Path | str,
    *,
    image_id: str | None = None,
    cfg: DetectorConfig | None = None,
    keep_debug: bool = True,
) -> DetectorOutput:
    """Run the fresh peak-first detector on a single image."""
    import time  # local so the module import stays fast

    cfg = cfg or DetectorConfig()

    t0 = time.perf_counter()

    rgb_native = load_image_rgb(image_path)
    rgb_work, scale_used = downscale(rgb_native, cfg.work_scale)

    blue_mask = build_blue_mask(rgb_work)
    response = compute_response_map(
        rgb_work,
        blue_mask=blue_mask,
        smooth_sigma=cfg.smooth_sigma,
    )
    allowed_mask = build_allowed_mask(
        response,
        abs_threshold=cfg.allowed_abs_threshold,
        min_percentile=cfg.allowed_min_percentile,
    )

    if cfg.use_component_split:
        split_cfg = ComponentSplitConfig(
            single_pupa_area_px=cfg.component_single_pupa_area_px,
            min_peak_distance_px=cfg.component_min_peak_distance_px,
            abs_score_threshold=cfg.component_abs_score_threshold,
            min_component_area_px=cfg.component_min_component_area_px,
            area_ratio_threshold=cfg.component_area_ratio_threshold,
            max_peaks_per_component=cfg.component_max_peaks,
            use_distance_transform=cfg.component_use_distance_transform,
            dt_weight=cfg.component_dt_weight,
            dt_min_distance_px=cfg.component_dt_min_distance_px,
            dt_min_edt_px=cfg.component_dt_min_edt_px,
            dt_expected_k_slack=cfg.component_dt_expected_k_slack,
            dedup_edt_min_distance_px=cfg.component_dedup_edt_min_distance_px,
            use_erosion_core_count=cfg.component_use_erosion_core_count,
            erosion_radius_px=cfg.component_erosion_radius_px,
            erosion_min_core_area_px=cfg.component_erosion_min_core_area_px,
            erosion_k_margin=cfg.component_erosion_k_margin,
            use_response_core_mask=cfg.component_use_response_core_mask,
            response_core_threshold=cfg.component_response_core_threshold,
            response_core_min_area_px=cfg.component_response_core_min_area_px,
            response_core_k_margin=cfg.component_response_core_k_margin,
        )
        peaks_work = detect_peaks_by_component(
            response,
            allowed_mask,
            cfg=split_cfg,
            edge_margin_px=cfg.peak_edge_margin_px,
        )
    else:
        peak_cfg = PeakConfig(
            min_distance_px=cfg.peak_min_distance_px,
            abs_score_threshold=cfg.peak_abs_score_threshold,
            plateau_erosion_radius=cfg.peak_plateau_erosion_radius,
            edge_margin_px=cfg.peak_edge_margin_px,
        )
        peaks_work = detect_peaks(response, allowed_mask, cfg=peak_cfg)

    if scale_used != 1.0 and not peaks_work.empty:
        inv = 1.0 / scale_used
        peaks_native = peaks_work.copy()
        peaks_native["x"] = peaks_work["x"].astype(float) * inv
        peaks_native["y"] = peaks_work["y"].astype(float) * inv
    else:
        peaks_native = peaks_work.copy()

    # Use a pseudo-radius in native pixels so instance bboxes make sense
    # against the teacher tables (teacher major axis ~36-42 px at native).
    pseudo_radius_native = 18.0

    instances = _build_instance_frame(
        peaks_native,
        image_id=image_id or Path(image_path).stem,
        source_path=str(image_path),
        cfg=cfg,
        pseudo_radius_px=pseudo_radius_native,
    )

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    debug: Dict[str, np.ndarray] = {}
    if keep_debug:
        peak_map = np.zeros(response.shape, dtype=np.uint8)
        if not peaks_work.empty:
            ys = peaks_work["y"].to_numpy(dtype=int)
            xs = peaks_work["x"].to_numpy(dtype=int)
            inside = (
                (ys >= 0) & (ys < response.shape[0]) & (xs >= 0) & (xs < response.shape[1])
            )
            peak_map[ys[inside], xs[inside]] = 255
        debug = {
            "rgb_native": rgb_native,
            "rgb_work": rgb_work,
            "blue_mask": blue_mask,
            "response_map": response,
            "allowed_mask": allowed_mask,
            "peak_map": peak_map,
        }

    return DetectorOutput(instances=instances, debug=debug, runtime_ms=runtime_ms)
