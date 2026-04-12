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

import cv2
import numpy as np
import pandas as pd

from .preprocess import build_blue_mask, downscale, load_image_rgb
from .response import build_allowed_mask, compute_response_map
from .peaks import PeakConfig, detect_peaks
from .paper_roi import PaperROIConfig, apply_paper_roi_to_response, detect_paper_roi
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

    # v3 — paper ROI (rewritten 2026-04-11, but default OFF after hand-label validation)
    use_paper_roi: bool = False
    """v3 status: REWRITTEN but DEFAULT OFF. The flood-fill rewrite is
    better than v1's destructive morphological close, but full-scan
    hand-label validation on 7 scans showed it STILL kills more real
    pupae than the FPs it catches (+2 total errors on 7 hand-labeled
    scans). The v8 teacher doesn't see the regression because of its
    ~3.4% noise floor. Keeping the code available as an opt-in CLI flag
    for further tuning. See docs/V3_PROGRESS_2026-04-11.md §"what
    worked and didn't" for the evidence."""
    paper_roi_brightness_threshold: int = 200
    paper_roi_close_kernel_px: int = 5
    paper_roi_erode_margin_px: int = 2
    paper_roi_min_paper_fraction: float = 0.05
    paper_roi_fill_holes: bool = True

    # v3 — response sharpening mode (adaptive is now default as of 2026-04-11)
    response_mode: str = "adaptive"
    """Which response computation path to use: ``smooth`` (v1 default, plain
    Gaussian blur), ``log`` (single-sigma Laplacian-of-Gaussian approximation),
    ``dog`` (difference-of-Gaussians with independent low/high sigmas), or
    ``adaptive`` (per-component sigma selection, the v3 default — sharper
    response for large components recovers touching-pair peaks)."""
    log_sigma: float = 1.0
    dog_sigma_low: float = 0.8
    dog_sigma_high: float = 2.0
    raw_response_saturation_floor: float = 0.0
    """Hard gate applied to the raw brown response. Default 0.0 = OFF.
    Initially set to 0.06 based on hand_cases (crops) testing, but
    full-scan hand-label validation showed that even 0.06 kills enough
    dim real pupae to register +8 errors on 7 hand-labeled scans. The
    v8 teacher's ~3.4% noise floor masked this regression. Available
    as an opt-in CLI flag for users who want more FP suppression at
    the cost of some recall. See docs/V3_PROGRESS_2026-04-11.md."""
    adaptive_small_sigma: float = 0.6
    adaptive_large_sigma: float = 0.6
    """Changed from 1.4 to 0.6 based on hand-labeled validation (2026-04-11).
    Hand_cases on 7 scans shows lsigma=0.6 gives 11 sum|delta| vs lsigma=1.4's
    17 — a 35% error reduction. The sharper response preserves touching-pair
    peaks that were being merged at lsigma=1.4. Note: this trades precision
    for recall on the full 20-image benchmark (+27 pupae found, +49 FPs), but
    the v8 teacher can't distinguish these changes below its ~3.4% noise
    floor, so the decision is driven by hand-label evidence. See
    docs/DENSE_CLUSTER_RESEARCH_2026-04-11.md §7.5."""
    adaptive_area_threshold_px: int = 500

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

    # v3 component split resolver (on by default as of 2026-04-11)
    use_component_split: bool = True
    """When True, replace the single global ``detect_peaks`` call with a
    per-component ``detect_peaks_by_component`` call so dense blobs can
    produce multiple peaks. This is the v1 watershed-lite resolver.
    Enabled by default in v3 because hand_cases on 7 scans validates it
    as essential for dense cluster recall."""
    component_single_pupa_area_px: float = 200.0
    """v1-tuned. See docs/FRESH_START_PEAK_FIRST_2026-04-10.md — sweep on
    the 5 hard images converged on ``area=200, ratio=1.20, mind=3, thr=0.18``."""
    component_area_ratio_threshold: float = 1.20
    component_min_peak_distance_px: int = 3
    component_abs_score_threshold: float = 0.18
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

    # v4 — resolver v2 (on by default for the force_multi_peak fix)
    use_resolver_v2: bool = True
    """v4 default: ON. Runs the same component splitter as v3 but with
    the ``force_multi_peak_above_area`` fix that targets the rounding
    trap in expected_k computation. Hand-label validation on 10 scans
    shows a clean 17 → 12 sum|err| improvement (−29%) with zero
    regressions. Other v2 knobs (pre_erosion, h_maxima, ceil_k) are
    available but off by default — they hurt hand-labels in testing."""
    resolver_v2_pre_erosion: bool = False
    resolver_v2_pre_erosion_radius_px: int = 1
    resolver_v2_use_h_maxima: bool = False
    resolver_v2_h_maxima_h: float = 0.03
    resolver_v2_use_ceil_expected_k: bool = False
    resolver_v2_force_multi_peak_above_area: float = 280.0
    """v4 fix: force multi-peak mode for components with area >= this
    value, even if ``round(area/single_pupa_area) == 1``. Targets the
    scan_3 #14 case where area=277 rounds to expected_k=1 but
    peak_local_max would have found 2 distinct peaks. Hand-label
    sweep validated 280 as the sweet spot: 260 is too aggressive
    (adds FPs), 300 has no effect. 280 gives 5 clean wins on 10 scans."""

    # v5 — seed completion rescue (experimental, opt-in)
    resolver_v2_use_seed_completion: bool = False
    """If True, underfilled hard components (expected_k > returned peaks)
    get a rescue pass that places the missing centers via EDT+response
    weighted farthest-point selection + 2 weighted Voronoi refinements.
    Addresses the dominant dense-cluster miss failure mode where
    peak_local_max can't find enough maxima inside a merged component."""
    resolver_v2_seed_completion_edt_weight: float = 0.35
    resolver_v2_seed_completion_refine_iters: int = 2
    resolver_v2_seed_completion_min_solidity: float = 0.92
    resolver_v2_seed_completion_min_extent: float = 0.60
    resolver_v2_seed_completion_min_area_ratio: float = 1.8
    resolver_v2_seed_completion_min_anchor_distance_px: float = 11.0
    resolver_v2_seed_completion_min_score_ratio: float = 0.6
    resolver_v2_seed_completion_max_new_seeds: int = 2

    # Paper-brightness filter (v5 — "only detect on white paper")
    min_background_brightness: float = 0.0
    """Minimum median grayscale brightness in an 11×11 work-pixel
    window around each detection. Below this threshold the detection
    is considered to be on a dark background (scanner bar, paper edge,
    shadow artifact) rather than on white paper, and is rejected.
    Set to 0.40 to kill scanner-strip FPs without touching real pupae.
    Default 0.0 = disabled (v4 behavior)."""

    # v3 — hybrid (iter1 + watershed rescue)
    use_hybrid_watershed: bool = False
    """When True, run iter1's component splitter first, then rescue
    under-counted components with the watershed resolver. This preserves
    iter1's strengths (its response-peak detection is very good on
    normal dense clusters) while letting watershed fill in the
    pathological cases where iter1 can't find enough peaks."""

    # v3 — pure EDT-watershed resolver (see docs/DENSE_CLUSTER_RESEARCH_2026-04-11.md)
    use_watershed_split: bool = False
    """When True, replace ``detect_peaks_by_component`` with an EDT-seeded
    watershed resolver. Each allowed-mask component is split by the
    Euclidean distance transform's local maxima, which cleanly separates
    touching pupae without relying on distinct local maxima in the
    smoothed response map. See the research doc for the rationale."""
    watershed_min_edt_seed_value: float = 2.0
    watershed_min_edt_seed_distance_px: int = 3
    watershed_min_basin_area_px: int = 20
    watershed_min_basin_peak_response: float = 0.12
    watershed_ellipse_no_overlap_min_distance_px: int = 14
    """Dr. Long's 'ellipses can't overlap' constraint. After watershed
    basin extraction, any two detections closer than this distance (in
    work pixels) are merged — the higher-score one is kept. Set to 0 to
    disable the constraint."""

    # v3 — ellipse template matching (Dr. Long 2026-04-11 suggestion)
    use_ellipse_template: bool = False
    """When True, compute an oriented ellipse-template cross-correlation on
    the smoothed brown response and use *that* as the input to peak
    detection. The allowed mask is still derived from the raw response, so
    the foreground gate is unchanged — only the thing peaks land on gets
    shape-aware. Dense clusters separate into per-pupa peaks because the
    template only lines up with one pupa at a time; non-pupa brown blobs
    (stains, dirt) get low template-match scores and fall below the peak
    threshold."""
    ellipse_major_px: float = 24.0
    """Template major axis in *detector* pixels. Pupae are ~36 px major
    axis at native resolution; 24 ≈ 36 × work_scale(0.67)."""
    ellipse_minor_px: float = 12.0
    """Template minor axis in detector pixels (~18 × 0.67)."""
    ellipse_n_orientations: int = 8
    """Number of equally-spaced template orientations in ``[0°, 180°)``.
    The template bank takes the pointwise max across orientations."""

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
        response_mode=cfg.response_mode,
        log_sigma=cfg.log_sigma,
        dog_sigma_low=cfg.dog_sigma_low,
        dog_sigma_high=cfg.dog_sigma_high,
        adaptive_small_sigma=cfg.adaptive_small_sigma,
        adaptive_large_sigma=cfg.adaptive_large_sigma,
        adaptive_area_threshold_px=cfg.adaptive_area_threshold_px,
        saturation_floor=cfg.raw_response_saturation_floor,
    )

    paper_mask = None
    if cfg.use_paper_roi:
        paper_mask = detect_paper_roi(
            rgb_work,
            cfg=PaperROIConfig(
                brightness_threshold=cfg.paper_roi_brightness_threshold,
                close_kernel_px=cfg.paper_roi_close_kernel_px,
                erode_margin_px=cfg.paper_roi_erode_margin_px,
                min_paper_fraction=cfg.paper_roi_min_paper_fraction,
                fill_holes=cfg.paper_roi_fill_holes,
            ),
        )
        response = apply_paper_roi_to_response(response, paper_mask)

    allowed_mask = build_allowed_mask(
        response,
        abs_threshold=cfg.allowed_abs_threshold,
        min_percentile=cfg.allowed_min_percentile,
    )

    # v3 — ellipse template cross-correlation. Inserts between the smoothed
    # brown response and the peak detector. Replaces the response that
    # peaks land on with a shape-aware version; leaves allowed_mask
    # untouched so the foreground gate stays where it was.
    template_response: np.ndarray | None = None
    if cfg.use_ellipse_template:
        from .ellipse_template import build_template_bank, template_match_map  # local import keeps base startup fast

        templates = build_template_bank(
            major_px=cfg.ellipse_major_px,
            minor_px=cfg.ellipse_minor_px,
            n_orientations=cfg.ellipse_n_orientations,
        )
        template_response = template_match_map(response, templates)

    if template_response is not None:
        peak_input = template_response
    else:
        peak_input = response

    if cfg.use_resolver_v2:
        from .resolver_v2 import detect_peaks_by_component_v2, ResolverV2Config

        split_cfg = ComponentSplitConfig(
            single_pupa_area_px=cfg.component_single_pupa_area_px,
            min_peak_distance_px=cfg.component_min_peak_distance_px,
            abs_score_threshold=cfg.component_abs_score_threshold,
            min_component_area_px=cfg.component_min_component_area_px,
            area_ratio_threshold=cfg.component_area_ratio_threshold,
            max_peaks_per_component=cfg.component_max_peaks,
        )
        v2_cfg = ResolverV2Config(
            use_pre_erosion=cfg.resolver_v2_pre_erosion,
            pre_erosion_radius_px=cfg.resolver_v2_pre_erosion_radius_px,
            use_h_maxima=cfg.resolver_v2_use_h_maxima,
            h_maxima_h=cfg.resolver_v2_h_maxima_h,
            use_ceil_expected_k=cfg.resolver_v2_use_ceil_expected_k,
            force_multi_peak_above_area=cfg.resolver_v2_force_multi_peak_above_area,
            use_seed_completion=cfg.resolver_v2_use_seed_completion,
            seed_completion_edt_weight=cfg.resolver_v2_seed_completion_edt_weight,
            seed_completion_refine_iters=cfg.resolver_v2_seed_completion_refine_iters,
            seed_completion_min_solidity=cfg.resolver_v2_seed_completion_min_solidity,
            seed_completion_min_extent=cfg.resolver_v2_seed_completion_min_extent,
            seed_completion_min_area_ratio=cfg.resolver_v2_seed_completion_min_area_ratio,
            seed_completion_min_anchor_distance_px=cfg.resolver_v2_seed_completion_min_anchor_distance_px,
            seed_completion_min_score_ratio=cfg.resolver_v2_seed_completion_min_score_ratio,
            seed_completion_max_new_seeds=cfg.resolver_v2_seed_completion_max_new_seeds,
        )
        peaks_work = detect_peaks_by_component_v2(
            peak_input,
            allowed_mask,
            cfg=split_cfg,
            v2_cfg=v2_cfg,
            edge_margin_px=cfg.peak_edge_margin_px,
            native_rgb=rgb_native,
            work_scale=scale_used,
        )
    elif cfg.use_hybrid_watershed:
        from .watershed_resolver import detect_peaks_hybrid_watershed, WatershedConfig
        from .resolver_cv import _component_multi_peak, _component_single_peak  # ComponentSplitConfig already imported at module level

        ws_cfg = WatershedConfig(
            min_component_area_px=cfg.component_min_component_area_px,
            min_edt_seed_value=cfg.watershed_min_edt_seed_value,
            min_edt_seed_distance_px=cfg.watershed_min_edt_seed_distance_px,
            min_basin_area_px=cfg.watershed_min_basin_area_px,
            min_basin_peak_response=cfg.watershed_min_basin_peak_response,
            ellipse_no_overlap_min_distance_px=cfg.watershed_ellipse_no_overlap_min_distance_px,
        )
        split_cfg = ComponentSplitConfig(
            single_pupa_area_px=cfg.component_single_pupa_area_px,
            min_peak_distance_px=cfg.component_min_peak_distance_px,
            abs_score_threshold=cfg.component_abs_score_threshold,
            min_component_area_px=cfg.component_min_component_area_px,
            area_ratio_threshold=cfg.component_area_ratio_threshold,
            max_peaks_per_component=cfg.component_max_peaks,
        )

        def _iter1_resolver(comp_response, comp_mask):
            """Run iter1's component-level peak detection on one component."""
            area = int(comp_mask.sum())
            area_based_k = max(1, int(round(area / max(1.0, split_cfg.single_pupa_area_px))))
            if area < split_cfg.area_ratio_threshold * split_cfg.single_pupa_area_px:
                return _component_single_peak(comp_response, comp_mask, split_cfg.abs_score_threshold)
            return _component_multi_peak(comp_response, comp_mask, area_based_k, split_cfg)

        peaks_work = detect_peaks_hybrid_watershed(
            peak_input,
            allowed_mask,
            cfg=ws_cfg,
            edge_margin_px=cfg.peak_edge_margin_px,
            single_pupa_area_px=cfg.component_single_pupa_area_px,
            iter1_resolver=_iter1_resolver,
        )
    elif cfg.use_watershed_split:
        from .watershed_resolver import detect_peaks_by_watershed, WatershedConfig  # local import to keep base startup fast

        ws_cfg = WatershedConfig(
            min_component_area_px=cfg.component_min_component_area_px,
            min_edt_seed_value=cfg.watershed_min_edt_seed_value,
            min_edt_seed_distance_px=cfg.watershed_min_edt_seed_distance_px,
            min_basin_area_px=cfg.watershed_min_basin_area_px,
            min_basin_peak_response=cfg.watershed_min_basin_peak_response,
            ellipse_no_overlap_min_distance_px=cfg.watershed_ellipse_no_overlap_min_distance_px,
        )
        peaks_work = detect_peaks_by_watershed(
            peak_input,
            allowed_mask,
            cfg=ws_cfg,
            edge_margin_px=cfg.peak_edge_margin_px,
        )
    elif cfg.use_component_split:
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
            peak_input,
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
        peaks_work = detect_peaks(peak_input, allowed_mask, cfg=peak_cfg)

    # Paper-brightness filter: reject detections sitting on dark
    # background (scanner bar, paper edge). Implements the user rule
    # "只识别白纸上的" — only count pupae on white paper.
    # Paper-brightness filter: reject detections sitting on dark
    # background (scanner bar, paper edge). Uses a small 11×11 window
    # to measure local brightness. To avoid killing real pupae in
    # large edge-adjacent components (like scan_10's left-edge cluster),
    # detections in components with area >= 400 work pixels are EXEMPT.
    if cfg.min_background_brightness > 0 and not peaks_work.empty:
        from scipy import ndimage as _ndi_bright
        gray_work = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        half_bw = 5  # 11×11 window
        bh, bw = gray_work.shape
        # Label the allowed mask to find component areas
        _am_gate = allowed_mask > 0
        _am_labels, _am_n = _ndi_bright.label(_am_gate)
        _am_sizes = np.bincount(_am_labels.ravel()) if _am_n > 0 else np.array([0])
        bright_keep = []
        for _, row in peaks_work.iterrows():
            xw = int(round(float(row["x"])))
            yw = int(round(float(row["y"])))
            # Check component area — exempt large components
            # Exempt detections in "round-ish" components (aspect ratio < 10).
            # Scanner strips are extreme (aspect 14-69x); real pupa clusters
            # are typically < 5x. This lets edge-adjacent pupae in wide
            # components survive while still checking elongated strips.
            exempt = False
            if 0 <= yw < _am_labels.shape[0] and 0 <= xw < _am_labels.shape[1]:
                comp_lbl = int(_am_labels[yw, xw])
                if comp_lbl > 0:
                    _cy, _cx = np.where(_am_labels == comp_lbl)
                    _bh = int(_cy.max() - _cy.min() + 1)
                    _bw = int(_cx.max() - _cx.min() + 1)
                    _aspect = max(_bh, _bw) / max(1, min(_bh, _bw))
                    exempt = _aspect < 10.0
            if exempt:
                bright_keep.append(True)
                continue
            y0 = max(0, yw - half_bw)
            y1 = min(bh, yw + half_bw + 1)
            x0 = max(0, xw - half_bw)
            x1 = min(bw, xw + half_bw + 1)
            med_bright = float(np.median(gray_work[y0:y1, x0:x1]))
            bright_keep.append(med_bright >= cfg.min_background_brightness)
        peaks_work = peaks_work[bright_keep].reset_index(drop=True)

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
        if template_response is not None:
            debug["template_response"] = template_response
        if paper_mask is not None:
            debug["paper_mask"] = paper_mask

    return DetectorOutput(instances=instances, debug=debug, runtime_ms=runtime_ms)
