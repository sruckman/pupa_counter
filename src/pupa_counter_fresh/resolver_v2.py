"""v2 component resolver — two structural improvements over iter5.

Targets the dominant failure mode identified from 10 hand-labeled scans:
``peak_local_max`` in the existing ``resolver_cv`` fails to find all peaks in
dense touching clusters because:

1. Adjacent pupae can form connected components with conjoined "dumbbell"
   shapes. The component splitter runs ``peak_local_max`` on the whole
   dumbbell, which finds 1 (or 2) peaks instead of 2 (or 3).
2. ``peak_local_max`` uses strict local-maxima semantics, which fails on
   response regions that are slightly plateau'd between two touching pupae.

This module implements two independent fixes that can be enabled separately:

* **Pre-erosion separator**: before detecting peaks, apply a small binary
  erosion to the allowed mask. This breaks narrow "necks" between touching
  pupae into separate connected components. Each component then gets its
  own ``_component_single_peak`` or ``_component_multi_peak`` pass.

* **H-maxima seeds**: instead of ``peak_local_max``, use
  ``skimage.morphology.h_maxima`` — regional maxima defined by a height
  difference from surrounding minima. This is robust to plateau'd tops.

Both are drop-in replacements for the resolver's per-component peak logic.
The module re-uses the existing ``ComponentSplitConfig`` to stay
compatible with ``detect_peaks_by_component``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import h_maxima

from .resolver_cv import (
    ComponentSplitConfig,
    _component_multi_peak,
    _component_single_peak,
)


@dataclass
class ResolverV2Config:
    """Extends ComponentSplitConfig with v2 knobs."""

    # Pre-erosion separator
    use_pre_erosion: bool = False
    """If True, erode the allowed mask by ``pre_erosion_radius_px`` before
    connected-component labeling. Breaks narrow necks between touching pupae."""

    pre_erosion_radius_px: int = 1
    """Radius of the structuring element used for pre-erosion. ``1`` is a
    3x3 kernel, ``2`` is a 5x5. Too large and single pupae vanish."""

    # H-maxima seeds
    use_h_maxima: bool = False
    """If True, use ``h_maxima`` to find peaks instead of ``peak_local_max``.
    More robust to plateau'd response regions."""

    h_maxima_h: float = 0.03
    """Height difference a bump must exceed to register as an h-maximum.
    Relative to the response map range ``[0, 1]``. Smaller values find
    more bumps (including noise); larger values miss real differences."""

    # Ceiling-based expected_k (fixes the "area=277 rounds to 1" trap)
    use_ceil_expected_k: bool = False
    """If True, compute expected_k via ``ceil(area / single_pupa_area)``
    instead of ``round``. Fixes cases where a component with area
    slightly above single_pupa_area (e.g., 240-300 for single=200)
    rounds to expected_k=1 and loses the second pupa of a touching pair.
    Debug showed scan_3 #14 had area=277, rounded to 1, and missed a
    clear second peak that peak_local_max would have found."""

    force_multi_peak_above_area: float = 0.0
    """If > 0, force multi-peak mode for any component with area >=
    this value, regardless of expected_k or area_ratio_threshold. Set
    to e.g. ``260.0`` to fix the scan_3 #14 case (area=277)."""

    # Seed completion rescue (v5 — underfilled hard-component resolver)
    use_seed_completion: bool = False
    """If True, when ``_component_multi_peak`` returns fewer peaks than
    ``expected_k`` on a component that matches ``_is_hard_component``
    (area/solidity/extent gate), run a second-pass "center completion"
    that places the missing centers via EDT+response weighted farthest-
    point selection, refined by 2 weighted Voronoi centroid updates.

    The happy path is untouched. Rescue only fires on components whose
    maxima pass under-reports relative to area prior. This targets dense
    touching clusters where ``peak_local_max`` finds fewer centers than
    the geometry says there should be."""

    seed_completion_edt_weight: float = 0.35
    """Fraction of the rescue weight map contributed by the normalized
    EDT. Response contributes ``1.0 - edt_weight``. Higher EDT weight
    pulls completion seeds toward geometric centers rather than response
    peaks — useful when response is plateau'd."""

    seed_completion_refine_iters: int = 2
    """Number of weighted-Voronoi centroid refine iterations after
    farthest-point placement. 0 = pure farthest point, 2 = tight
    centroids. Values > 3 rarely matter."""

    seed_completion_min_solidity: float = 0.92
    """Component solidity threshold for the "hard component" gate.
    Below this, the component is considered a merged/dumbbell shape
    and becomes a rescue candidate."""

    seed_completion_min_extent: float = 0.60
    """Component extent threshold (filled_area / bbox_area) for the
    hard gate. Below this, the component is considered irregularly
    shaped — another rescue indicator."""

    seed_completion_min_area_ratio: float = 1.8
    """Component area / single_pupa_area ratio gate. If the component
    is at least this many single-pupa areas, it's a rescue candidate
    regardless of solidity/extent."""

    seed_completion_min_anchor_distance_px: float = 11.0
    """Rescue-specific NMS distance (work pixels) between a newly-placed
    seed and every existing anchor. Much larger than the resolver's
    ``min_peak_distance_px=3`` because the rescue uses weighted
    farthest-point placement (not strict local-maxima semantics), and
    a 3 px separation is small enough that the rescue can place a new
    seed **inside the same pupa** as an anchor. A pupa's minor axis
    is ~12 work px, so two peaks closer than ~9 work px are almost
    certainly on the same pupa. Default 9 work px ≈ 13.5 native px."""

    seed_completion_min_score_ratio: float = 0.6
    """A rescue-added seed must have a response score at least this
    fraction of the **minimum anchor score** to be accepted. Filters
    out seeds placed on plateau regions that snap back to low-response
    pixels which are clearly not pupa centers. Default 0.6 means a
    new seed needs score ≥ 0.6 × min(anchor_scores); raising this
    toward 1.0 is strictly more conservative."""

    seed_completion_max_new_seeds: int = 2
    """Maximum number of NEW seeds (beyond existing anchors) that
    the rescue can add per component. Prevents wild overshooting on
    mega-clusters where ``expected_k`` from the area prior is too
    high and the rescue blindly fills all slots. Default 2 means
    rescue can add at most 2 new centers per component. The existing
    anchors from ``peak_local_max`` are always preserved."""


def _component_h_maxima_peaks(
    comp_response: np.ndarray,
    comp_mask: np.ndarray,
    h: float,
    abs_threshold: float,
    max_peaks: int,
) -> list[tuple[int, int, float]]:
    """Return peaks via h-maxima transform instead of strict local maxima."""
    masked = np.where(comp_mask, comp_response, 0.0).astype(np.float32)
    if masked.max() < abs_threshold:
        return []

    # h-maxima returns a binary mask of regional maxima that are at least
    # ``h`` higher than any connected lower region
    regional_max = h_maxima(masked, float(h))
    if not regional_max.any():
        # Fall back to a single max if h-maxima finds nothing
        idx = int(np.argmax(masked))
        y, x = np.unravel_index(idx, masked.shape)
        score = float(masked[y, x])
        if score < abs_threshold:
            return []
        return [(int(y), int(x), score)]

    # Each connected region of regional_max is one "bump" — find its centroid
    labels, n = ndi.label(regional_max)
    results: list[tuple[int, int, float]] = []
    for label in range(1, n + 1):
        region_mask = labels == label
        # Centroid position
        ys, xs = np.where(region_mask)
        if len(ys) == 0:
            continue
        cy = int(np.round(ys.mean()))
        cx = int(np.round(xs.mean()))
        # Score: max response in the h-max region
        score = float(comp_response[ys, xs].max())
        if score < abs_threshold:
            continue
        results.append((cy, cx, score))

    # Sort by score descending and cap
    results.sort(key=lambda t: -t[2])
    return results[: max(1, int(max_peaks))]


def _shape_features(
    comp_mask: np.ndarray,
    area: int,
    single_pupa_area: float,
) -> dict[str, float]:
    """Cheap whole-shape features for single-vs-multi decisions."""
    single = max(1.0, float(single_pupa_area))
    area_ratio = float(area) / single
    rp_list = regionprops(comp_mask.astype(np.uint8))
    defaults = {"area_ratio": area_ratio, "major_ratio": 1.0, "minor_ratio": 1.0,
                "span_ratio": 1.0, "circularity": 1.0, "solidity": 1.0, "defect_count": 0.0}
    if not rp_list:
        return defaults
    rp = rp_list[0]
    major = float(getattr(rp, "axis_major_length", getattr(rp, "major_axis_length", 0.0)) or 0.0)
    minor = max(1.0, float(getattr(rp, "axis_minor_length", getattr(rp, "minor_axis_length", 0.0)) or 0.0))
    solidity = float(rp.solidity) if rp.solidity is not None else 1.0
    single_minor = float(np.sqrt(2.0 * single / np.pi))
    single_major = 2.0 * single_minor
    major_ratio = major / max(single_major, 1.0)
    minor_ratio = minor / max(single_minor, 1.0)
    span_ratio = max(major_ratio, minor_ratio)

    smooth = ndi.binary_closing(comp_mask, structure=np.ones((3, 3), dtype=bool)).astype(np.uint8)
    cnts, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = 0.0
    defect_count = 0
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, True))
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 4 and len(cnt) >= 5:
            try:
                defects = cv2.convexityDefects(cnt, hull_idx)
                if defects is not None:
                    depth_thr = max(1.5, 0.18 * single_minor)
                    chord_thr = max(3.0, 0.80 * single_minor)
                    for s, e, _f, d in defects[:, 0]:
                        depth = float(d) / 256.0
                        p0 = cnt[int(s), 0].astype(np.float32)
                        p1 = cnt[int(e), 0].astype(np.float32)
                        chord = float(np.hypot(float(p0[0]-p1[0]), float(p0[1]-p1[1])))
                        if depth >= depth_thr and chord >= chord_thr:
                            defect_count += 1
            except cv2.error:
                pass
    circularity = float(4.0 * np.pi * float(area) / (perimeter * perimeter)) if perimeter > 1e-6 else 1.0
    return {"area_ratio": area_ratio, "major_ratio": major_ratio, "minor_ratio": minor_ratio,
            "span_ratio": span_ratio, "circularity": circularity, "solidity": solidity,
            "defect_count": float(defect_count)}


def _is_multi_pupa(comp_mask: np.ndarray, area: int, single_pupa_area: float) -> bool:
    """Binary gate: does this blob clearly look like >1 pupa?"""
    f = _shape_features(comp_mask, area, single_pupa_area)
    if (f["area_ratio"] <= 1.15 and f["span_ratio"] <= 1.14
            and f["circularity"] >= 0.77 and f["solidity"] >= 0.94
            and f["defect_count"] == 0):
        return False
    score = 0
    if f["area_ratio"] >= 1.55: score += 2
    elif f["area_ratio"] >= 1.25: score += 1
    if f["span_ratio"] >= 1.34: score += 2
    elif f["span_ratio"] >= 1.22: score += 1
    if f["circularity"] <= 0.70: score += 2
    elif f["circularity"] <= 0.76: score += 1
    if f["solidity"] <= 0.90: score += 1
    elif f["solidity"] <= 0.94 and f["area_ratio"] >= 1.30: score += 1
    if f["defect_count"] >= 2: score += 1
    return score >= 4


def _should_reduce_area_k(comp_mask: np.ndarray, area: int, single_pupa_area: float, base_k: int) -> bool:
    """Conservative veto: area vote is probably one too high."""
    if base_k <= 1:
        return False
    f = _shape_features(comp_mask, area, single_pupa_area)
    if base_k == 2:
        return (not _is_multi_pupa(comp_mask, area, single_pupa_area)
                and f["area_ratio"] < 1.55 and f["span_ratio"] < 1.18
                and f["circularity"] > 0.77 and f["solidity"] > 0.95)
    return (f["span_ratio"] < (float(base_k) - 1.15)
            and f["circularity"] > 0.60 and f["solidity"] > 0.82)


def _snap_peak_to_body(comp_mask, comp_response, y, x, *, abs_threshold, search_radius=4, edt_weight=0.20):
    """Snap a peak to the nearest pixel with real body support."""
    edt = ndi.distance_transform_edt(comp_mask).astype(np.float32)
    h, w = comp_mask.shape
    y0 = max(0, int(y)-search_radius); y1 = min(h, int(y)+search_radius+1)
    x0 = max(0, int(x)-search_radius); x1 = min(w, int(x)+search_radius+1)
    local_mask = comp_mask[y0:y1, x0:x1]
    if not local_mask.any():
        return None
    local_resp = comp_response[y0:y1, x0:x1].astype(np.float32)
    local_edt = edt[y0:y1, x0:x1]
    edt_norm = local_edt / max(float(local_edt.max()), 1e-6)
    score_map = local_resp + float(edt_weight) * edt_norm
    score_map[~local_mask] = -1.0
    iy, ix = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
    sy, sx = y0+int(iy), x0+int(ix)
    s = float(comp_response[sy, sx])
    return (sy, sx, s) if s >= abs_threshold else None


def _has_body_support(comp_mask, comp_response, y, x, single_pupa_area):
    """Reject centers on flimsy support."""
    edt = ndi.distance_transform_edt(comp_mask).astype(np.float32)
    thickness = float(edt[int(y), int(x)])
    single = max(1.0, float(single_pupa_area))
    min_thickness = max(2.0, 0.14 * np.sqrt(single))
    peak_score = float(comp_response[int(y), int(x)])
    core_thr = max(0.18, 0.72 * peak_score)
    core = (comp_response >= core_thr) & comp_mask
    labels, _ = ndi.label(core)
    lab = int(labels[int(y), int(x)])
    local_core_area = 0 if lab == 0 else int((labels == lab).sum())
    return thickness >= min_thickness or local_core_area >= max(8, int(round(0.05 * single)))


def _count_dt_lobes(
    comp_mask: np.ndarray,
    single_pupa_area: float,
) -> int:
    """Count geometric lobes from the smoothed distance transform."""
    if not comp_mask.any():
        return 1
    single = max(1.0, float(single_pupa_area))
    edt = ndi.distance_transform_edt(comp_mask).astype(np.float32)
    dt_floor = max(2.2, 0.16 * np.sqrt(single))
    if float(edt.max()) < dt_floor:
        return 1
    # Lighter smoothing preserves touching-pair lobes at 0.67x
    edt_s = cv2.GaussianBlur(edt, (0, 0), 0.6)
    min_dist = max(3, int(round(0.16 * np.sqrt(single))))
    coords = peak_local_max(
        edt_s, min_distance=min_dist, threshold_abs=dt_floor, exclude_border=False,
    )
    count = sum(1 for y, x in coords if comp_mask[int(y), int(x)])
    return max(1, count)


def _count_response_cores(
    comp_mask: np.ndarray,
    comp_response: np.ndarray,
    single_pupa_area: float,
) -> int:
    """Count high-confidence response islands inside one permissive blob."""
    vals = comp_response[comp_mask]
    if vals.size == 0:
        return 1
    local_max = float(vals.max())
    if local_max <= 1e-6:
        return 1
    single = max(1.0, float(single_pupa_area))
    # Lower threshold preserves touching-pair core separation
    thr = max(0.18, 0.55 * local_max)
    core = (comp_response >= thr) & comp_mask
    if not core.any():
        return 1
    opened = ndi.binary_opening(core, structure=np.ones((3, 3), dtype=bool))
    if opened.any():
        core = opened
    labels, n = ndi.label(core)
    if n == 0:
        return 1
    sizes = np.bincount(labels.ravel())
    min_core_area = max(6, int(round(0.04 * single)))
    return max(1, int((sizes[1:] >= min_core_area).sum()))


def _estimate_expected_k_shape_aware(
    comp_mask: np.ndarray,
    comp_response: np.ndarray,
    single_pupa_area: float,
) -> int:
    """Shape-aware expected_k: area gives candidate range, geometry decides."""
    area = int(comp_mask.sum())
    if area <= 0:
        return 1
    single = max(1.0, float(single_pupa_area))
    area_ratio = area / single

    if area_ratio < 0.95:
        return 1

    rp_list = regionprops(comp_mask.astype(np.uint8))
    if not rp_list:
        return max(1, int(round(area_ratio)))
    rp = rp_list[0]
    solidity = float(rp.solidity) if rp.solidity is not None else 1.0
    extent = float(rp.extent) if rp.extent is not None else 1.0
    major = float(getattr(rp, "axis_major_length", getattr(rp, "major_axis_length", 0.0)) or 0.0)
    minor = float(getattr(rp, "axis_minor_length", getattr(rp, "minor_axis_length", 0.0)) or 0.0)
    minor = max(1.0, minor)
    aspect = major / minor
    ellipse_area = np.pi * 0.25 * major * minor
    ellipse_fill = area / max(ellipse_area, 1.0)

    # Fast path for compact singles
    if (area_ratio < 1.25 and solidity > 0.95 and extent > 0.62
            and ellipse_fill > 0.88 and aspect < 2.40):
        return 1

    k_area = max(1, int(round(area_ratio)))
    k_dt = _count_dt_lobes(comp_mask, single)
    k_core = _count_response_cores(comp_mask, comp_response, single)
    single_major = max(1.0, 1.60 * np.sqrt(single))
    k_major = max(1, int(round(major / single_major)))

    non_single = (
        solidity < 0.94 or extent < 0.64 or ellipse_fill < 0.78
        or (aspect > 2.35 and area_ratio > 1.20)
    )

    k_min = max(1, int(np.floor(area_ratio)) - 1)
    k_max = max(k_min, int(np.ceil(area_ratio)) + 1)

    best_k, best_score = 1, float("inf")
    for k in range(k_min, k_max + 1):
        score = 1.80 * abs(area_ratio - k)
        score += 0.90 * abs(k - k_dt)
        score += 0.65 * abs(k - k_core)
        score += 0.35 * abs(k - k_major)
        if k == 1 and non_single:
            score += 0.95
        if k == k_area:
            score -= 0.10
        if score < best_score:
            best_score = score
            best_k = k
    return max(1, int(best_k))


def _is_border_thin_artifact(
    comp_mask: np.ndarray,
    comp_response: np.ndarray,
    peak_y: int,
    peak_x: int,
    single_pupa_area: float,
    *,
    touches_image_border: bool,
) -> bool:
    """Reject thin ridge-like peaks on border-connected artifacts."""
    if not touches_image_border:
        return False
    single = max(1.0, float(single_pupa_area))
    edt = ndi.distance_transform_edt(comp_mask).astype(np.float32)
    peak_thickness = float(edt[int(peak_y), int(peak_x)])
    min_thickness = max(2.3, 0.16 * np.sqrt(single))
    peak_score = float(comp_response[int(peak_y), int(peak_x)])
    core_thr = max(0.18, 0.75 * peak_score)
    core = (comp_response >= core_thr) & comp_mask
    labels, _ = ndi.label(core)
    lab = int(labels[int(peak_y), int(peak_x)])
    if lab == 0:
        return peak_thickness < min_thickness
    local_core = labels == lab
    local_area = int(local_core.sum())
    rp_list = regionprops(local_core.astype(np.uint8))
    local_solidity = float(rp_list[0].solidity) if rp_list and rp_list[0].solidity is not None else 1.0
    return peak_thickness < min_thickness and (local_area < 0.18 * single or local_solidity < 0.72)


def _is_hard_component(
    comp_mask: np.ndarray,
    area: int,
    single_pupa_area_px: float,
    *,
    min_solidity: float,
    min_extent: float,
    min_area_ratio: float,
) -> bool:
    """Gate for the seed completion rescue.

    A component is "hard" if it's either large relative to a single
    pupa or visibly irregular (low solidity / low extent). Both cues
    fire for merged dumbbell / peanut / mega-cluster shapes that need
    center completion but don't have enough local maxima to produce
    them via ``peak_local_max``.
    """
    if area <= 0:
        return False
    if area >= float(min_area_ratio) * float(single_pupa_area_px):
        return True
    rp_list = regionprops(comp_mask.astype(np.uint8))
    if not rp_list:
        return False
    rp = rp_list[0]
    solidity = float(rp.solidity) if rp.solidity is not None else 1.0
    extent = float(rp.extent) if rp.extent is not None else 1.0
    return solidity < float(min_solidity) or extent < float(min_extent)


def _complete_missing_peaks(
    comp_response: np.ndarray,
    comp_mask: np.ndarray,
    anchors: list[tuple[int, int, float]],
    expected_k: int,
    *,
    abs_threshold: float,
    min_peak_distance_px: int,
    edt_weight: float,
    refine_iters: int,
    min_anchor_distance_px: float = 9.0,
    min_score_ratio: float = 0.6,
    max_new_seeds: int = 2,
) -> list[tuple[int, int, float]]:
    """Place k centers inside a merged component instead of searching maxima.

    When ``_component_multi_peak`` reports fewer peaks than ``expected_k``
    we interpret that as "the response map doesn't have enough strict
    local maxima for this blob, but geometry says it should contain k
    pupae". Rather than relax the maxima criterion (which added FPs in
    every experiment to date), we actively complete the missing centers:

    1. Build a combined weight map on the component: a blend of the
       normalized response (brightness of pupa-like pixels) and the
       normalized Euclidean distance transform (distance from background).
       The EDT captures geometric "center-ness" on plateau regions where
       the response map is flat.

    2. Keep the existing anchors as fixed seeds. Greedily place each
       missing seed at the pixel that maximizes
       ``weight * distance_to_nearest_existing_seed²``. This is a
       farthest-point rule weighted by the combined feature map.

    3. Refine all seeds (including the anchors) with ``refine_iters``
       weighted-Voronoi centroid updates. Each iteration assigns every
       pixel to its nearest seed, then moves each seed to the weighted
       centroid of its cell. Anchors drift slightly if the response
       plateau shifts the weighted centroid.

    4. Snap each center back to the nearest foreground pixel whose
       response clears ``abs_threshold``, enforcing ``min_peak_distance_px``
       NMS against accepted points. Return the accepted list (score
       descending, capped at expected_k).
    """
    if expected_k <= 0:
        return list(anchors)
    ys, xs = np.where(comp_mask)
    if len(ys) == 0:
        return list(anchors)

    edt = ndi.distance_transform_edt(comp_mask).astype(np.float32)
    resp = np.where(comp_mask, comp_response, 0.0).astype(np.float32)

    def _norm(arr: np.ndarray) -> np.ndarray:
        peak = float(arr.max())
        if peak <= 1e-6:
            return arr
        return arr / peak

    edt_weight_f = float(edt_weight)
    resp_weight = 1.0 - edt_weight_f
    weight = resp_weight * _norm(resp) + edt_weight_f * _norm(edt)

    pts = np.column_stack([ys.astype(np.float32), xs.astype(np.float32)])
    w = weight[ys, xs].astype(np.float32)

    n_anchors = len(anchors)
    if anchors:
        centers = np.array([(float(y), float(x)) for y, x, _ in anchors], dtype=np.float32)
    else:
        i0 = int(np.argmax(w))
        centers = pts[i0 : i0 + 1].copy()

    # Farthest-point completion (capped by max_new_seeds)
    max_total = n_anchors + int(max_new_seeds)
    target_k = min(expected_k, max_total)
    while len(centers) < target_k:
        diff = pts[:, None, :] - centers[None, :, :]
        d2 = (diff * diff).sum(axis=2).min(axis=1)
        best = int(np.argmax(w * d2))
        centers = np.vstack([centers, pts[best : best + 1]])

    # Weighted-Voronoi refine — only refine NEW seeds (indices ≥ n_anchors).
    # Anchors stay pinned at their original positions so that the rescue
    # never displaces a validated peak_local_max result.
    for _ in range(int(refine_iters)):
        diff = pts[:, None, :] - centers[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        owner = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for k in range(n_anchors, len(centers)):  # skip anchors
            keep_pts = owner == k
            if not np.any(keep_pts):
                continue
            ww = w[keep_pts] + 1e-6
            total = float(ww.sum())
            new_centers[k, 0] = float((pts[keep_pts, 0] * ww).sum() / total)
            new_centers[k, 1] = float((pts[keep_pts, 1] * ww).sum() / total)
        centers = new_centers

    # Snap to valid foreground pixels. Anchors and new seeds are treated
    # asymmetrically:
    #
    # * Anchors (the first ``n_anchors`` rows of ``centers``) are always
    #   accepted and snapped back with just the standard NMS.
    # * Newly placed seeds must additionally clear:
    #     (a) ``min_anchor_distance_px`` from EVERY anchor, and
    #     (b) ``min_score_ratio * min(anchor_scores)`` at their pixel.
    #
    # (a) stops rescue from placing a seed inside the same pupa as an
    #     anchor — the default ``min_peak_distance_px=3`` is too tight
    #     because it's sized for strict local-maxima semantics.
    # (b) stops rescue from accepting a weak plateau seed when all the
    #     real pupae in this component have high response.
    min_d2 = float(min_peak_distance_px) ** 2
    min_anchor_d2 = float(min_anchor_distance_px) ** 2
    anchor_scores = [float(s) for _, _, s in anchors]
    min_anchor_score = min(anchor_scores) if anchor_scores else 0.0
    score_floor = max(abs_threshold, float(min_score_ratio) * min_anchor_score)

    out: list[tuple[int, int, float]] = []
    accepted_anchor_xy: list[tuple[int, int]] = []

    for center_idx, (cy, cx) in enumerate(centers):
        is_anchor = center_idx < n_anchors
        d2 = (pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2
        order = np.argsort(d2)
        for idx in order[:20]:
            y = int(pts[idx, 0])
            x = int(pts[idx, 1])
            score = float(comp_response[y, x])
            if score < abs_threshold:
                continue
            # Standard NMS: no two accepted points closer than min_peak_distance
            if any((y - py) ** 2 + (x - px) ** 2 < min_d2 for py, px, _ in out):
                continue
            if not is_anchor:
                # Rescue-added seed must clear the stricter distance
                # from EVERY already-accepted point (anchor or other
                # new seed), not just the standard 3 px NMS.
                if any(
                    (y - py) ** 2 + (x - px) ** 2 < min_anchor_d2
                    for py, px, _ in out
                ):
                    continue
                # Rescue-added seed must not be a weak plateau point
                if score < score_floor:
                    continue
            out.append((y, x, score))
            if is_anchor:
                accepted_anchor_xy.append((y, x))
            break

    out.sort(key=lambda t: -t[2])
    return out[: int(target_k)]


def detect_peaks_by_component_v2(
    response: np.ndarray,
    allowed_mask: np.ndarray,
    *,
    cfg: ComponentSplitConfig | None = None,
    v2_cfg: ResolverV2Config | None = None,
    edge_margin_px: int = 4,
    native_rgb: np.ndarray | None = None,
    work_scale: float = 0.67,
) -> pd.DataFrame:
    """Drop-in replacement for ``detect_peaks_by_component`` with v2 tricks.

    If both ``use_pre_erosion`` and ``use_h_maxima`` are False, this is
    bit-compatible with the original function (same output).
    """
    cfg = cfg or ComponentSplitConfig()
    v2_cfg = v2_cfg or ResolverV2Config()
    _native_rgb = native_rgb
    _work_scale = work_scale

    gate = allowed_mask > 0
    if not gate.any():
        return pd.DataFrame(columns=["x", "y", "score", "resolver_type"])

    # Optional pre-erosion — break narrow necks
    if v2_cfg.use_pre_erosion and v2_cfg.pre_erosion_radius_px > 0:
        r = int(v2_cfg.pre_erosion_radius_px)
        kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=np.uint8)
        gate = cv2.erode(gate.astype(np.uint8), kernel).astype(bool)
        if not gate.any():
            # Erosion wiped everything — fall back to original mask
            gate = allowed_mask > 0

    labels, n_components = ndi.label(gate)
    if n_components == 0:
        return pd.DataFrame(columns=["x", "y", "score", "resolver_type"])

    component_sizes = np.bincount(labels.ravel())
    slices = ndi.find_objects(labels)

    xs: list[int] = []
    ys: list[int] = []
    scores: list[float] = []
    resolver_types: list[str] = []

    h, w = response.shape
    for comp_idx in range(1, n_components + 1):
        area = int(component_sizes[comp_idx])
        if area < cfg.min_component_area_px:
            continue

        comp_slice = slices[comp_idx - 1]
        if comp_slice is None:
            continue

        sub_labels = labels[comp_slice]
        comp_mask = sub_labels == comp_idx
        comp_response = response[comp_slice]

        if v2_cfg.use_ceil_expected_k:
            area_based_k = max(1, int(np.ceil(area / max(1.0, cfg.single_pupa_area_px))))
        else:
            area_based_k = max(1, int(round(area / max(1.0, cfg.single_pupa_area_px))))
        expected_k = min(int(cfg.max_peaks_per_component), area_based_k)

        force_multi = (
            v2_cfg.force_multi_peak_above_area > 0
            and area >= v2_cfg.force_multi_peak_above_area
        )
        if force_multi and expected_k == 1:
            # Solidity gate: only bump if component doesn't look like
            # a compact single pupa. Calibrated from 70 hand-labeled
            # components: all false bumps had sol >= 0.92.
            _fm_rp = regionprops(comp_mask.astype(np.uint8))
            _fm_sol = float(_fm_rp[0].solidity) if _fm_rp else 1.0
            if _fm_sol < 0.92:
                expected_k = 2
            else:
                force_multi = False

        # Native-resolution EDT gate: for dead-zone components where area
        # rounds to k=1, upscale the component mask to native resolution,
        # recompute response + EDT, and check if the EDT has 2+ lobes.
        # At native resolution (1.5x work), EDT valleys between touching
        # pupae are deep enough to resolve, while at 0.67x they merge.
        # Overhead: ~0.3ms per component, ~4ms per scan.
        if (
            v2_cfg.use_seed_completion
            and expected_k == 1
            and area >= 1.2 * cfg.single_pupa_area_px
            and area <= 1.7 * cfg.single_pupa_area_px
            and _native_rgb is not None
        ):
            y0c, x0c = comp_slice[0].start, comp_slice[1].start
            y1c, x1c = comp_slice[0].stop, comp_slice[1].stop
            inv = 1.0 / _work_scale
            pad_n = 12
            y0n = max(0, int((y0c - 5) * inv))
            y1n = min(_native_rgb.shape[0], int((y1c + 5) * inv))
            x0n = max(0, int((x0c - 5) * inv))
            x1n = min(_native_rgb.shape[1], int((x1c + 5) * inv))
            _crop = _native_rgb[y0n:y1n, x0n:x1n]
            if _crop.size > 0:
                from .preprocess import build_blue_mask as _bbm
                from .response import compute_response_map as _crm, build_allowed_mask as _bam
                _bl = _bbm(_crop)
                _rn = _crm(_crop, _bl, smooth_sigma=1.2)
                _mn = _bam(_rn, abs_threshold=0.12, min_percentile=0.0)
                _gn = _mn > 0
                _ln, _nn = ndi.label(_gn)
                if _nn > 0:
                    _ns = np.bincount(_ln.ravel())
                    _ml = int(np.argmax(_ns[1:])) + 1
                    _cm = _ln == _ml
                    _edt = ndi.distance_transform_edt(_cm).astype(np.float32)
                    _edt_s = cv2.GaussianBlur(_edt, (0, 0), 1.5)
                    _peaks = peak_local_max(
                        _edt_s, min_distance=7, threshold_abs=3.0, exclude_border=False
                    )
                    _valid = sum(1 for _py, _px in _peaks if _cm[int(_py), int(_px)])
                    # Also check solidity: only bump if the component
                    # doesn't look like a compact single pupa
                    _rp_sol = regionprops(comp_mask.astype(np.uint8))
                    _sol = float(_rp_sol[0].solidity) if _rp_sol else 1.0
                    if _valid >= 2 and _sol < 0.90:
                        expected_k = 2
                        force_multi = True

        # Determine if this component touches the image border (for artifact veto)
        touches_image_border = (
            comp_slice[0].start == 0
            or comp_slice[1].start == 0
            or comp_slice[0].stop == h
            or comp_slice[1].stop == w
        )

        # Pick which peak-finding method to use
        if v2_cfg.use_h_maxima:
            peaks = _component_h_maxima_peaks(
                comp_response,
                comp_mask,
                v2_cfg.h_maxima_h,
                cfg.abs_score_threshold,
                cfg.max_peaks_per_component,
            )
            # If h-maxima found fewer than expected_k, that's fine — trust it
            resolver_tag = "h_maxima"
        elif force_multi:
            # Force multi-peak even when expected_k==1 would normally apply
            peaks = _component_multi_peak(comp_response, comp_mask, expected_k, cfg)
            resolver_tag = "force_multi"
        elif expected_k == 1 or area < cfg.area_ratio_threshold * cfg.single_pupa_area_px:
            peaks = _component_single_peak(
                comp_response, comp_mask, cfg.abs_score_threshold
            )
            resolver_tag = "singleton_peak"
        else:
            peaks = _component_multi_peak(comp_response, comp_mask, expected_k, cfg)
            resolver_tag = "component_split"

        # Seed completion rescue — underfilled hard-component branch.
        # Fires only when:
        #   1. ``use_seed_completion`` is on
        #   2. We were in a multi-peak branch (expected_k > 1)
        #   3. The resolver returned fewer peaks than expected
        #   4. The component passes the hard gate (area / solidity / extent)
        if (
            v2_cfg.use_seed_completion
            and expected_k > 1
            and len(peaks) < expected_k
            and _is_hard_component(
                comp_mask,
                area,
                cfg.single_pupa_area_px,
                min_solidity=v2_cfg.seed_completion_min_solidity,
                min_extent=v2_cfg.seed_completion_min_extent,
                min_area_ratio=v2_cfg.seed_completion_min_area_ratio,
            )
        ):
            rescued = _complete_missing_peaks(
                comp_response,
                comp_mask,
                anchors=peaks,
                expected_k=expected_k,
                abs_threshold=cfg.abs_score_threshold,
                min_peak_distance_px=cfg.min_peak_distance_px,
                edt_weight=v2_cfg.seed_completion_edt_weight,
                refine_iters=v2_cfg.seed_completion_refine_iters,
                min_anchor_distance_px=v2_cfg.seed_completion_min_anchor_distance_px,
                min_score_ratio=v2_cfg.seed_completion_min_score_ratio,
                max_new_seeds=v2_cfg.seed_completion_max_new_seeds,
            )
            if len(rescued) > len(peaks):
                peaks = rescued
                resolver_tag = "seed_completion"

        y0, x0 = comp_slice[0].start, comp_slice[1].start
        for y_local, x_local, score in peaks:
            x_global = x0 + x_local
            y_global = y0 + y_local
            if edge_margin_px > 0 and (
                x_global < edge_margin_px
                or y_global < edge_margin_px
                or x_global >= w - edge_margin_px
                or y_global >= h - edge_margin_px
            ):
                continue
            # Border thin-artifact veto
            if v2_cfg.use_seed_completion and _is_border_thin_artifact(
                comp_mask, comp_response, y_local, x_local,
                cfg.single_pupa_area_px, touches_image_border=touches_image_border,
            ):
                continue
            # Center sanitization disabled — _has_body_support is too
            # aggressive at 0.67x, killing real pupae at cluster edges.
            xs.append(int(x_global))
            ys.append(int(y_global))
            scores.append(float(score))
            resolver_types.append(resolver_tag)

    # Border-artifact veto: detections sitting in small components that
    # touch the image border are almost always scanner-bar or edge-dust
    # artifacts. Real edge-adjacent pupae live in larger components
    # (area >= 150). This replaces the old strip-geometry veto with a
    # simpler and more effective criterion.
    if edge_margin_px > 0 and xs:
        border_veto_max_area = 150
        border_labels: set[int] = set()
        for comp_idx2 in range(1, n_components + 1):
            sl2 = slices[comp_idx2 - 1]
            if sl2 is None:
                continue
            area2 = int(component_sizes[comp_idx2])
            if area2 >= border_veto_max_area:
                continue
            y_min, y_max = sl2[0].start, sl2[0].stop
            x_min, x_max = sl2[1].start, sl2[1].stop
            if y_min <= 1 or y_max >= h - 2 or x_min <= 1 or x_max >= w - 2:
                border_labels.add(comp_idx2)
        if border_labels:
            bkeep = []
            for i in range(len(xs)):
                xw, yw = xs[i], ys[i]
                if 0 <= yw < labels.shape[0] and 0 <= xw < labels.shape[1]:
                    bkeep.append(int(labels[yw, xw]) not in border_labels)
                else:
                    bkeep.append(True)
            if not all(bkeep):
                xs = [xs[i] for i in range(len(xs)) if bkeep[i]]
                ys = [ys[i] for i in range(len(ys)) if bkeep[i]]
                scores = [scores[i] for i in range(len(scores)) if bkeep[i]]
                resolver_types = [resolver_types[i] for i in range(len(resolver_types)) if bkeep[i]]

    # Global NMS: when a rescue peak is too close to ANY other peak,
    # always kill the RESCUE peak (never the original v4 peak). This
    # prevents rescue from displacing validated peak_local_max results.
    # If both are rescue peaks, kill the weaker one.
    if v2_cfg.use_seed_completion and xs:
        global_nms_d2 = float(v2_cfg.seed_completion_min_anchor_distance_px) ** 2
        rescue_tags = {"seed_completion", "edge_recovery"}
        n_pts = len(xs)
        keep = [True] * n_pts
        for i in range(n_pts):
            if not keep[i]:
                continue
            if resolver_types[i] not in rescue_tags:
                continue
            # This is a rescue peak — check if it's too close to
            # any other KEPT peak (rescue or not).
            for j in range(n_pts):
                if j == i or not keep[j]:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                if dx * dx + dy * dy < global_nms_d2:
                    # Kill this rescue peak, keep the other
                    keep[i] = False
                    break
        xs = [xs[i] for i in range(n_pts) if keep[i]]
        ys = [ys[i] for i in range(n_pts) if keep[i]]
        scores = [scores[i] for i in range(n_pts) if keep[i]]
        resolver_types = [resolver_types[i] for i in range(n_pts) if keep[i]]

    return pd.DataFrame(
        {"x": xs, "y": ys, "score": scores, "resolver_type": resolver_types}
    )
