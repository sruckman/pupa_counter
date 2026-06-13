"""Pure-CV local resolver for the fresh peak-first detector.

v0 had one global ``peak_local_max`` call with ``min_distance ≈ 10``, which
silently dropped ~20% of teacher instances whenever several pupae touched
inside the same allowed-mask component. The visual audit confirmed the
failure mode: dense clusters of 2–4 touching pupae collapse into a single v0
peak.

v1 fixes that by running ``peak_local_max`` **per connected component**,
letting the number of peaks per component be governed by the component's
area divided by a single-pupa area prior. The inner ``min_distance`` is
allowed to be smaller than the global v0 value because we already know all
peaks in a given call share a blob — there is no risk of a single pupa
splitting into two peaks unless the response map has two genuinely separated
maxima inside it.

This stays pure CV. No watershed basin labels, no learned resolver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


@dataclass
class ComponentSplitConfig:
    single_pupa_area_px: float = 230.0
    """Nominal area (in *work-resolution* pixels) of one pupa inside the
    allowed mask. At 0.67x a teacher pupa is ~24x12 px (ellipse area ~226),
    and we measure allowed-mask footprints not ellipse areas so 230 matches
    the observed footprint on flat backgrounds."""

    min_peak_distance_px: int = 5
    """Per-component NMS radius. Set well below the v0 global ``10`` because
    the caller has already restricted attention to a single blob, so very
    close maxima can legitimately correspond to touching pupae."""

    abs_score_threshold: float = 0.20
    """Absolute response floor a split peak must clear. Slightly lower than
    the global ``0.22`` gate because split peaks sit off-center and can be
    dimmer than isolated peaks."""

    min_component_area_px: int = 60
    """Skip components smaller than this — they are almost certainly noise
    or scan dust, not pupae."""

    area_ratio_threshold: float = 1.35
    """A component needs to be at least this many times the single-pupa
    area before we consider it for splitting. Below that it is treated as
    a single pupa and the brightest local maximum wins."""

    max_peaks_per_component: int = 20
    """Cap on split peaks per blob, to guard against runaway seeds on very
    large merged components. Values above ``20`` almost always indicate
    that ``single_pupa_area_px`` is mis-calibrated for this scan."""

    # v2 — distance transform assisted splitting --------------------------
    use_distance_transform: bool = False
    """If True, augment the per-component peak search with a distance
    transform pass. Each touching pupa in a dense blob shows up as a
    separate EDT maximum regardless of the area-prior expected_k, which
    lets v2 recover the "tucked next to a match" misses v1 leaves behind.

    The detection signal inside a blob becomes

        signal = response + dt_weight * (edt / edt_max)

    so bright isolated pupae stay easy to find while touching pairs get a
    geometric boost."""

    dt_weight: float = 0.60
    dt_min_distance_px: int = 4
    dt_min_edt_px: float = 3.0
    dt_expected_k_slack: float = 1.25
    """Slack factor on the area-prior ``expected_k``. v1 used exactly
    ``round(area/single_pupa_area)``; v2 allows up to ``1.25 * expected_k``
    total peaks per component (hard cap)."""

    dedup_edt_min_distance_px: int = 6
    """Radius used to dedupe EDT seeds against response seeds before the
    global NMS. Larger than ``min_peak_distance_px`` so EDT seeds only
    survive when they cover genuinely new geometry."""

    # v2 — erosion-based core counting --------------------------------
    use_erosion_core_count: bool = False
    """When True, derive ``expected_k`` from the number of sub-cores that
    survive a morphological erosion of each component's allowed mask,
    instead of from the area prior alone. Thin-neck bridges between
    touching pupae collapse under erosion, so a k-pupa blob ideally breaks
    into k separate cores — a much sharper count than
    ``round(area / single_pupa_area)`` on merged clusters."""

    erosion_radius_px: int = 2
    """Radius (in work-resolution pixels) of the structuring element used
    to erode each component. Too small and narrow bridges survive; too
    large and single pupae vanish. ``2`` at 0.67x ≈ 3 native px, which is
    narrower than a healthy pupa's minor axis (~12 native px at 0.67x ≈
    18 native px natively)."""

    erosion_min_core_area_px: int = 20
    """Any eroded sub-component smaller than this is treated as noise and
    excluded from the core count."""

    erosion_k_margin: int = 0
    """Added to the erosion-derived core count before it is used as
    ``expected_k``. Positive values bias the detector toward recall."""

    # v2 — response core mask -----------------------------------------
    use_response_core_mask: bool = False
    """When True, count pupae inside each permissive component by taking
    a **stricter** response threshold (the "core mask") and labeling its
    sub-components. This separates touching pupae at the pixel level
    because the response drops in the valley between two pupae before it
    drops in the allowed mask. The derived core count then supplies
    ``expected_k`` independently of the area prior."""

    response_core_threshold: float = 0.26
    """Absolute response threshold for the core mask. Must be higher than
    ``abs_score_threshold`` and well above ``allowed_abs_threshold``."""

    response_core_min_area_px: int = 15
    """Minimum core sub-component area in work pixels."""

    response_core_k_margin: int = 0
    """Added to the core count before it is used as ``expected_k``."""


def _component_single_peak(
    comp_response: np.ndarray,
    comp_mask: np.ndarray,
    abs_threshold: float,
) -> list[tuple[int, int, float]]:
    """Return the single brightest ``(y, x, score)`` in a component, or []."""
    masked = np.where(comp_mask, comp_response, -1.0)
    idx = int(np.argmax(masked))
    y, x = np.unravel_index(idx, masked.shape)
    score = float(masked[y, x])
    if score < abs_threshold:
        return []
    return [(int(y), int(x), score)]


def _component_multi_peak(
    comp_response: np.ndarray,
    comp_mask: np.ndarray,
    expected_k: int,
    cfg: ComponentSplitConfig,
) -> list[tuple[int, int, float]]:
    """Return up to ``expected_k`` local maxima inside a component.

    When :attr:`ComponentSplitConfig.use_distance_transform` is on, v2
    augments the response-driven peak set with a second, independent seed
    pass over the component's Euclidean distance transform. The union of
    the two is then NMSed so isolated pupae (found by response) and
    touching pairs with identical response values (found by EDT local
    maxima at the two geometric centers) both get represented. Adding the
    EDT to the response directly does not work — EDT is *highest* at the
    geometric centers of each touching pupa but also on the interior
    skeleton, which would pull a single peak onto the valley and lose the
    pair. The two-pass union avoids that.

    Discipline rules baked in after the first v2 run overshot precision:

    * Total peaks per component are capped at ``max_peaks_per_component``
      and at ``round(expected_k * dt_expected_k_slack)``, whichever is
      smaller. The second cap keeps the area prior honest.
    * EDT seeds must clear the full ``abs_score_threshold`` at their pixel
      — no 0.7x discount. Otherwise the EDT pass fires on dim background
      fringes of thick components.
    * EDT seeds are deduped against the response pass at
      ``dedup_edt_min_distance_px`` (wider than the response↔response NMS
      distance) so they only add genuinely new coverage instead of
      duplicating existing peaks with a small offset.
    """
    if expected_k <= 0:
        return []

    response_masked = np.where(comp_mask, comp_response, 0.0).astype(np.float32)

    # Pass 1 — response-driven peaks (v1 behavior)
    response_coords = peak_local_max(
        response_masked,
        min_distance=int(cfg.min_peak_distance_px),
        num_peaks=int(expected_k),
        threshold_abs=float(cfg.abs_score_threshold),
        exclude_border=False,
    )

    response_pts: list[tuple[int, int, float]] = []
    for row in response_coords:
        y, x = int(row[0]), int(row[1])
        if not bool(comp_mask[y, x]):
            continue
        response_pts.append((y, x, float(comp_response[y, x])))
    response_pts.sort(key=lambda t: -t[2])

    # Pass 2 — EDT-driven seeds. Only meaningful when the component is
    # thick enough that the distance transform has genuine interior maxima.
    edt_pts: list[tuple[int, int, float]] = []
    if cfg.use_distance_transform:
        edt = ndi.distance_transform_edt(comp_mask).astype(np.float32)
        if float(edt.max()) >= cfg.dt_min_edt_px:
            # Ask for enough EDT candidates to cover any deficit and then
            # some extra slack, bounded below by 1.
            num_edt = max(
                1,
                int(round(expected_k * cfg.dt_expected_k_slack)),
            )
            edt_coords = peak_local_max(
                edt,
                min_distance=int(cfg.dt_min_distance_px),
                num_peaks=num_edt,
                threshold_abs=float(cfg.dt_min_edt_px),
                exclude_border=False,
            )
            dedup_d2 = float(cfg.dedup_edt_min_distance_px) ** 2
            for row in edt_coords:
                y, x = int(row[0]), int(row[1])
                if not bool(comp_mask[y, x]):
                    continue
                score = float(comp_response[y, x])
                # Full response threshold — no discount. Drops EDT seeds
                # that land on dim background skirts of thick components.
                if score < cfg.abs_score_threshold:
                    continue
                # Suppress EDT seeds that duplicate a response peak.
                duplicate = False
                for ry, rx, _ in response_pts:
                    dy = y - ry
                    dx = x - rx
                    if dy * dy + dx * dx < dedup_d2:
                        duplicate = True
                        break
                if duplicate:
                    continue
                edt_pts.append((y, x, score))

    all_pts = response_pts + edt_pts
    if not all_pts:
        return _component_single_peak(
            comp_response, comp_mask, cfg.abs_score_threshold
        )

    # Global cap so the area prior still governs the maximum count.
    hard_cap = min(
        int(cfg.max_peaks_per_component),
        max(1, int(round(expected_k * cfg.dt_expected_k_slack))),
    )

    all_pts.sort(key=lambda t: -t[2])
    accepted: list[tuple[int, int, float]] = []
    min_d2 = float(cfg.min_peak_distance_px) ** 2
    for y, x, s in all_pts:
        if len(accepted) >= hard_cap:
            break
        ok = True
        for ay, ax, _ in accepted:
            dy = y - ay
            dx = x - ax
            if dy * dy + dx * dx < min_d2:
                ok = False
                break
        if ok:
            accepted.append((y, x, s))
    return accepted


def _erosion_core_count(
    comp_mask: np.ndarray,
    *,
    erosion_radius: int,
    min_core_area: int,
) -> int:
    """Return the number of post-erosion sub-cores in a component mask.

    Uses a ``2*r + 1`` square structuring element. A k-pupa blob whose
    pupae meet through thin bridges breaks into k separate sub-components
    after the erosion; a single fat pupa stays as one.
    """
    if erosion_radius <= 0:
        return 1
    structure = np.ones((2 * int(erosion_radius) + 1, 2 * int(erosion_radius) + 1), dtype=bool)
    eroded = ndi.binary_erosion(comp_mask, structure=structure, border_value=0)
    if not eroded.any():
        return 0
    sub_labels, n_sub = ndi.label(eroded)
    if n_sub == 0:
        return 0
    sizes = np.bincount(sub_labels.ravel())
    # sizes[0] is background; real cores are sizes[1:]
    return int(((sizes[1:] >= int(min_core_area)).sum()))


def _response_core_count(
    comp_response: np.ndarray,
    comp_mask: np.ndarray,
    *,
    threshold: float,
    min_area: int,
) -> int:
    """Return the number of sub-components of ``response >= threshold`` within
    a permissive component mask.

    The "core mask" idea: at a stricter threshold the response valleys
    between touching pupae drop below the line, so a k-pupa blob separates
    cleanly into k sub-components even when the permissive allowed mask
    keeps them merged. The count of surviving sub-components then supplies
    a much sharper ``expected_k`` than the area prior for dense clusters.
    """
    core = (comp_response >= float(threshold)) & comp_mask
    if not core.any():
        return 0
    sub_labels, n_sub = ndi.label(core)
    if n_sub == 0:
        return 0
    sizes = np.bincount(sub_labels.ravel())
    return int((sizes[1:] >= int(min_area)).sum())


def detect_peaks_by_component(
    response: np.ndarray,
    allowed_mask: np.ndarray,
    *,
    cfg: ComponentSplitConfig | None = None,
    edge_margin_px: int = 4,
) -> pd.DataFrame:
    """Return peaks split per allowed-mask component.

    Drop-in replacement for ``peaks.detect_peaks`` in the v1 detector. Columns
    returned match the v0 contract: ``x``, ``y``, ``score``, plus an extra
    ``resolver_type`` column that records ``singleton_peak`` for components
    handled as one pupa, ``component_split`` for components divided by the
    area-prior, and ``erosion_split`` for components whose ``expected_k``
    was bumped by the erosion core counter.
    """
    cfg = cfg or ComponentSplitConfig()

    gate = allowed_mask > 0
    if not gate.any():
        return pd.DataFrame(columns=["x", "y", "score", "resolver_type"])

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

        area_based_k = max(
            1,
            int(round(area / max(1.0, cfg.single_pupa_area_px))),
        )

        resolver_tag = "singleton_peak"
        expected_k = area_based_k

        if cfg.use_erosion_core_count:
            core_k = _erosion_core_count(
                comp_mask,
                erosion_radius=int(cfg.erosion_radius_px),
                min_core_area=int(cfg.erosion_min_core_area_px),
            )
            if core_k > 0:
                # Take the maximum of the two priors so a thin-bridge blob
                # that erodes into 4 cores is not undercounted by an area
                # prior that rounds to 3. Bias by ``erosion_k_margin`` if
                # the user wants to lean further toward recall.
                boosted = core_k + int(cfg.erosion_k_margin)
                if boosted > expected_k:
                    expected_k = boosted
                    resolver_tag = "erosion_split"

        if cfg.use_response_core_mask:
            response_core_k = _response_core_count(
                comp_response,
                comp_mask,
                threshold=float(cfg.response_core_threshold),
                min_area=int(cfg.response_core_min_area_px),
            )
            if response_core_k > 0:
                boosted = response_core_k + int(cfg.response_core_k_margin)
                if boosted > expected_k:
                    expected_k = boosted
                    resolver_tag = "core_mask_split"

        expected_k = max(1, min(int(cfg.max_peaks_per_component), expected_k))

        if expected_k == 1 or area < cfg.area_ratio_threshold * cfg.single_pupa_area_px:
            peaks = _component_single_peak(
                comp_response, comp_mask, cfg.abs_score_threshold
            )
            if resolver_tag == "singleton_peak":
                pass
            elif len(peaks) <= 1:
                resolver_tag = "singleton_peak"
        else:
            peaks = _component_multi_peak(comp_response, comp_mask, expected_k, cfg)
            if len(peaks) > 1 and resolver_tag == "singleton_peak":
                resolver_tag = "component_split"

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
            xs.append(int(x_global))
            ys.append(int(y_global))
            scores.append(float(score))
            resolver_types.append(resolver_tag)

    return pd.DataFrame(
        {
            "x": xs,
            "y": ys,
            "score": scores,
            "resolver_type": resolver_types,
        }
    )


def tag_resolver_type(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """Back-compat tag for v0 that has no ``resolver_type`` column."""
    if "resolver_type" in peaks_df.columns:
        result = peaks_df.copy()
        result["status"] = "accepted"
        return result
    if peaks_df.empty:
        return peaks_df.assign(status="accepted", resolver_type="singleton_peak")
    result = peaks_df.copy()
    result["status"] = "accepted"
    result["resolver_type"] = "singleton_peak"
    return result
