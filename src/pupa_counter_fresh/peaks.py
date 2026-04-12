"""Peak proposal and NMS for the fresh peak-first detector.

Stage B of the pipeline:

* Local maxima of the response map, restricted to the allowed mask.
* Non-maximum suppression with a minimum-distance footprint.
* Absolute score threshold to drop background false positives.

The output is deliberately a plain pandas DataFrame so it composes cleanly
with the audit harness and the disagreement mining scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


@dataclass
class PeakConfig:
    min_distance_px: int = 14
    """Minimum separation between accepted peaks, in *detector* pixels.

    At 0.67x downscale, pupae have major axis ≈ 27 px so ``14`` keeps two
    touching centroids separable without fragmenting a single pupa.
    """

    abs_score_threshold: float = 0.30
    """Absolute response threshold a peak must clear to be accepted."""

    plateau_erosion_radius: int = 1
    """Morphological erosion radius used to collapse flat plateaus before
    selecting local maxima. ``0`` disables it."""

    edge_margin_px: int = 4
    """Reject peaks within this many *detector* pixels of any image border.

    Scanned images have CLAHE / percentile-normalization edge artifacts that
    look like peaks but sit on the image frame itself. The v0 smoke test saw
    several peaks pinned to ``x = 0``; ``4`` px at 0.67x is ~6 native px,
    well inside the paper ROI but enough to cut those edge artifacts out."""


def _local_maxima(response: np.ndarray, footprint_radius: int) -> np.ndarray:
    """Return a boolean mask of strict local maxima of ``response``.

    Uses a square footprint of side ``2r + 1``; ties at the same value are
    broken by keeping all equal pixels, which is fine because the NMS step
    deduplicates later.
    """
    size = max(3, 2 * int(footprint_radius) + 1)
    dilated = ndi.maximum_filter(response, size=size, mode="nearest")
    return response == dilated


def detect_peaks(
    response: np.ndarray,
    allowed_mask: np.ndarray,
    *,
    cfg: PeakConfig | None = None,
) -> pd.DataFrame:
    """Return candidate centers as a DataFrame in detector-pixel coordinates.

    Columns: ``x``, ``y``, ``score``. The caller is responsible for rescaling
    coordinates back to native resolution if the image was downscaled.
    """
    cfg = cfg or PeakConfig()

    work = response.astype(np.float32, copy=True)
    gate = allowed_mask > 0
    work[~gate] = 0.0

    if cfg.plateau_erosion_radius > 0:
        kernel_size = 2 * int(cfg.plateau_erosion_radius) + 1
        eroded = ndi.grey_erosion(work, size=(kernel_size, kernel_size))
        # Subtract a tiny fraction of the eroded response to break plateaus
        # so that maximum_filter picks a single pixel instead of N equal ones.
        work = work - 1e-4 * eroded

    is_max = _local_maxima(work, cfg.min_distance_px // 2)
    candidate = is_max & gate & (response >= cfg.abs_score_threshold)

    if cfg.edge_margin_px > 0:
        m = int(cfg.edge_margin_px)
        if m > 0 and candidate.shape[0] > 2 * m and candidate.shape[1] > 2 * m:
            edge_cut = np.zeros_like(candidate)
            edge_cut[m : candidate.shape[0] - m, m : candidate.shape[1] - m] = True
            candidate &= edge_cut

    if not candidate.any():
        return pd.DataFrame(columns=["x", "y", "score"])

    ys, xs = np.where(candidate)
    scores = response[ys, xs]

    order = np.argsort(-scores)
    ys = ys[order]
    xs = xs[order]
    scores = scores[order]

    # Greedy NMS using squared distance — keeps it fast on ~1-2k candidates.
    min_d2 = float(cfg.min_distance_px) ** 2
    kept_x: list[int] = []
    kept_y: list[int] = []
    kept_score: list[float] = []
    for x, y, s in zip(xs.tolist(), ys.tolist(), scores.tolist()):
        ok = True
        for kx, ky in zip(kept_x, kept_y):
            dx = x - kx
            dy = y - ky
            if dx * dx + dy * dy < min_d2:
                ok = False
                break
        if ok:
            kept_x.append(x)
            kept_y.append(y)
            kept_score.append(float(s))

    return pd.DataFrame({"x": kept_x, "y": kept_y, "score": kept_score})
