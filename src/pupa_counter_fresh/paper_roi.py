"""Paper region of interest detection for the fresh peak-first detector.

The 2026-04-10 visual audit found that a big chunk of both v1's false
positives and v8's spurious teacher labels sit on the *scanner gray
strip* at the left edge of the image — a dark vertical chrome bar that
passes the brown/dark test because it is legitimately dark, but is
obviously not a pupa. A paper ROI mask cuts the problem at the source:
compute the largest bright connected region (the actual paper), erode
it inward by a small margin, and zero out any response that falls
outside.

This is a Stage-A additive step, not a model change. Every current
detector parameter stays the same; the ROI mask just restricts where
peaks can be emitted.

Why largest-connected-component instead of a simple intensity threshold:
on a scan with hundreds of brown pupae, a naive ``gray > 180`` mask
produces thousands of tiny holes. After a generous morphological close
the pupae get filled in and the paper becomes one solid blob. Picking
the largest connected component is the cleanest way to isolate it from
the scanner chrome, page numbers, staple marks, and anything else the
scanner captured in the margins.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class PaperROIConfig:
    brightness_threshold: int = 180
    """Grayscale value above which a pixel is 'paper-ish'. The scanner
    produces paper at ~245 gray and the chrome bar at ~60 gray, so any
    value in the 120–200 range works — 180 gives a comfortable margin."""

    close_kernel_px: int = 15
    """Square morphological close kernel size. Must be larger than the
    biggest pupa diameter so pupae (dark holes in paper) get filled in
    before the largest connected component step."""

    erode_margin_px: int = 6
    """Inward erosion of the detected paper boundary. Pulls the mask a
    few pixels inside the true paper edge so peaks sitting directly on
    the edge transition are also suppressed."""

    min_paper_fraction: float = 0.05
    """Reject the detection (return ``None``) if the largest connected
    component covers less than this fraction of the image. Defends
    against completely black or completely white inputs."""


def detect_paper_roi(
    rgb: np.ndarray,
    *,
    cfg: PaperROIConfig | None = None,
) -> np.ndarray | None:
    """Return an ``HxW`` uint8 mask of the paper region, or ``None``.

    ``None`` means the algorithm could not identify a confident paper
    region — the caller should fall back to no ROI restriction.
    """
    cfg = cfg or PaperROIConfig()

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    paper_candidate = (gray > int(cfg.brightness_threshold)).astype(np.uint8) * 255
    if paper_candidate.sum() == 0:
        return None

    k = max(3, int(cfg.close_kernel_px) | 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    closed = cv2.morphologyEx(paper_candidate, cv2.MORPH_CLOSE, kernel)

    labels, n = ndi.label(closed > 0)
    if n == 0:
        return None

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # exclude background label
    largest = int(np.argmax(sizes))
    largest_size = int(sizes[largest])
    if largest_size <= 0:
        return None

    fraction = largest_size / float(h * w)
    if fraction < float(cfg.min_paper_fraction):
        return None

    paper_mask = (labels == largest).astype(np.uint8) * 255

    if cfg.erode_margin_px > 0:
        em = int(cfg.erode_margin_px)
        erode_kernel = np.ones((2 * em + 1, 2 * em + 1), dtype=np.uint8)
        paper_mask = cv2.erode(paper_mask, erode_kernel)

    return paper_mask


def apply_paper_roi_to_response(
    response: np.ndarray,
    paper_mask: np.ndarray | None,
) -> np.ndarray:
    """Return ``response`` zeroed outside ``paper_mask``.

    If ``paper_mask`` is ``None`` the response is returned unchanged, so
    callers can safely pipe through ``detect_paper_roi`` without branching
    on the paper-detect-succeeded case.
    """
    if paper_mask is None:
        return response
    out = response.copy()
    out[paper_mask == 0] = 0.0
    return out
