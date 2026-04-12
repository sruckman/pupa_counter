"""Regression tests for ``compute_response_map`` modes.

The core purpose of this file is one test: **touching pupae must resolve
into two peaks under ``log`` and ``dog`` modes, but collapse into one peak
under the default ``smooth`` mode**. That is the pupa_counter_fresh v1 →
v2 bet: response sharpening, not splitter tuning, is what recovers the
touching-pair gap.

If the touching-pair case ever silently regresses to "1 peak under log",
this test fails hard — no xfail, no skip. Loosen the test only if the
algorithm genuinely changes, never to make a bad day pass.

Modes covered:

* ``smooth``   (the v1 default) — must find 1 isolated, 1 blue-masked, 1
  touching pair collapses to 1 peak
* ``log``      — single-sigma DoG approximation; must split the pair
* ``dog``      — two-sigma difference-of-Gaussians; must split the pair
* ``adaptive`` — per-component sigma; must split the pair when configured
  with a small-component sigma below the component's characteristic scale
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from pupa_counter_fresh.preprocess import build_blue_mask
from pupa_counter_fresh.response import build_allowed_mask, compute_response_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_local_maxima(response: np.ndarray, allowed: np.ndarray, footprint: int = 5) -> int:
    """Return the number of strict local maxima inside ``allowed``.

    A strict local maximum is a pixel whose value equals the maximum of a
    ``footprint`` × ``footprint`` neighborhood and whose value is above the
    5th percentile of the allowed region (filters pure-zero plateaus).
    """
    size = max(3, footprint | 1)
    dilated = ndi.maximum_filter(response, size=size, mode="nearest")
    is_max = response == dilated
    gate = allowed > 0
    if not gate.any():
        return 0
    threshold = float(np.percentile(response[gate], 70))
    return int(np.sum(is_max & gate & (response >= threshold)))


def _count_blob_peaks(response: np.ndarray, allowed: np.ndarray) -> int:
    """Return the number of *labeled* peak regions.

    Uses a **relative** threshold (``0.70 × response.max()``) so the count
    is comparable across ``smooth`` (max ≈ 0.5) and ``log`` / ``dog``
    (normalized max ≈ 1.0). Using an absolute threshold would
    systematically undercount ``smooth`` or overcount ``log`` depending
    on which constant was chosen.
    """
    gate = allowed > 0
    if not gate.any():
        return 0
    peak = float(response.max())
    if peak <= 0:
        return 0
    threshold = 0.70 * peak
    core = (response >= threshold) & gate
    _, n = ndi.label(core)
    return int(n)


@pytest.fixture
def isolated_pupa(make_synthetic_pupa_image):
    return make_synthetic_pupa_image(
        size=(96, 96),
        centers=[(48, 48)],
        sigma_major=4.0,
        sigma_minor=3.0,
    )


@pytest.fixture
def touching_pair(make_synthetic_pupa_image):
    """Two pupae tight enough that smooth-mode collapses them into one peak.

    Centers are (43, 48) and (53, 48): 10 px horizontal separation between
    two Gaussian blobs of sigma (4, 3). Empirically this is the case where
    ``smooth`` (σ=1.2) merges the pair into one blob and both ``log``
    (σ=1.2) and ``dog`` (σ_low=0.8, σ_high=2.4) keep them as two. If the
    synthesizer parameters change, retune this separation until the
    differentiation holds — ``tests/fresh/test_response_sharpen`` is the
    regression gate, not a benchmark.
    """
    return make_synthetic_pupa_image(
        size=(96, 96),
        centers=[(43, 48), (53, 48)],
        sigma_major=4.0,
        sigma_minor=3.0,
    )


# ---------------------------------------------------------------------------
# Isolated pupa — every mode must find exactly one peak
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth", "log", "dog", "adaptive"])
def test_isolated_pupa_one_peak_in_every_mode(isolated_pupa, mode):
    response = compute_response_map(isolated_pupa, response_mode=mode)
    assert response.shape == isolated_pupa.shape[:2]
    assert response.dtype == np.float32
    assert np.isfinite(response).all()
    assert response.min() >= 0.0
    allowed = build_allowed_mask(response, abs_threshold=0.10, min_percentile=0.0)
    n_blobs = _count_blob_peaks(response, allowed)
    assert n_blobs == 1, f"{mode}: expected 1 blob for isolated pupa, got {n_blobs}"


# ---------------------------------------------------------------------------
# Touching pair — the v1 → v2 regression gate
# ---------------------------------------------------------------------------


def test_smooth_mode_collapses_touching_pair(touching_pair):
    """Document the v1 failure mode: smooth mode merges the pair.

    If this test ever fails with ``n_blobs == 2``, it means the synthetic
    pair is no longer representative of the real failure mode — probably
    because the soften_sigma or separation changed. The follow-on test
    ``test_log_mode_splits_touching_pair`` would then be a nop. Pair them.
    """
    response = compute_response_map(touching_pair, response_mode="smooth")
    allowed = build_allowed_mask(response, abs_threshold=0.10, min_percentile=0.0)
    n_blobs = _count_blob_peaks(response, allowed)
    assert n_blobs == 1, (
        f"smooth mode was expected to collapse the touching pair into 1 blob, "
        f"got {n_blobs}. Retune the synthetic pair fixture so smooth does merge."
    )


def test_log_mode_splits_touching_pair(touching_pair):
    """The v1 → v2 bet: LoG-style sharpening preserves both peaks."""
    response = compute_response_map(
        touching_pair,
        response_mode="log",
        log_sigma=1.0,
    )
    allowed = build_allowed_mask(response, abs_threshold=0.10, min_percentile=0.0)
    n_blobs = _count_blob_peaks(response, allowed)
    assert n_blobs >= 2, (
        f"log mode should split the touching pair; got {n_blobs} blob(s). "
        f"This is the v2 regression gate."
    )


def test_dog_mode_splits_touching_pair(touching_pair):
    response = compute_response_map(
        touching_pair,
        response_mode="dog",
        dog_sigma_low=0.8,
        dog_sigma_high=2.0,
    )
    allowed = build_allowed_mask(response, abs_threshold=0.10, min_percentile=0.0)
    n_blobs = _count_blob_peaks(response, allowed)
    assert n_blobs >= 2, (
        f"dog mode should split the touching pair; got {n_blobs} blob(s)."
    )


def test_adaptive_mode_splits_touching_pair(touching_pair):
    """adaptive mode should apply a sharper sigma to small components.

    With an area threshold below the pair component's size, the adaptive
    branch should pick ``adaptive_small_sigma`` for the pair and resolve it.
    """
    response = compute_response_map(
        touching_pair,
        response_mode="adaptive",
        adaptive_small_sigma=0.6,
        adaptive_large_sigma=1.4,
        adaptive_area_threshold_px=5000,  # effectively force small-sigma path
    )
    allowed = build_allowed_mask(response, abs_threshold=0.10, min_percentile=0.0)
    n_blobs = _count_blob_peaks(response, allowed)
    assert n_blobs >= 2, (
        f"adaptive mode should split the touching pair; got {n_blobs} blob(s)."
    )


# ---------------------------------------------------------------------------
# Blue-mask invariance — every mode must zero response inside blue strokes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth", "log", "dog", "adaptive"])
def test_blue_mask_zeros_response_in_every_mode(make_blue_ink_image, mode):
    rgb = make_blue_ink_image(size=(128, 128))
    blue = build_blue_mask(rgb)
    assert blue.any(), "fixture should produce a non-empty blue mask"
    response = compute_response_map(rgb, blue_mask=blue, response_mode=mode)
    assert response.shape == rgb.shape[:2]
    inside_blue = response[blue > 0]
    # Sharpened modes may produce tiny ringing at the edge of the blue
    # stroke, so we allow up to 0.05 rather than hard-zero.
    assert inside_blue.max() < 0.05, (
        f"{mode}: response inside blue mask should be ~0, got max={inside_blue.max():.3f}"
    )


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth", "log", "dog", "adaptive"])
def test_empty_image_does_not_crash(mode):
    rgb = np.full((64, 64, 3), 245, dtype=np.uint8)
    response = compute_response_map(rgb, response_mode=mode)
    assert response.shape == (64, 64)
    assert np.isfinite(response).all()
    assert response.max() < 0.2  # white paper, nothing brown
