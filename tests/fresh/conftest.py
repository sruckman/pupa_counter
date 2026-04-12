"""Shared fixtures for the ``pupa_counter_fresh`` test suite.

The point of these fixtures is to keep every test *hermetic*: no test in this
package may read from the user's local ``/Users/stephenyu/...`` probe data.
All images are synthesized on the fly so ``pytest tests/fresh`` runs on any
machine.

Synthetic pupae are painted as **elliptical Gaussian blobs** rather than
filled ellipses. Filled ellipses produce flat-topped response maps, and
LoG / DoG of a flat-topped blob peaks at the *edge* of the plateau (a
ring shape), not the center. Real pupae have Gaussian-like intensity
falloff, so a Gaussian-blob fixture is the right representative of the
real failure mode we're building the v2 sharpener against.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import pytest


PupaCenter = tuple[int, int]


def _paint_gaussian_pupa(
    rgb: np.ndarray,
    center: PupaCenter,
    *,
    sigma_major: float,
    sigma_minor: float,
    angle_deg: float,
    color_rgb: tuple[int, int, int],
) -> None:
    """Paint an elliptical Gaussian pupa into an RGB array in-place.

    The Gaussian is anchored at ``center``; intensity blends the existing
    background toward ``color_rgb`` with weight ``exp(-r²/2σ²)``. Rotation
    is supported via ``angle_deg`` so elongated pupae can be oriented
    arbitrarily.
    """
    h, w = rgb.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = x - float(center[0])
    dy = y - float(center[1])

    angle = np.deg2rad(angle_deg)
    ca = float(np.cos(angle))
    sa = float(np.sin(angle))
    xp = ca * dx + sa * dy
    yp = -sa * dx + ca * dy

    sigma_major = max(1e-3, float(sigma_major))
    sigma_minor = max(1e-3, float(sigma_minor))
    gauss = np.exp(
        -(xp * xp) / (2.0 * sigma_major * sigma_major)
        - (yp * yp) / (2.0 * sigma_minor * sigma_minor)
    ).astype(np.float32)

    rgb_f = rgb.astype(np.float32)
    one_minus = 1.0 - gauss
    for c in range(3):
        rgb_f[:, :, c] = rgb_f[:, :, c] * one_minus + float(color_rgb[c]) * gauss
    np.clip(rgb_f, 0, 255, out=rgb_f)
    rgb[:] = rgb_f.astype(np.uint8)


@pytest.fixture
def make_synthetic_pupa_image():
    """Build a synthetic RGB image with brown Gaussian-blob pupae on white paper.

    Returned callable signature::

        rgb = make_synthetic_pupa_image(
            size=(256, 256),
            centers=[(120, 128), (130, 128)],
            sigma_major=4.0,
            sigma_minor=3.0,
            angle_deg=0.0,
            pupa_color=(130, 75, 35),   # RGB
            bg_color=(245, 245, 245),
        )

    Defaults produce a pupa with sigma ≈ (4, 3) — close to the effective
    Gaussian width of a real 0.67x-scaled benchmark pupa. Two such pupae
    at 10 px separation are resolvable by LoG / DoG but not by a naive
    Gaussian smoothing with sigma=1.2, which is the v1 → v2 regression
    case we care about.
    """

    def _make(
        size: tuple[int, int] = (256, 256),
        centers: Sequence[PupaCenter] | None = None,
        sigma_major: float = 4.0,
        sigma_minor: float = 3.0,
        angle_deg: float = 0.0,
        pupa_color: tuple[int, int, int] = (130, 75, 35),
        bg_color: tuple[int, int, int] = (245, 245, 245),
    ) -> np.ndarray:
        h, w = size
        img = np.full((h, w, 3), bg_color, dtype=np.uint8)
        if centers:
            for center in centers:
                _paint_gaussian_pupa(
                    img,
                    center,
                    sigma_major=sigma_major,
                    sigma_minor=sigma_minor,
                    angle_deg=angle_deg,
                    color_rgb=pupa_color,
                )
        return img

    return _make


@pytest.fixture
def make_blue_ink_image():
    """Build a white image with a single thin blue ink stroke.

    Used to verify that every ``compute_response_map`` mode zeroes response
    inside the blue annotation mask.
    """

    def _make(size: tuple[int, int] = (128, 128)) -> np.ndarray:
        h, w = size
        img = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)
        cv2.line(
            img,
            pt1=(20, h // 2),
            pt2=(w - 20, h // 2),
            color=(0, 80, 230),  # deep blue in RGB
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        return img

    return _make
