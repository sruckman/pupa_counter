"""Keypoint head for ambiguous component resolution.

A tiny CNN that predicts a center heatmap from a native-resolution
component crop. Each pupa produces one Gaussian peak in the output.
Count = number of peaks. Positions = peak coordinates.

This replaces the expected_k estimation for ambiguous components,
directly predicting both count AND positions in one forward pass.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import peak_local_max

CROP_SIZE = 96
HEATMAP_SIZE = 48
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "tmp" / "keypoint_head.pt"


class KeypointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)).squeeze(1)


_model: KeypointHead | None = None


def _get_model() -> KeypointHead:
    global _model
    if _model is None:
        _model = KeypointHead()
        if MODEL_PATH.exists():
            _model.load_state_dict(torch.load(str(MODEL_PATH), weights_only=True, map_location="cpu"))
        _model.eval()
    return _model


def predict_centers(
    native_crop: np.ndarray,
    *,
    peak_threshold: float = 0.3,
    min_peak_distance: int = 4,
) -> list[tuple[float, float, float]]:
    """Predict pupa centers from a native-resolution crop.

    Returns list of (y_frac, x_frac, confidence) where y_frac and x_frac
    are in [0, 1] relative to the crop dimensions.
    """
    model = _get_model()

    # Prepare input
    resized = cv2.resize(native_crop, (CROP_SIZE, CROP_SIZE)).astype(np.float32) / 255.0
    tensor = torch.FloatTensor(resized).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        heatmap = model(tensor)[0].numpy()  # 48×48

    # Find peaks
    peaks = peak_local_max(
        heatmap,
        min_distance=min_peak_distance,
        threshold_abs=peak_threshold,
        exclude_border=False,
    )

    results = []
    for y, x in peaks:
        conf = float(heatmap[y, x])
        y_frac = float(y) / HEATMAP_SIZE
        x_frac = float(x) / HEATMAP_SIZE
        results.append((y_frac, x_frac, conf))

    return results
