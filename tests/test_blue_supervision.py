from __future__ import annotations

import cv2
import numpy as np

from pupa_counter.annotate.blue_supervision import extract_blue_components, summarize_blue_supervision


def test_blue_supervision_extracts_lines_and_dots():
    mask = np.zeros((200, 120), dtype=np.uint8)
    cv2.line(mask, (5, 40), (115, 40), 255, 2)
    cv2.line(mask, (5, 120), (115, 120), 255, 2)
    for x, y in [(20, 60), (40, 70), (60, 80), (80, 90), (100, 100)]:
        cv2.circle(mask, (x, y), 2, 255, -1)

    components = extract_blue_components(mask, mask.shape)
    summary = summarize_blue_supervision(components, mask, mask.shape)

    assert summary["trusted_line_upper_y"] is not None
    assert summary["trusted_line_lower_y"] is not None
    assert summary["trusted_dot_total"] == 5
    assert summary["trusted_dot_middle"] == 5
    assert summary["dots_look_human_placed"] is True


def test_blue_supervision_rejects_clean_pdf_noise():
    """Clean PDFs without any annotations were classified as 'dot_count_only'
    with bogus 168-310 trusted dot counts because scan noise tripped the
    blue mask. The annotation_mode classifier must reject this and report
    annotation_mode='clean' with trusted_dot_total=None."""
    rng = np.random.default_rng(42)
    mask = np.zeros((1200, 800), dtype=np.uint8)
    # 200 isolated tiny noise spots scattered like scan dust.
    for _ in range(200):
        y = int(rng.integers(50, 1150))
        x = int(rng.integers(50, 750))
        cv2.circle(mask, (x, y), 1, 255, -1)

    components = extract_blue_components(mask, mask.shape)
    summary = summarize_blue_supervision(components, mask, mask.shape)

    assert summary["annotation_mode"] == "clean"
    assert summary["trusted_dot_total"] is None
    assert summary["trusted_dot_middle"] is None
    assert summary["dots_look_human_placed"] is False
    # Raw count is still preserved for debugging.
    assert summary["raw_blue_dot_component_count"] >= 100


def test_blue_supervision_passes_real_dense_annotation():
    """A realistic annotated page (50 dots, two horizontal band lines well
    inside the non-footer zone) must still classify as dot_count_with_lines
    and surface a trusted total."""
    mask = np.zeros((1200, 800), dtype=np.uint8)
    # Both lines must sit above the footer zone (y < 1200 * 0.72 = 864)
    # to be picked up as horizontal candidates.
    cv2.line(mask, (40, 200), (760, 200), 255, 4)
    cv2.line(mask, (40, 700), (760, 700), 255, 4)
    rng = np.random.default_rng(7)
    placed = 0
    while placed < 50:
        y = int(rng.integers(220, 680))
        x = int(rng.integers(80, 720))
        cv2.circle(mask, (x, y), 4, 255, -1)
        placed += 1

    components = extract_blue_components(mask, mask.shape)
    summary = summarize_blue_supervision(components, mask, mask.shape)

    assert summary["annotation_mode"] == "dot_count_with_lines"
    assert summary["trusted_confidence"] == "high"
    assert summary["trusted_dot_total"] == 50
    assert summary["trusted_dot_middle"] == 50
