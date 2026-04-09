from __future__ import annotations

import numpy as np
import pandas as pd

from pupa_counter.count.anchors import compute_band_geometry
from pupa_counter.detect.components import build_component_row
from pupa_counter.report.overlay import build_overlay


def test_overlay_adds_labels_to_middle_instances_only_when_enabled():
    image = np.zeros((240, 120, 3), dtype=np.uint8)

    middle_mask = np.zeros((240, 120), dtype=bool)
    middle_mask[150:166, 30:44] = True
    top_mask = np.zeros((240, 120), dtype=bool)
    top_mask[190:206, 70:84] = True

    top_row = build_component_row(top_mask, 0, 0, top_mask.shape, "cmp_top")
    top_row["band"] = "top"
    top_row["synthetic_instance"] = False

    control_row = build_component_row(middle_mask, 0, 0, middle_mask.shape, "cmp_control")
    control_row["band"] = "top"
    control_row["synthetic_instance"] = False

    middle_row = build_component_row(middle_mask, 0, 0, middle_mask.shape, "cmp_middle")
    middle_row["band"] = "middle"
    middle_row["synthetic_instance"] = False

    overlay_without_label = build_overlay(image, pd.DataFrame([control_row, top_row]), show_middle_labels=True)
    overlay_with_label = build_overlay(image, pd.DataFrame([middle_row, top_row]), show_middle_labels=True)

    label_patch_without = overlay_without_label[138:156, 44:62]
    label_patch_with = overlay_with_label[138:156, 44:62]

    assert np.any(label_patch_with != label_patch_without)


def test_overlay_hides_middle_labels_by_default():
    image = np.zeros((240, 120, 3), dtype=np.uint8)
    middle_mask = np.zeros((240, 120), dtype=bool)
    middle_mask[150:166, 30:44] = True
    top_mask = np.zeros((240, 120), dtype=bool)
    top_mask[190:206, 70:84] = True

    middle_row = build_component_row(middle_mask, 0, 0, middle_mask.shape, "cmp_middle")
    middle_row["band"] = "middle"
    middle_row["synthetic_instance"] = False

    top_row = build_component_row(top_mask, 0, 0, top_mask.shape, "cmp_top")
    top_row["band"] = "top"
    top_row["synthetic_instance"] = False

    overlay_default = build_overlay(image, pd.DataFrame([middle_row, top_row]))
    overlay_labeled = build_overlay(image, pd.DataFrame([middle_row, top_row]), show_middle_labels=True)

    label_patch_default = overlay_default[138:156, 44:62]
    label_patch_labeled = overlay_labeled[138:156, 44:62]

    assert np.any(label_patch_labeled != label_patch_default)


def test_overlay_draws_five_percent_guide_when_geometry_exists():
    image = np.zeros((200, 120, 3), dtype=np.uint8)
    frame = pd.DataFrame(
        {
            "centroid_x": [40.0, 60.0, 50.0],
            "centroid_y": [20.0, 100.0, 180.0],
            "bbox_x0": [35.0, 55.0, 45.0],
            "bbox_y0": [15.0, 95.0, 175.0],
            "bbox_x1": [45.0, 65.0, 55.0],
            "bbox_y1": [25.0, 105.0, 185.0],
            "mask": [np.ones((10, 10), dtype=bool)] * 3,
            "band": ["top", "middle", "bottom"],
            "synthetic_instance": [False, False, False],
            "confidence": [0.9, 0.9, 0.9],
        }
    )
    geometry = compute_band_geometry(frame, anchor_mode="centroid")
    overlay = build_overlay(image, frame, geometry=geometry)
    y = int(round(geometry.upper_five_pct_y))
    assert np.any(overlay[max(0, y - 1): min(overlay.shape[0], y + 2), 0:40] != 0)
