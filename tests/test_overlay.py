from __future__ import annotations

import numpy as np
import pandas as pd

from pupa_counter.detect.components import build_component_row
from pupa_counter.report.overlay import build_overlay


def test_overlay_adds_labels_to_middle_instances_only():
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

    overlay_without_label = build_overlay(image, pd.DataFrame([control_row, top_row]))
    overlay_with_label = build_overlay(image, pd.DataFrame([middle_row, top_row]))

    label_patch_without = overlay_without_label[138:156, 44:62]
    label_patch_with = overlay_with_label[138:156, 44:62]

    assert np.any(label_patch_with != label_patch_without)
