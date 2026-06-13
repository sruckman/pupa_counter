from __future__ import annotations

import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.cellpose_dense_patch import refine_dense_cellpose_patches
from pupa_counter.detect.components import build_component_row


def _build_rows_from_boxes(boxes, shape, prefix):
    rows = []
    for idx, (y0, y1, x0, x1) in enumerate(boxes, start=1):
        mask = np.zeros(shape, dtype=bool)
        mask[y0:y1, x0:x1] = True
        row = build_component_row(mask, 0, 0, shape, f"{prefix}_{idx:03d}")
        row["image_height"] = shape[0]
        row["image_width"] = shape[1]
        rows.append(row)
    return pd.DataFrame(rows)


def test_dense_patch_refine_replaces_cluster_with_modest_gain():
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    boxes = [
        (60, 76, 50, 62),
        (62, 78, 66, 78),
        (66, 82, 82, 94),
        (84, 100, 56, 68),
        (86, 102, 72, 84),
        (90, 106, 88, 100),
        (108, 124, 60, 72),
        (110, 126, 78, 90),
    ]
    components = _build_rows_from_boxes(boxes, image.shape[:2], "cp")
    cfg = AppConfig()

    def fake_detect(_patch, _cfg, **kwargs):
        patch_shape = _patch.shape[:2]
        offset_row = int(kwargs["offset_row"])
        offset_col = int(kwargs["offset_col"])
        global_shape = kwargs["global_image_shape"]
        refined_boxes = [
            (10, 24, 10, 20),
            (12, 26, 24, 34),
            (14, 28, 38, 48),
            (30, 44, 14, 24),
            (32, 46, 28, 38),
            (34, 48, 42, 52),
            (50, 64, 18, 28),
            (52, 66, 32, 42),
            (54, 68, 46, 56),
        ]
        rows = []
        for idx, (y0, y1, x0, x1) in enumerate(refined_boxes, start=1):
            local_mask = np.zeros(patch_shape, dtype=bool)
            local_mask[y0:y1, x0:x1] = True
            row = build_component_row(
                local_mask,
                offset_row,
                offset_col,
                global_shape,
                component_id=f"fake_{idx:03d}",
            )
            row["image_height"] = global_shape[0]
            row["image_width"] = global_shape[1]
            rows.append(row)
        return pd.DataFrame(rows)

    refined = refine_dense_cellpose_patches(
        image,
        components,
        source_type="annotated_png",
        cfg=cfg,
        detect_fn=fake_detect,
    )

    assert len(refined) == 9
    assert int(refined["dense_patch_refined"].astype(bool).sum()) == 9
    assert set(refined["component_id"]) == {f"fake_{idx:03d}" for idx in range(1, 10)}
