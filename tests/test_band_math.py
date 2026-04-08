from __future__ import annotations

import pandas as pd

from pupa_counter.count.anchors import compute_band_geometry
from pupa_counter.count.assign import assign_bands


def test_band_geometry_uses_quarter_and_three_quarters():
    instances = pd.DataFrame(
        {
            "centroid_y": [10.0, 50.0, 90.0],
            "bbox_y0": [8.0, 48.0, 88.0],
            "bbox_y1": [12.0, 52.0, 92.0],
        }
    )
    geometry = compute_band_geometry(instances, anchor_mode="centroid")
    assert geometry.top_y == 10.0
    assert geometry.bottom_y == 90.0
    assert geometry.upper_middle_y == 30.0
    assert geometry.lower_middle_y == 70.0

    assigned = assign_bands(instances, geometry)
    assert assigned["band"].tolist() == ["top", "middle", "bottom"]
