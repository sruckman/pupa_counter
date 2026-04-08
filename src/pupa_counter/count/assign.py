"""Band assignment for final instances."""

from __future__ import annotations

import pandas as pd

from pupa_counter.types import BandGeometry


def assign_bands(instances_df: pd.DataFrame, geometry: BandGeometry) -> pd.DataFrame:
    if instances_df.empty:
        return instances_df.copy()
    frame = instances_df.copy()
    positions = frame["centroid_y"].astype(float)
    frame["band"] = "middle"
    frame.loc[positions < geometry.upper_middle_y, "band"] = "top"
    frame.loc[positions > geometry.lower_middle_y, "band"] = "bottom"
    return frame
