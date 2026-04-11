"""Band assignment — deliberately deferred in v0.

The fresh-start plan is explicit that band geometry is the *last* thing to
tune. v0 exposes a no-op that writes ``band = "unassigned"`` into the export
frame so downstream consumers don't crash on a missing column.
"""

from __future__ import annotations

import pandas as pd


def assign_bands(instances_df: pd.DataFrame) -> pd.DataFrame:
    if instances_df.empty:
        return instances_df.assign(band="unassigned", is_top_5pct=False)
    result = instances_df.copy()
    result["band"] = "unassigned"
    result["is_top_5pct"] = False
    return result
