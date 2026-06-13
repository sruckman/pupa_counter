"""Band assignment for fresh peak-first detector instances.

Each image is split into three horizontal bands based on the y-extent of
*its own* detected pupae. **Teacher semantics, NOT image-visual semantics:**
the scanned paper is placed on the scanner such that the physical top of
the paper appears at the visual BOTTOM of the image (large centroid_y).
Therefore:

* ``top``    — visually at the image bottom — ``centroid_y > bottom_y - 0.25 * span``
* ``middle`` — everything between
* ``bottom`` — visually at the image top   — ``centroid_y < top_y + 0.25 * span``

``is_top_5pct`` flags rows at ``centroid_y >= bottom_y - 0.05 * span``, i.e.
the 5% closest to the visual image bottom (teacher's physical "top"). The
legacy ``pupa_counter.count.anchors`` code has the opposite convention —
intentionally diverged here so the Excel column ``顶部5%`` actually means
what the teacher expects when they look at the printed scan.

The extrema are computed per-image so the split is relative to the detected
distribution — not fixed to the image canvas. Kept intentionally small: no
``BandGeometry`` dataclass, no bbox blending, no anchor-mode branching.
"""

from __future__ import annotations

import pandas as pd


def assign_bands(instances_df: pd.DataFrame) -> pd.DataFrame:
    if instances_df.empty:
        return instances_df.assign(band="unassigned", is_top_5pct=False)

    frame = instances_df.copy()
    positions = frame["centroid_y"].astype(float)
    img_top_y = float(positions.min())        # visual top of image, small y
    img_bottom_y = float(positions.max())     # visual bottom of image, large y
    span = img_bottom_y - img_top_y

    # Teacher "top" = physical top of paper = visual IMAGE BOTTOM (large y).
    top_5pct_cutoff = img_bottom_y - 0.05 * span     # rows with y >= this  -> top 5%
    top_band_cutoff = img_bottom_y - 0.25 * span     # rows with y >  this  -> top
    bottom_band_cutoff = img_top_y + 0.25 * span     # rows with y <  this  -> bottom

    frame["is_top_5pct"] = positions >= top_5pct_cutoff
    frame["band"] = "middle"
    frame.loc[positions > top_band_cutoff, "band"] = "top"
    frame.loc[positions < bottom_band_cutoff, "band"] = "bottom"
    return frame
