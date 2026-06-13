from __future__ import annotations

import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.rule_filter import rule_classify_components


def _row(**overrides):
    base = {
        "component_id": "x",
        "area_px": 85.0,
        "solidity": 0.82,
        "eccentricity": 0.92,
        "aspect_ratio": 2.2,
        "color_score": 0.44,
        "local_contrast": 22.0,
        "mean_s": 90.0,
        "mean_v": 160.0,
        "blue_overlap_ratio": 0.0,
        "border_touch_ratio": 0.0,
        "major_axis_px": 18.0,
        "extent": 0.72,
        "image_height": 2400,
        "image_width": 1200,
    }
    base.update(overrides)
    return base


def test_rule_filter_separates_pupa_and_artifact():
    frame = pd.DataFrame(
        [
            _row(component_id="good"),
            _row(
                component_id="bad",
                area_px=12.0,
                solidity=0.20,
                eccentricity=0.10,
                aspect_ratio=1.0,
                color_score=0.02,
                local_contrast=1.0,
                mean_s=4.0,
                mean_v=250.0,
                major_axis_px=4.0,
                extent=0.20,
            ),
        ]
    )

    labeled = rule_classify_components(frame, AppConfig())
    labels = dict(zip(labeled["component_id"], labeled["label"]))
    assert labels["good"] == "pupa"
    assert labels["bad"] == "artifact"


def test_rule_filter_rejects_oversized_stain_as_artifact():
    """A blob 50x the median pupa area is a paper stain or scan artifact —
    not a pile of 8 pupae the cluster_fallback should fabricate."""
    frame = pd.DataFrame(
        [
            # 30 normal-sized pupae establish the median area baseline.
            *(_row(component_id="pupa_%02d" % i) for i in range(30)),
            # One huge "cluster" candidate, far bigger than any plausible
            # pile of pupae and bright (typical paper stain).
            _row(
                component_id="stain",
                area_px=85.0 * 50,
                major_axis_px=200.0,
                mean_v=230.0,
                extent=0.4,
                solidity=0.75,
            ),
        ]
    )
    labeled = rule_classify_components(frame, AppConfig())
    label_lookup = dict(zip(labeled["component_id"], labeled["label"]))
    assert label_lookup["stain"] == "artifact"


def test_rule_filter_rejects_border_hugging_cluster():
    """The black noise band along the top/right edge of a PDF scan is a
    cluster-shaped blob with very high border_touch_ratio. Reject it."""
    frame = pd.DataFrame(
        [
            *(_row(component_id="pupa_%02d" % i) for i in range(20)),
            _row(
                component_id="edge_band",
                area_px=85.0 * 6,
                major_axis_px=80.0,
                border_touch_ratio=0.85,
            ),
        ]
    )
    labeled = rule_classify_components(frame, AppConfig())
    label_lookup = dict(zip(labeled["component_id"], labeled["label"]))
    assert label_lookup["edge_band"] == "artifact"


def test_rule_filter_keeps_legitimate_pupa_cluster():
    """A 5x-area cluster that isn't on the border and isn't bright should
    still be labeled 'cluster' so the fallback estimator can count it."""
    frame = pd.DataFrame(
        [
            *(_row(component_id="pupa_%02d" % i) for i in range(20)),
            _row(
                component_id="real_cluster",
                area_px=85.0 * 5,
                major_axis_px=45.0,
                mean_v=110.0,
                solidity=0.7,
                extent=0.55,
            ),
        ]
    )
    labeled = rule_classify_components(frame, AppConfig())
    label_lookup = dict(zip(labeled["component_id"], labeled["label"]))
    assert label_lookup["real_cluster"] == "cluster"
