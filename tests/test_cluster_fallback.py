from __future__ import annotations

import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.cluster_fallback import attach_cluster_count_estimates, synthesize_cluster_instances


def test_cluster_fallback_estimates_and_synthesizes_instances():
    frame = pd.DataFrame(
        [
            {
                "component_id": "p1",
                "is_active": True,
                "label": "pupa",
                "confidence": 0.8,
                "area_px": 100.0,
                "major_axis_px": 20.0,
                "cluster_unresolved": False,
                "centroid_x": 10.0,
                "centroid_y": 10.0,
                "bbox_x0": 0,
                "bbox_y0": 0,
                "bbox_x1": 20,
                "bbox_y1": 20,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
            },
            {
                "component_id": "c1",
                "is_active": True,
                "label": "cluster",
                "confidence": 0.5,
                "area_px": 320.0,
                "major_axis_px": 55.0,
                "cluster_unresolved": True,
                "centroid_x": 50.0,
                "centroid_y": 50.0,
                "bbox_x0": 30,
                "bbox_y0": 30,
                "bbox_x1": 80,
                "bbox_y1": 80,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
            },
        ]
    )
    estimated = attach_cluster_count_estimates(frame, AppConfig())
    cluster_row = estimated.loc[estimated["component_id"] == "c1"].iloc[0]
    assert int(cluster_row["estimated_cluster_count"]) >= 2

    synthetic = synthesize_cluster_instances(estimated, AppConfig())
    assert not synthetic.empty
    assert synthetic["synthetic_instance"].all()


def test_cluster_fallback_excludes_split_children_and_small_fragments():
    cfg = AppConfig()
    frame = pd.DataFrame(
        [
            {
                "component_id": "p1",
                "is_active": True,
                "label": "pupa",
                "confidence": 0.9,
                "area_px": 100.0,
                "major_axis_px": 20.0,
                "cluster_unresolved": False,
                "split_from_cluster": False,
                "centroid_x": 10.0,
                "centroid_y": 10.0,
                "bbox_x0": 0,
                "bbox_y0": 0,
                "bbox_x1": 20,
                "bbox_y1": 20,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
            },
            {
                "component_id": "parent_cluster",
                "is_active": True,
                "label": "cluster",
                "confidence": 0.7,
                "area_px": 320.0,
                "major_axis_px": 45.0,
                "cluster_unresolved": True,
                "split_from_cluster": False,
                "centroid_x": 50.0,
                "centroid_y": 50.0,
                "bbox_x0": 30,
                "bbox_y0": 30,
                "bbox_x1": 80,
                "bbox_y1": 80,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
            },
            {
                "component_id": "child_fragment",
                "is_active": True,
                "label": "cluster",
                "confidence": 0.8,
                "area_px": 260.0,
                "major_axis_px": 30.0,
                "cluster_unresolved": True,
                "split_from_cluster": True,
                "centroid_x": 70.0,
                "centroid_y": 70.0,
                "bbox_x0": 55,
                "bbox_y0": 55,
                "bbox_x1": 95,
                "bbox_y1": 95,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
            },
        ]
    )
    estimated = attach_cluster_count_estimates(frame, cfg)

    parent = estimated.loc[estimated["component_id"] == "parent_cluster"].iloc[0]
    child = estimated.loc[estimated["component_id"] == "child_fragment"].iloc[0]

    assert bool(parent["cluster_fallback_eligible"])
    assert int(parent["estimated_cluster_count"]) >= 2
    assert not bool(child["cluster_fallback_eligible"])
    assert int(child["estimated_cluster_count"]) == 0


def test_synthesize_cluster_instances_uses_vision_even_when_local_fallback_disabled():
    cfg = AppConfig()
    cfg.cluster_fallback.use_for_counts = False
    frame = pd.DataFrame(
        [
            {
                "component_id": "vision_cluster",
                "is_active": True,
                "label": "cluster",
                "confidence": 0.7,
                "area_px": 400.0,
                "major_axis_px": 50.0,
                "cluster_unresolved": True,
                "cluster_fallback_eligible": True,
                "estimated_cluster_count": 3,
                "cluster_count_source": "vision",
                "centroid_x": 50.0,
                "centroid_y": 50.0,
                "bbox_x0": 30,
                "bbox_y0": 30,
                "bbox_x1": 80,
                "bbox_y1": 80,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
            }
        ]
    )
    synthetic = synthesize_cluster_instances(frame, cfg)
    assert len(synthetic) == 3
