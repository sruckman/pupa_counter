"""Unit tests for the fresh-start teacher matcher.

These tests exist so the autonomous sweep loop can trust the metrics it
reads from ``meta.json`` — if matching itself drifts, every downstream
comparison is invalid.
"""

from __future__ import annotations

import pandas as pd

from pupa_counter_fresh.eval_instances import (
    MatchConfig,
    canonical_scan_number,
    evaluate_disagreement,
    match_one_image,
)


def _pred_row(image_id: str, cx: float, cy: float, score: float = 0.5) -> dict:
    return {
        "image_id": image_id,
        "centroid_x": cx,
        "centroid_y": cy,
        "score": score,
        "major_axis_px": 36.0,
        "minor_axis_px": 18.0,
        "bbox_x0": cx - 18,
        "bbox_y0": cy - 9,
        "bbox_x1": cx + 18,
        "bbox_y1": cy + 9,
    }


def _teacher_row(image_id: str, cx: float, cy: float) -> dict:
    return {
        "image_id": image_id,
        "component_id": f"cp_{int(cx)}_{int(cy)}",
        "centroid_x": cx,
        "centroid_y": cy,
        "major_axis_px": 36.0,
        "minor_axis_px": 18.0,
        "bbox_x0": cx - 18,
        "bbox_y0": cy - 9,
        "bbox_x1": cx + 18,
        "bbox_y1": cy + 9,
        "confidence": 0.8,
    }


def test_canonical_scan_number_handles_filenames_and_ids():
    assert canonical_scan_number("Scan_20260313 (25).png") == "scan_20260313_25"
    assert canonical_scan_number("scan_20260313_25_db0b949a97") == "scan_20260313_25"
    assert canonical_scan_number("/tmp/Scan_20260313_25.png") == "scan_20260313_25"
    assert canonical_scan_number(None) == ""


def test_perfect_match_five_pupae():
    preds = pd.DataFrame(
        [_pred_row("img", cx, cy) for cx, cy in [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200)]]
    )
    teachers = pd.DataFrame(
        [_teacher_row("img", cx, cy) for cx, cy in [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200)]]
    )
    matches, t_only, p_only = match_one_image(preds, teachers, cfg=MatchConfig())
    assert len(matches) == 5
    assert len(t_only) == 0
    assert len(p_only) == 0
    assert (matches["centroid_distance_px"] < 1e-6).all()


def test_one_teacher_missed():
    preds = pd.DataFrame(
        [_pred_row("img", cx, cy) for cx, cy in [(100, 100), (200, 100), (300, 100), (100, 200)]]
    )
    teachers = pd.DataFrame(
        [_teacher_row("img", cx, cy) for cx, cy in [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200)]]
    )
    matches, t_only, p_only = match_one_image(preds, teachers, cfg=MatchConfig())
    assert len(matches) == 4
    assert len(t_only) == 1
    assert len(p_only) == 0
    assert int(t_only.iloc[0]["centroid_x"]) == 200
    assert int(t_only.iloc[0]["centroid_y"]) == 200


def test_one_false_positive():
    preds = pd.DataFrame(
        [_pred_row("img", cx, cy) for cx, cy in [(100, 100), (200, 100), (300, 100), (100, 200), (500, 500)]]
    )
    teachers = pd.DataFrame(
        [_teacher_row("img", cx, cy) for cx, cy in [(100, 100), (200, 100), (300, 100), (100, 200)]]
    )
    matches, t_only, p_only = match_one_image(preds, teachers, cfg=MatchConfig())
    assert len(matches) == 4
    assert len(t_only) == 0
    assert len(p_only) == 1
    assert int(p_only.iloc[0]["centroid_x"]) == 500


def test_close_distance_matches_within_tolerance():
    """A prediction 20 px off a teacher should still match — the default
    tolerance is 32 px centroid distance."""
    preds = pd.DataFrame([_pred_row("img", 120.0, 100.0)])
    teachers = pd.DataFrame([_teacher_row("img", 100.0, 100.0)])
    matches, _, _ = match_one_image(preds, teachers, cfg=MatchConfig())
    assert len(matches) == 1
    assert abs(float(matches.iloc[0]["centroid_distance_px"]) - 20.0) < 1e-6


def test_evaluate_disagreement_cross_image_join():
    """The top-level evaluator must join on canonical scan number, not the
    raw image_id column."""
    preds = pd.DataFrame(
        [_pred_row("Scan_20260313 (25).png", 100, 100)]
    )
    teachers = pd.DataFrame(
        [_teacher_row("scan_20260313_25_abc123", 100, 100)]
    )
    summary, matches, t_only, p_only = evaluate_disagreement(preds, teachers, cfg=MatchConfig())
    assert len(summary) == 1
    assert int(summary.iloc[0]["matched"]) == 1
    assert int(summary.iloc[0]["teacher_only"]) == 0
    assert int(summary.iloc[0]["pred_only"]) == 0
