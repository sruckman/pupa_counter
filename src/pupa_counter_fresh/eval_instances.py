"""Teacher-based instance matching for the fresh peak-first detector.

This is the audit harness the handoff insists on: "No detector changes before
this harness exists." It is intentionally standalone — the only input beyond
the prediction frame is ``teacher_v8_20_instances.csv``.

Matching strategy
-----------------

* Greedy assignment by centroid distance — ``dist <= max_centroid_px`` or
  ``dist <= major_axis_scale * teacher_major_axis`` whichever is larger.
* Bbox IoU is only used as a tiebreaker for teacher instances the peak
  proposer can't provide bboxes for; peaks-only runs fall back to distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical image id helpers
# ---------------------------------------------------------------------------


_SCAN_NUMBER_RE = re.compile(r"scan[_\s]*20260313[_\s]*\(?(\d+)\)?", re.IGNORECASE)


def canonical_scan_number(value: str | Path | None) -> str:
    """Return the ``scan_20260313_<n>`` prefix of a filename/id if present."""
    if value is None:
        return ""
    stem = Path(str(value)).stem.lower().strip()
    match = _SCAN_NUMBER_RE.search(stem)
    if match:
        return f"scan_20260313_{int(match.group(1))}"
    digits = re.findall(r"\d+", stem)
    if digits:
        return f"scan_20260313_{int(digits[0])}"
    return stem


def build_teacher_image_key(teacher_df: pd.DataFrame) -> pd.Series:
    """Return a Series of canonical scan numbers for each teacher row."""
    return teacher_df["image_id"].map(canonical_scan_number)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class MatchConfig:
    max_centroid_px: float = 32.0
    major_axis_scale: float = 0.70
    min_bbox_iou: float = 0.05


def _bbox_iou(pred: pd.Series, teacher: pd.Series) -> float:
    cols = ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
    if any(col not in pred.index or col not in teacher.index for col in cols):
        return 0.0
    try:
        px0, py0, px1, py1 = (float(pred[c]) for c in cols)
        tx0, ty0, tx1, ty1 = (float(teacher[c]) for c in cols)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite([px0, py0, px1, py1, tx0, ty0, tx1, ty1]).all():
        return 0.0
    ix0, iy0 = max(px0, tx0), max(py0, ty0)
    ix1, iy1 = min(px1, tx1), min(py1, ty1)
    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    pred_area = max(1.0, (px1 - px0) * (py1 - py0))
    teacher_area = max(1.0, (tx1 - tx0) * (ty1 - ty0))
    return inter / max(1.0, pred_area + teacher_area - inter)


def match_one_image(
    pred_df: pd.DataFrame,
    teacher_df: pd.DataFrame,
    *,
    cfg: MatchConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match predictions to teacher rows for a single image.

    Returns ``(matches, teacher_only, pred_only)`` as three DataFrames.
    """
    cfg = cfg or MatchConfig()
    empty = pd.DataFrame()
    if pred_df.empty and teacher_df.empty:
        return empty, empty, empty

    pred_frame = pred_df.reset_index(drop=True).copy()
    teacher_frame = teacher_df.reset_index(drop=True).copy()

    if pred_frame.empty:
        return empty, teacher_frame, empty
    if teacher_frame.empty:
        return empty, empty, pred_frame

    candidates: List[Tuple[float, int, int, float, float]] = []
    pred_x = pred_frame["centroid_x"].to_numpy(dtype=np.float64)
    pred_y = pred_frame["centroid_y"].to_numpy(dtype=np.float64)
    teacher_x = teacher_frame["centroid_x"].to_numpy(dtype=np.float64)
    teacher_y = teacher_frame["centroid_y"].to_numpy(dtype=np.float64)
    teacher_major = (
        teacher_frame["major_axis_px"].to_numpy(dtype=np.float64)
        if "major_axis_px" in teacher_frame.columns
        else np.full(len(teacher_frame), 36.0)
    )

    for pi in range(len(pred_frame)):
        for ti in range(len(teacher_frame)):
            dx = pred_x[pi] - teacher_x[ti]
            dy = pred_y[pi] - teacher_y[ti]
            dist = float(np.hypot(dx, dy))
            scale = max(12.0, float(teacher_major[ti]) if np.isfinite(teacher_major[ti]) else 36.0)
            iou = _bbox_iou(pred_frame.iloc[pi], teacher_frame.iloc[ti])
            allowed = (
                dist <= max(cfg.max_centroid_px, cfg.major_axis_scale * scale)
                or iou >= cfg.min_bbox_iou
            )
            if not allowed:
                continue
            cost = dist / max(scale, 1.0) - 0.25 * iou
            candidates.append((cost, pi, ti, dist, iou))

    candidates.sort(key=lambda item: item[0])

    used_pred: set[int] = set()
    used_teacher: set[int] = set()
    matches: list[dict] = []
    for cost, pi, ti, dist, iou in candidates:
        if pi in used_pred or ti in used_teacher:
            continue
        used_pred.add(pi)
        used_teacher.add(ti)
        matches.append(
            {
                "image_id": str(pred_frame.iloc[pi].get("image_id", "")),
                "pred_index": int(pi),
                "teacher_index": int(ti),
                "centroid_distance_px": dist,
                "bbox_iou": iou,
                "match_cost": cost,
                "pred_score": pred_frame.iloc[pi].get("score"),
                "teacher_confidence": teacher_frame.iloc[ti].get("confidence"),
            }
        )

    match_df = pd.DataFrame(matches)
    teacher_only = teacher_frame.loc[
        [i for i in range(len(teacher_frame)) if i not in used_teacher]
    ].copy()
    pred_only = pred_frame.loc[
        [i for i in range(len(pred_frame)) if i not in used_pred]
    ].copy()
    return match_df, teacher_only, pred_only


def evaluate_disagreement(
    pred_df: pd.DataFrame,
    teacher_df: pd.DataFrame,
    *,
    cfg: MatchConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match ``pred_df`` against ``teacher_df`` image-by-image.

    Returns ``(summary_df, matches_df, teacher_only_df, pred_only_df)``.

    ``image_id`` alignment is done via :func:`canonical_scan_number` so the
    pred frame can use any consistent naming (e.g. filename stem) and still
    join to the teacher table.
    """
    cfg = cfg or MatchConfig()

    pred = pred_df.copy()
    teacher = teacher_df.copy()
    pred["match_key"] = pred["image_id"].map(canonical_scan_number)
    teacher["match_key"] = build_teacher_image_key(teacher)

    keys = sorted(set(pred["match_key"]) | set(teacher["match_key"]))

    summaries: List[dict] = []
    all_matches: List[pd.DataFrame] = []
    all_teacher_only: List[pd.DataFrame] = []
    all_pred_only: List[pd.DataFrame] = []

    for key in keys:
        pred_img = pred.loc[pred["match_key"] == key].reset_index(drop=True)
        teacher_img = teacher.loc[teacher["match_key"] == key].reset_index(drop=True)
        match_df, t_only, p_only = match_one_image(pred_img, teacher_img, cfg=cfg)

        if not match_df.empty:
            match_df["match_key"] = key
            all_matches.append(match_df)
        if not t_only.empty:
            t_only = t_only.assign(disagreement_type="teacher_only", match_key=key)
            all_teacher_only.append(t_only)
        if not p_only.empty:
            p_only = p_only.assign(disagreement_type="pred_only", match_key=key)
            all_pred_only.append(p_only)

        summaries.append(
            {
                "match_key": key,
                "teacher_instances": int(len(teacher_img)),
                "pred_instances": int(len(pred_img)),
                "matched": int(len(match_df)),
                "teacher_only": int(len(t_only)),
                "pred_only": int(len(p_only)),
                "mean_centroid_distance_px": (
                    float(match_df["centroid_distance_px"].mean())
                    if not match_df.empty
                    else None
                ),
            }
        )

    summary_df = pd.DataFrame(summaries)
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    teacher_only_df = (
        pd.concat(all_teacher_only, ignore_index=True) if all_teacher_only else pd.DataFrame()
    )
    pred_only_df = (
        pd.concat(all_pred_only, ignore_index=True) if all_pred_only else pd.DataFrame()
    )
    return summary_df, matches_df, teacher_only_df, pred_only_df


def load_teacher_instances(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    numeric_cols = (
        "centroid_x",
        "centroid_y",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "major_axis_px",
        "minor_axis_px",
        "confidence",
    )
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame
