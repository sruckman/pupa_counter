"""Render a three-color diagnostic overlay for a single image.

Given a fresh-run directory and the teacher instances table, draw:

* GREEN  — v1/v0 predictions that matched a teacher instance
* YELLOW — v1/v0 predictions that did NOT match (false positives)
* RED    — teacher instances that were NOT matched (real misses)
* BLUE   — cellpose_split teacher misses (likely benign, second half of a
           teacher-forced pair)

Use this to visually confirm whether remaining teacher_only rows are in
merged dense clusters, in separate regions the detector never enters, or on
cellpose-split pairs we don't need to chase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pupa_counter_fresh.eval_instances import (  # noqa: E402
    MatchConfig,
    canonical_scan_number,
    evaluate_disagreement,
    load_teacher_instances,
)


def render_diagnostic(
    rgb_native: np.ndarray,
    matched_pred: pd.DataFrame,
    pred_only: pd.DataFrame,
    teacher_missed_real: pd.DataFrame,
    teacher_missed_split: pd.DataFrame,
    image_id: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_native, cv2.COLOR_RGB2BGR).copy()

    def _draw(frame: pd.DataFrame, color, radius: int, thickness: int) -> None:
        for _, row in frame.iterrows():
            cx = int(round(float(row["centroid_x"])))
            cy = int(round(float(row["centroid_y"])))
            cv2.circle(bgr, (cx, cy), radius, color, thickness, cv2.LINE_AA)

    # Matched predictions → GREEN
    _draw(matched_pred, (0, 200, 0), 14, 2)
    # Unmatched predictions (false positives) → YELLOW
    _draw(pred_only, (0, 220, 255), 16, 3)
    # Real missed teacher → RED
    _draw(teacher_missed_real, (0, 0, 255), 18, 3)
    # cellpose_split missed teacher → BLUE
    _draw(teacher_missed_split, (255, 120, 0), 16, 2)

    header = (
        f"{image_id}  matched={len(matched_pred)}  FP={len(pred_only)}"
        f"  real_miss={len(teacher_missed_real)}  split_miss={len(teacher_missed_split)}"
    )
    cv2.putText(bgr, header, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(bgr, header, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    legend = "GREEN=match  YELLOW=FP  RED=real_miss  BLUE=split_miss"
    cv2.putText(bgr, legend, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, legend, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def main() -> int:
    parser = argparse.ArgumentParser(description="Render diagnostic overlays for a fresh run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-instances",
        type=Path,
        default=Path(
            "/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20_instances.csv"
        ),
    )
    parser.add_argument("--image-dir", type=Path, default=Path(
        "/Users/stephenyu/Documents/New project/data/probe_inputs/all_20"
    ))
    parser.add_argument("--only", nargs="*", help="Canonical keys to render, e.g. scan_20260313_25")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        parser.error(f"run-dir not found: {run_dir}")

    instances = pd.read_csv(run_dir / "instances.csv")
    if instances.empty:
        parser.error("instances.csv is empty")

    teacher = load_teacher_instances(args.teacher_instances)
    teacher["is_split"] = teacher["component_id"].astype(str).str.contains("_split")

    summary_df, matches_df, teacher_only_df, pred_only_df = evaluate_disagreement(
        instances, teacher, cfg=MatchConfig()
    )
    teacher_only_df["is_split"] = teacher_only_df["component_id"].astype(str).str.contains("_split")

    out_dir = run_dir / "diagnostic_overlays"
    out_dir.mkdir(exist_ok=True)

    counts_df = pd.read_csv(run_dir / "counts.csv")
    counts_df["key"] = counts_df["image_id"].map(canonical_scan_number)

    keys = sorted(counts_df["key"].dropna().unique())
    if args.only:
        keys = [k for k in keys if k in args.only]

    for key in keys:
        match_row = counts_df.loc[counts_df["key"] == key].iloc[0]
        image_path = Path(match_row["source_path"])
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  ! skip {key} — cannot read {image_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        inst_img = instances.loc[instances["image_id"].map(canonical_scan_number) == key]
        matches_img = matches_df.loc[matches_df.get("match_key") == key] if not matches_df.empty else pd.DataFrame()

        if not matches_img.empty and not inst_img.empty:
            matched_idx = set(matches_img["pred_index"].tolist())
            inst_img_reset = inst_img.reset_index(drop=True)
            matched_pred = inst_img_reset.iloc[sorted(matched_idx)] if matched_idx else pd.DataFrame()
        else:
            matched_pred = pd.DataFrame()

        pred_only_img = pred_only_df.loc[pred_only_df.get("match_key") == key] if not pred_only_df.empty else pd.DataFrame()
        teacher_only_img = teacher_only_df.loc[teacher_only_df.get("match_key") == key] if not teacher_only_df.empty else pd.DataFrame()
        real_miss = teacher_only_img.loc[~teacher_only_img["is_split"]]
        split_miss = teacher_only_img.loc[teacher_only_img["is_split"]]

        overlay = render_diagnostic(rgb, matched_pred, pred_only_img, real_miss, split_miss, key)
        out_path = out_dir / f"{key}_diagnostic.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"  . {key}  matched={len(matched_pred)}  fp={len(pred_only_img)}  real_miss={len(real_miss)}  split_miss={len(split_miss)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
