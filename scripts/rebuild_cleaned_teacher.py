"""Build a ``teacher_v8_20_instances_cleaned.csv`` that drops v8's own
scanner-edge false positives.

The 2026-04-10 visual audit found that teacher_v8_20_instances.csv
contains a handful of rows whose centroids sit directly on the scanner
gray-strip at the left edge of the image (confirmed on scans 15, 20, 25).
Those labels are not real pupae — they are the same failure mode v1
suffers from, just stored in the teacher.

This script:

1. Loads every image referenced by the teacher table (via
   canonical_scan_number key).
2. Computes a paper ROI mask using the same algorithm as the online
   detector (``pupa_counter_fresh.paper_roi.detect_paper_roi``).
3. For each teacher row, maps its centroid into the paper_mask
   coordinate space (the mask is computed at work_scale, not native)
   and drops the row if the centroid is outside the mask.
4. Writes the cleaned frame next to the original with
   ``_cleaned.csv`` suffix, plus a ``cleanup_summary.csv`` showing
   per-image drop counts.

The cleaned teacher is an OFFLINE artifact. The online detector never
touches v8. But every subsequent v1/v2 audit can use the cleaned teacher
as a more honest ground truth.
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

from pupa_counter_fresh.eval_instances import canonical_scan_number  # noqa: E402
from pupa_counter_fresh.paper_roi import PaperROIConfig, detect_paper_roi  # noqa: E402


DEFAULT_TEACHER_INSTANCES = Path(
    "/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20_instances.csv"
)
DEFAULT_IMAGE_DIR = Path("/Users/stephenyu/Documents/New project/data/probe_inputs/all_20")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean v8 teacher via paper ROI.")
    parser.add_argument("--teacher-instances", type=Path, default=DEFAULT_TEACHER_INSTANCES)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the cleaned teacher CSV. Defaults to "
        "<teacher_instances>.parent / <stem>_cleaned.csv",
    )
    parser.add_argument("--brightness-threshold", type=int, default=180)
    parser.add_argument("--close-kernel", type=int, default=15)
    parser.add_argument("--erode-margin", type=int, default=3)
    parser.add_argument("--min-paper-fraction", type=float, default=0.05)
    args = parser.parse_args()

    teacher = pd.read_csv(args.teacher_instances)
    print(f"teacher rows: {len(teacher)}")

    if args.out is None:
        out_path = args.teacher_instances.with_name(
            f"{args.teacher_instances.stem}_cleaned.csv"
        )
    else:
        out_path = args.out

    # Build a canonical-scan-number -> image path lookup
    image_paths = sorted(args.image_dir.glob("*.png"))
    key_to_path: dict[str, Path] = {}
    for p in image_paths:
        k = canonical_scan_number(p.stem)
        if k:
            key_to_path[k] = p
    print(f"image lookup built for {len(key_to_path)} images")

    teacher["_key"] = teacher["image_id"].map(canonical_scan_number)
    keep_mask = pd.Series(True, index=teacher.index)
    per_image: list[dict] = []

    roi_cfg = PaperROIConfig(
        brightness_threshold=int(args.brightness_threshold),
        close_kernel_px=int(args.close_kernel),
        erode_margin_px=int(args.erode_margin),
        min_paper_fraction=float(args.min_paper_fraction),
    )

    for key, sub in teacher.groupby("_key"):
        image_path = key_to_path.get(str(key))
        if image_path is None:
            print(f"  ! {key}: no image found, keeping all {len(sub)} rows")
            per_image.append(
                {"image_key": key, "teacher_before": len(sub), "teacher_after": len(sub),
                 "dropped": 0, "reason": "no_image"}
            )
            continue

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  ! {key}: cv2.imread failed, keeping all {len(sub)} rows")
            per_image.append(
                {"image_key": key, "teacher_before": len(sub), "teacher_after": len(sub),
                 "dropped": 0, "reason": "imread_failed"}
            )
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        paper_mask = detect_paper_roi(rgb, cfg=roi_cfg)
        if paper_mask is None:
            print(f"  ! {key}: paper_roi returned None, keeping all {len(sub)} rows")
            per_image.append(
                {"image_key": key, "teacher_before": len(sub), "teacher_after": len(sub),
                 "dropped": 0, "reason": "roi_none"}
            )
            continue

        h, w = paper_mask.shape
        xs = pd.to_numeric(sub["centroid_x"], errors="coerce").fillna(-1).to_numpy()
        ys = pd.to_numeric(sub["centroid_y"], errors="coerce").fillna(-1).to_numpy()
        xs_i = np.clip(xs.astype(int), 0, w - 1)
        ys_i = np.clip(ys.astype(int), 0, h - 1)
        inside = paper_mask[ys_i, xs_i] > 0
        # Rows with out-of-bounds or invalid coords also get dropped
        valid_coords = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        inside &= valid_coords
        keep_mask.loc[sub.index] = inside

        dropped = int((~inside).sum())
        before = len(sub)
        per_image.append(
            {
                "image_key": key,
                "teacher_before": before,
                "teacher_after": before - dropped,
                "dropped": dropped,
                "reason": "cleaned" if dropped > 0 else "no_drop_needed",
            }
        )
        marker = "  -" if dropped > 0 else "   "
        print(f"{marker} {key}: {before:3d} -> {before - dropped:3d}  (dropped {dropped})")

    cleaned = teacher.loc[keep_mask].drop(columns="_key").copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print()
    print(f"wrote {out_path}  ({len(cleaned)} rows)")
    print(f"dropped {len(teacher) - len(cleaned)} teacher rows in total")

    summary_df = pd.DataFrame(per_image).sort_values("dropped", ascending=False)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
