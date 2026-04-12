"""Build a zoomed crop gallery of v1 disagreements for human-style visual review.

For a given run directory + image_id, produce two gallery PNGs:

* ``<key>_pred_only.png`` — one zoomed crop per v1 prediction that had no
  teacher match (candidates for "v8 missed a real pupa" vs "v1 fired on
  dust/pen/scan artifact")
* ``<key>_teacher_only.png`` — one zoomed crop per v8 teacher centroid that
  v1 did not match (candidates for "real touching-pupae miss" vs
  "cellpose_split artifact" vs "teacher error")

Each crop is a 200×200 native-resolution patch centered on the centroid,
upscaled 2× to 400×400 for clarity, with a small red ring at the centroid
and a caption showing index + coordinates.

Usage::

    python scripts/fresh_disagreement_gallery.py \\
        --run-dir /path/to/fresh_peak_v1_final \\
        --image-key scan_20260313_15
"""

from __future__ import annotations

import argparse
import math
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


CROP_HALF = 100
ZOOM = 2
LABEL_HEIGHT = 40
GRID_COLS = 5


def _clip(cx: int, cy: int, half: int, h: int, w: int) -> tuple[int, int, int, int]:
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    return x0, y0, x1, y1


def _extract_tile(
    rgb: np.ndarray,
    cx: float,
    cy: float,
    label: str,
    color: tuple[int, int, int],
    extras: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    x0, y0, x1, y1 = _clip(cx_i, cy_i, CROP_HALF, h, w)
    patch = rgb[y0:y1, x0:x1].copy()

    # Pad to square CROP_HALF*2 if the centroid was near a border
    pad_top = max(0, CROP_HALF - (cy_i - y0))
    pad_left = max(0, CROP_HALF - (cx_i - x0))
    pad_bottom = max(0, (2 * CROP_HALF) - patch.shape[0] - pad_top)
    pad_right = max(0, (2 * CROP_HALF) - patch.shape[1] - pad_left)
    if pad_top or pad_left or pad_bottom or pad_right:
        patch = cv2.copyMakeBorder(
            patch, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(230, 230, 230),
        )

    bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    big = cv2.resize(bgr, (patch.shape[1] * ZOOM, patch.shape[0] * ZOOM), interpolation=cv2.INTER_CUBIC)

    center = (big.shape[1] // 2, big.shape[0] // 2)
    cv2.circle(big, center, 22, color, 3, cv2.LINE_AA)
    cv2.circle(big, center, 3, color, -1, cv2.LINE_AA)

    # Draw any nearby predictions/teacher markers (extras) within the crop
    if extras:
        for ex, ey in extras:
            local_x = int(round((ex - cx_i + CROP_HALF + pad_left) * ZOOM))
            local_y = int(round((ey - cy_i + CROP_HALF + pad_top) * ZOOM))
            if 0 <= local_x < big.shape[1] and 0 <= local_y < big.shape[0]:
                cv2.circle(big, (local_x, local_y), 12, (0, 200, 0), 2, cv2.LINE_AA)

    caption = np.ones((LABEL_HEIGHT, big.shape[1], 3), dtype=np.uint8) * 245
    cv2.putText(
        caption, label, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1, cv2.LINE_AA,
    )
    tile = np.vstack([big, caption])
    cv2.rectangle(
        tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
        (120, 120, 120), 1,
    )
    return tile


def _mosaic(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    if not tiles:
        return np.ones((100, 100, 3), dtype=np.uint8) * 245
    rows = math.ceil(len(tiles) / cols)
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.ones((rows * tile_h + 10, cols * tile_w + 10, 3), dtype=np.uint8) * 245
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        y = 5 + r * tile_h
        x = 5 + c * tile_w
        canvas[y : y + tile_h, x : x + tile_w] = tile
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Disagreement crop gallery builder.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-key", type=str, required=True, help="e.g. scan_20260313_25")
    parser.add_argument(
        "--teacher-instances",
        type=Path,
        default=Path(
            "/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20_instances.csv"
        ),
    )
    parser.add_argument("--cols", type=int, default=GRID_COLS)
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f"run-dir not found: {args.run_dir}")

    instances = pd.read_csv(args.run_dir / "instances.csv")
    teacher = load_teacher_instances(args.teacher_instances)

    instances["key"] = instances["image_id"].map(canonical_scan_number)
    teacher["key"] = teacher["image_id"].map(canonical_scan_number)
    teacher["is_split"] = teacher["component_id"].astype(str).str.contains("_split")

    key = args.image_key
    inst_img = instances.loc[instances["key"] == key].reset_index(drop=True)
    teacher_img = teacher.loc[teacher["key"] == key].reset_index(drop=True)

    if inst_img.empty:
        parser.error(f"No predictions for {key}")

    counts_df = pd.read_csv(args.run_dir / "counts.csv")
    counts_df["key"] = counts_df["image_id"].map(canonical_scan_number)
    row = counts_df.loc[counts_df["key"] == key]
    if row.empty:
        parser.error(f"counts.csv has no row for {key}")
    image_path = Path(row.iloc[0]["source_path"])
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"Cannot read {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Match predictions to teacher for this image only
    summary_df, matches_df, teacher_only_df, pred_only_df = evaluate_disagreement(
        inst_img, teacher_img, cfg=MatchConfig()
    )

    pred_coords = list(zip(inst_img["centroid_x"].tolist(), inst_img["centroid_y"].tolist()))
    teacher_coords = list(zip(teacher_img["centroid_x"].tolist(), teacher_img["centroid_y"].tolist()))

    out_dir = args.run_dir / "disagreement_galleries"
    out_dir.mkdir(exist_ok=True)

    # Gallery 1: pred_only
    pred_tiles: list[np.ndarray] = []
    if not pred_only_df.empty:
        # Sort by centroid_y,centroid_x for stable ordering
        po = pred_only_df.sort_values(["centroid_y", "centroid_x"]).reset_index(drop=True)
        for idx, rec in po.iterrows():
            cx = float(rec["centroid_x"])
            cy = float(rec["centroid_y"])
            label = f"[pred_only #{idx+1}]  xy=({cx:.0f}, {cy:.0f})  score={float(rec.get('score', 0)):.2f}"
            nearby = [
                (tx, ty) for tx, ty in teacher_coords
                if abs(tx - cx) < CROP_HALF and abs(ty - cy) < CROP_HALF
            ]
            tile = _extract_tile(rgb, cx, cy, label, color=(0, 200, 255), extras=nearby)
            pred_tiles.append(tile)
    gallery_p = _mosaic(pred_tiles, args.cols)
    pred_out = out_dir / f"{key}_pred_only_gallery.png"
    cv2.imwrite(str(pred_out), gallery_p)

    # Gallery 2: teacher_only (split cellpose_split out; flag them separately)
    teacher_tiles: list[np.ndarray] = []
    if not teacher_only_df.empty:
        to = teacher_only_df.copy()
        to["is_split"] = to["component_id"].astype(str).str.contains("_split")
        to = to.sort_values(["is_split", "centroid_y", "centroid_x"]).reset_index(drop=True)
        for idx, rec in to.iterrows():
            cx = float(rec["centroid_x"])
            cy = float(rec["centroid_y"])
            kind = "split_miss" if bool(rec["is_split"]) else "real_miss"
            color = (200, 120, 0) if kind == "split_miss" else (0, 0, 220)
            label = f"[{kind} #{idx+1}]  xy=({cx:.0f}, {cy:.0f})"
            nearby_preds = [
                (px, py) for px, py in pred_coords
                if abs(px - cx) < CROP_HALF and abs(py - cy) < CROP_HALF
            ]
            tile = _extract_tile(rgb, cx, cy, label, color=color, extras=nearby_preds)
            teacher_tiles.append(tile)
    gallery_t = _mosaic(teacher_tiles, args.cols)
    teacher_out = out_dir / f"{key}_teacher_only_gallery.png"
    cv2.imwrite(str(teacher_out), gallery_t)

    print(f"{key}:  pred_only={len(pred_tiles)}  teacher_only={len(teacher_tiles)}")
    print(f"  pred gallery    : {pred_out}")
    print(f"  teacher gallery : {teacher_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
