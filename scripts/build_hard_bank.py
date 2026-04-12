"""Build a hard-component bank + single-pupa bank for targeted experiments.

Exports crops, masks, response maps, and metadata for:
1. Hard components (area_ratio > 1.15, rescue-fired, border-touching, etc.)
2. Clean single pupae from exact scans (for prototype prior)

The user then labels each hard crop with gt_k and approximate center clicks.
All future experiments test against this bank first, not full scans.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pupa_counter_fresh.preprocess import load_image_rgb, downscale, build_blue_mask
from pupa_counter_fresh.response import compute_response_map, build_allowed_mask

SCANS = ['3', '7', '8', '10', '15', '20', '22', '25', '30', '35']
SCALE = 0.67
SINGLE_PUPA_AREA = 200.0
PAD_WORK = 8


def main() -> int:
    out_dir = REPO_ROOT / "tmp" / "hard_bank"
    hard_dir = out_dir / "hard"
    single_dir = out_dir / "single"
    hard_dir.mkdir(parents=True, exist_ok=True)
    single_dir.mkdir(parents=True, exist_ok=True)

    sc_df = pd.read_csv(REPO_ROOT / "tmp" / "seed_completion_bench" / "sc_instances.csv")

    hard_rows = []
    single_rows = []
    hard_idx = 0
    single_idx = 0

    for scan_num in SCANS:
        scan_id = f"scan_20260313_{scan_num}"
        src = f"/Users/stephenyu/Documents/New project/data/probe_inputs/all_20/Scan_20260313 ({scan_num}).png"
        rgb_native = load_image_rgb(src)
        rgb_work, scale = downscale(rgb_native, SCALE)
        blue = build_blue_mask(rgb_work)
        resp = compute_response_map(
            rgb_work, blue, response_mode="adaptive",
            adaptive_small_sigma=0.6, adaptive_large_sigma=0.6,
            adaptive_area_threshold_px=500,
        )
        mask = build_allowed_mask(resp, abs_threshold=0.12, min_percentile=0.0)
        labels, n = ndi.label(mask > 0)
        sizes = np.bincount(labels.ravel())
        slices = ndi.find_objects(labels)

        sc_sub = sc_df[sc_df["image_id"] == scan_id].reset_index(drop=True)

        for c in range(1, n + 1):
            area = int(sizes[c])
            if area < 60:
                continue
            sl = slices[c - 1]
            if sl is None:
                continue

            cm = labels[sl] == c
            cr = resp[sl]
            rp = regionprops(cm.astype(np.uint8))
            if not rp:
                continue
            r = rp[0]
            sol = float(r.solidity) if r.solidity is not None else 1.0
            ext = float(r.extent) if r.extent is not None else 1.0
            major = float(getattr(r, "axis_major_length", getattr(r, "major_axis_length", 0)) or 0)
            minor = max(1.0, float(getattr(r, "axis_minor_length", getattr(r, "minor_axis_length", 0)) or 0))

            area_ratio = area / SINGLE_PUPA_AREA
            area_k = max(1, round(area_ratio))
            touches_border = (
                sl[0].start == 0 or sl[1].start == 0
                or sl[0].stop == resp.shape[0] or sl[1].stop == resp.shape[1]
            )

            # Count sc detections in this component
            n_det = 0
            det_types = set()
            for _, dr in sc_sub.iterrows():
                xw = int(round(float(dr["centroid_x"]) * SCALE))
                yw = int(round(float(dr["centroid_y"]) * SCALE))
                if 0 <= yw < labels.shape[0] and 0 <= xw < labels.shape[1]:
                    if labels[yw, xw] == c:
                        n_det += 1
                        det_types.add(dr.get("resolver_type", "unknown"))

            # Classify: hard or single?
            is_hard = (
                area_ratio >= 1.15
                or "seed_completion" in det_types
                or touches_border
                or sol < 0.92
                or n_det != area_k
            )

            is_clean_single = (
                not is_hard
                and area_ratio >= 0.8
                and area_ratio <= 1.25
                and sol >= 0.93
                and n_det == 1
                and scan_num in ['7', '8', '25', '30']  # exact scans
            )

            if not is_hard and not is_clean_single:
                continue

            # Crop native + work images
            y0w, y1w = sl[0].start, sl[0].stop
            x0w, x1w = sl[1].start, sl[1].stop
            inv = 1.0 / SCALE
            y0n = max(0, int((y0w - PAD_WORK) * inv))
            y1n = min(rgb_native.shape[0], int((y1w + PAD_WORK) * inv))
            x0n = max(0, int((x0w - PAD_WORK) * inv))
            x1n = min(rgb_native.shape[1], int((x1w + PAD_WORK) * inv))

            meta = {
                "image_id": scan_id,
                "comp_id": int(c),
                "area": area,
                "area_ratio": round(area_ratio, 2),
                "area_k": area_k,
                "n_det": n_det,
                "det_types": sorted(det_types),
                "solidity": round(sol, 3),
                "extent": round(ext, 3),
                "major": round(major, 1),
                "minor": round(minor, 1),
                "touches_border": touches_border,
                "bbox_work": [y0w, x0w, y1w, x1w],
                "bbox_native": [y0n, x0n, y1n, x1n],
            }

            if is_hard:
                tag = f"hard_{hard_idx:04d}"
                d = hard_dir / tag
                d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / "native_crop.png"), cv2.cvtColor(rgb_native[y0n:y1n, x0n:x1n], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(d / "work_crop.png"), cv2.cvtColor(rgb_work[max(0,y0w-PAD_WORK):y1w+PAD_WORK, max(0,x0w-PAD_WORK):x1w+PAD_WORK], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(d / "comp_mask.png"), cm.astype(np.uint8) * 255)
                resp_vis = np.clip(cr * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(str(d / "response_map.png"), cv2.applyColorMap(resp_vis, cv2.COLORMAP_INFERNO))

                # Overlay with detections
                overlay = cv2.cvtColor(rgb_work[max(0,y0w-PAD_WORK):y1w+PAD_WORK, max(0,x0w-PAD_WORK):x1w+PAD_WORK], cv2.COLOR_RGB2BGR).copy()
                for _, dr in sc_sub.iterrows():
                    xw = int(round(float(dr["centroid_x"]) * SCALE))
                    yw = int(round(float(dr["centroid_y"]) * SCALE))
                    if max(0,y0w-PAD_WORK) <= yw < y1w+PAD_WORK and max(0,x0w-PAD_WORK) <= xw < x1w+PAD_WORK:
                        color = (0,0,255) if dr.get("resolver_type") == "seed_completion" else (0,200,255)
                        cv2.circle(overlay, (xw-max(0,x0w-PAD_WORK), yw-max(0,y0w-PAD_WORK)), 5, color, 1, cv2.LINE_AA)
                cv2.imwrite(str(d / "overlay.png"), overlay)

                with (d / "meta.json").open("w") as f:
                    json.dump(meta, f, indent=2)
                hard_rows.append(meta)
                hard_idx += 1
            elif is_clean_single:
                tag = f"single_{single_idx:04d}"
                d = single_dir / tag
                d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / "native_crop.png"), cv2.cvtColor(rgb_native[y0n:y1n, x0n:x1n], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(d / "comp_mask.png"), cm.astype(np.uint8) * 255)
                with (d / "meta.json").open("w") as f:
                    json.dump(meta, f, indent=2)
                single_rows.append(meta)
                single_idx += 1

    print(f"Hard components: {hard_idx}")
    print(f"Single pupae: {single_idx}")
    print(f"Output: {out_dir}")

    # Summary CSV
    pd.DataFrame(hard_rows).to_csv(out_dir / "hard_summary.csv", index=False)
    pd.DataFrame(single_rows).to_csv(out_dir / "single_summary.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
