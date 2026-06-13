"""Benchmark the seed completion rescue vs v4 on hand-labeled scans.

Primary metric: **pupa-level** (FP + miss) using the information we actually
have in corrections.csv. For each scan we know:
* The exact FP indices in the iter5 detection ordering (from
  ``fp_numbers``), which we map to iter5 (x, y).
* The exact miss count (``miss_count``), but not positions.

From that we derive:
* v4 baseline: run the v4 detector, then for each known-FP iter5 position,
  test whether v4 has a detection within a matching radius of it. Count
  such detections as ``FP_kept``. FPs that v4 successfully dropped are
  ``FP_dropped``. v4's new detections that don't overlap any iter5
  detection are counted as ``new_detections`` (potential miss recoveries
  OR potential new FPs — by design we can't tell without position data).
* v4_sc (rescue): same metric on the rescue variant. A rescue win is
  a strict increase in ``new_detections`` and/or a strict decrease in
  ``FP_kept``, with bounded delta in the other direction.

This is still approximate — ``new_detections`` conflates "miss recovered"
with "new FP". But combined with image-level delta vs GT, it gives a
more honest picture than image-level alone.

We also report:
* ``sum|image_delta|`` vs GT_COUNTS (hand-label totals)
* ``seed_completion_fires`` — how many times the rescue pass actually
  bumped the returned peak count per scan
* ``removed`` / ``added`` peaks per scan (v4 vs v4_sc)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pupa_counter_fresh import DetectorConfig, run_detector  # noqa: E402

HAND_SCANS = [
    "scan_20260313_3",
    "scan_20260313_7",
    "scan_20260313_8",
    "scan_20260313_10",
    "scan_20260313_15",
    "scan_20260313_20",
    "scan_20260313_22",
    "scan_20260313_25",
    "scan_20260313_30",
    "scan_20260313_35",
]

GT_COUNTS = {
    "scan_20260313_3": 97,
    "scan_20260313_7": 106,
    "scan_20260313_8": 126,
    "scan_20260313_10": 98,
    "scan_20260313_15": 112,
    "scan_20260313_20": 116,
    "scan_20260313_22": 137,
    "scan_20260313_25": 128,
    "scan_20260313_30": 100,
    "scan_20260313_35": 113,
}

# Match tolerance in NATIVE pixels. A pupa is ~36x18 native px; 20 is
# comfortably within a single pupa.
MATCH_RADIUS_PX = 20.0


def _load_corrections() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "hand_labels" / "corrections.csv")


def _parse_fp_numbers(value) -> list[int]:
    if pd.isna(value) or value == "":
        return []
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [int(p) for p in parts if p.isdigit()]


def _match_positions(
    source: pd.DataFrame,
    targets: pd.DataFrame,
    radius: float,
) -> np.ndarray:
    """For each row in ``source`` return the nearest ``targets`` index (or -1)."""
    if source.empty or targets.empty:
        return np.full(len(source), -1, dtype=np.int64)
    sx = source["centroid_x"].to_numpy(dtype=np.float32)
    sy = source["centroid_y"].to_numpy(dtype=np.float32)
    tx = targets["centroid_x"].to_numpy(dtype=np.float32)
    ty = targets["centroid_y"].to_numpy(dtype=np.float32)
    matched = np.full(len(source), -1, dtype=np.int64)
    r2 = radius * radius
    used = np.zeros(len(targets), dtype=bool)
    for i in range(len(source)):
        dx = tx - sx[i]
        dy = ty - sy[i]
        d2 = dx * dx + dy * dy
        d2[used] = np.inf
        j = int(np.argmin(d2))
        if d2[j] <= r2:
            matched[i] = j
            used[j] = True
    return matched


def _run_detector_all(cfg: DetectorConfig, image_dir: Path) -> pd.DataFrame:
    rows = []
    for scan in HAND_SCANS:
        # scan_20260313_3 -> Scan_20260313 (3).png  etc
        num = scan.split("_")[-1]
        src = image_dir / f"Scan_20260313 ({num}).png"
        if not src.exists():
            print(f"  ! missing {src}")
            continue
        out = run_detector(str(src), image_id=scan, cfg=cfg, keep_debug=False)
        inst = out.instances.copy()
        inst["image_id"] = scan
        rows.append(inst)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/Users/stephenyu/Documents/New project/data/probe_inputs/all_20"),
    )
    parser.add_argument(
        "--scans",
        nargs="+",
        default=None,
        help="Subset of scans to run (default: all 10 hand-labeled).",
    )
    parser.add_argument("--edt-weight", type=float, default=0.35)
    parser.add_argument("--min-solidity", type=float, default=0.92)
    parser.add_argument("--min-extent", type=float, default=0.60)
    parser.add_argument("--min-area-ratio", type=float, default=1.8)
    args = parser.parse_args()

    global HAND_SCANS
    if args.scans:
        HAND_SCANS = args.scans

    corrections = _load_corrections()
    corrections = corrections[corrections["image_id"].isin(HAND_SCANS)]

    # Load iter5 (to resolve fp_numbers → positions)
    iter5_csv = REPO_ROOT / "tmp" / "iter5_for_crops" / "iter5" / "instances.csv"
    iter5 = pd.read_csv(iter5_csv)

    # Build FP position table: one row per (image_id, fp_index)
    fp_positions: dict[str, pd.DataFrame] = {}
    for image_id in HAND_SCANS:
        row = corrections[corrections["image_id"] == image_id]
        if row.empty:
            continue
        fps = _parse_fp_numbers(row.iloc[0]["fp_numbers"])
        if not fps:
            fp_positions[image_id] = pd.DataFrame(columns=["centroid_x", "centroid_y", "idx"])
            continue
        sub = iter5[iter5["image_id"] == image_id].reset_index(drop=True)
        pts = []
        for fp in fps:
            if 1 <= fp <= len(sub):
                r = sub.iloc[fp - 1]
                pts.append({"centroid_x": r["centroid_x"], "centroid_y": r["centroid_y"], "idx": fp})
        fp_positions[image_id] = pd.DataFrame(pts)

    cfg_v4 = DetectorConfig()  # defaults = v4
    cfg_sc = DetectorConfig(
        resolver_v2_use_seed_completion=True,
        resolver_v2_seed_completion_edt_weight=args.edt_weight,
        resolver_v2_seed_completion_min_solidity=args.min_solidity,
        resolver_v2_seed_completion_min_extent=args.min_extent,
        resolver_v2_seed_completion_min_area_ratio=args.min_area_ratio,
        min_background_brightness=0.40,
    )

    print("running v4 baseline...")
    v4_df = _run_detector_all(cfg_v4, args.image_dir)
    print("running v4 + seed completion...")
    sc_df = _run_detector_all(cfg_sc, args.image_dir)

    rows = []
    for scan in HAND_SCANS:
        v4 = v4_df[v4_df["image_id"] == scan].reset_index(drop=True)
        sc = sc_df[sc_df["image_id"] == scan].reset_index(drop=True)
        gt = GT_COUNTS.get(scan)

        corr = corrections[corrections["image_id"] == scan]
        miss_count = int(corr.iloc[0]["miss_count"]) if not corr.empty else 0
        fps = fp_positions.get(scan, pd.DataFrame())

        # Known-FP overlap (match iter5 FP positions to current detections)
        if not fps.empty and not v4.empty:
            m_v4 = _match_positions(fps, v4, MATCH_RADIUS_PX)
            v4_fp_kept = int((m_v4 >= 0).sum())
        else:
            v4_fp_kept = 0
        if not fps.empty and not sc.empty:
            m_sc = _match_positions(fps, sc, MATCH_RADIUS_PX)
            sc_fp_kept = int((m_sc >= 0).sum())
        else:
            sc_fp_kept = 0

        # Match v4 to sc (which v4 peaks survive, which sc peaks are new)
        m_v4_to_sc = _match_positions(v4, sc, MATCH_RADIUS_PX)
        v4_in_sc = int((m_v4_to_sc >= 0).sum())
        removed = len(v4) - v4_in_sc
        added = len(sc) - v4_in_sc

        sc_new_rows = sc[sc["resolver_type"] == "seed_completion"]
        sc_fires = int(len(sc_new_rows))

        rows.append(
            {
                "image_id": scan,
                "gt": gt,
                "miss_count": miss_count,
                "n_known_fp": len(fps),
                "v4_n": len(v4),
                "sc_n": len(sc),
                "v4_fp_kept": v4_fp_kept,
                "sc_fp_kept": sc_fp_kept,
                "removed_by_sc": removed,
                "added_by_sc": added,
                "sc_fires": sc_fires,
                "v4_delta": len(v4) - gt if gt else None,
                "sc_delta": len(sc) - gt if gt else None,
            }
        )

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    print()
    if all(df["v4_delta"].notna()):
        v4_err = df["v4_delta"].abs().sum()
        sc_err = df["sc_delta"].abs().sum()
        print(f"image-level sum|err|:  v4={v4_err}  v4_sc={sc_err}")
    print(f"known FPs kept:         v4={df['v4_fp_kept'].sum()}  v4_sc={df['sc_fp_kept'].sum()}")
    print(f"total seed_completion fires: {df['sc_fires'].sum()}")
    print(f"total added by sc:      {df['added_by_sc'].sum()}")
    print(f"total removed by sc:    {df['removed_by_sc'].sum()}")

    out_dir = REPO_ROOT / "tmp" / "seed_completion_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    v4_df.to_csv(out_dir / "v4_instances.csv", index=False)
    sc_df.to_csv(out_dir / "sc_instances.csv", index=False)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
