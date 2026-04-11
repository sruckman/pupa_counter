"""Run the fresh peak-first detector on the 20-image benchmark.

Outputs (under ``--out-dir`` / a dated subdirectory):

* ``counts.csv`` — one row per image: teacher total, pred total, runtime
* ``instances.csv`` — every accepted peak, native coords, per-image export
* ``run_summary.csv`` — per-image matched / teacher_only / pred_only + totals
* ``disagreement_vs_teacher.csv`` — per-image disagreement counts alongside
  the peak totals so the user can see where the gains/regressions came from
* ``teacher_only_instances.csv``, ``pred_only_instances.csv`` — the actual
  disagreement rows for mining
* ``overlays/<image_id>_overlay.png`` — original + accepted peak centers
* ``debug/<image_id>/{blue_mask,response_map,allowed_mask,peak_map}.png``
* ``meta.json`` — detector config + total runtime
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pupa_counter_fresh import DetectorConfig, run_detector  # noqa: E402
from pupa_counter_fresh.eval_instances import (  # noqa: E402
    MatchConfig,
    canonical_scan_number,
    evaluate_disagreement,
    load_teacher_instances,
)


DEFAULT_IMAGE_DIR = Path("/Users/stephenyu/Documents/New project/data/probe_inputs/all_20")
DEFAULT_TEACHER_COUNTS = Path(
    "/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20.csv"
)
DEFAULT_TEACHER_INSTANCES = Path(
    "/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20_instances.csv"
)
DEFAULT_OUT_DIR = Path(
    "/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs"
)


# ---------------------------------------------------------------------------
# Debug / overlay rendering
# ---------------------------------------------------------------------------


def _save_mask(path: Path, mask: np.ndarray) -> None:
    if mask.dtype != np.uint8:
        mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), mask)


def _save_response_map(path: Path, response: np.ndarray) -> None:
    scaled = np.clip(response * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), color)


def _render_overlay(rgb_native: np.ndarray, instances: pd.DataFrame) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_native, cv2.COLOR_RGB2BGR).copy()
    for _, row in instances.iterrows():
        cx = int(round(float(row["centroid_x"])))
        cy = int(round(float(row["centroid_y"])))
        cv2.circle(bgr, (cx, cy), 14, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.circle(bgr, (cx, cy), 2, (0, 180, 255), -1, cv2.LINE_AA)
    header = f"fresh_peak  n={len(instances)}"
    cv2.putText(
        bgr,
        header,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        header,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return bgr


def _write_debug(
    debug: dict[str, np.ndarray], out_dir: Path, image_id: str
) -> None:
    image_debug_dir = out_dir / "debug" / image_id
    image_debug_dir.mkdir(parents=True, exist_ok=True)
    if "blue_mask" in debug:
        _save_mask(image_debug_dir / "blue_mask.png", debug["blue_mask"])
    if "allowed_mask" in debug:
        _save_mask(image_debug_dir / "allowed_mask.png", debug["allowed_mask"])
    if "peak_map" in debug:
        _save_mask(image_debug_dir / "peak_map.png", debug["peak_map"])
    if "response_map" in debug:
        _save_response_map(image_debug_dir / "response_map.png", debug["response_map"])


# ---------------------------------------------------------------------------
# Per-image driver
# ---------------------------------------------------------------------------


def process_image(
    image_path: Path,
    cfg: DetectorConfig,
    out_dir: Path,
) -> dict:
    stem = image_path.stem
    image_id = canonical_scan_number(stem) or stem
    result = run_detector(image_path, image_id=image_id, cfg=cfg, keep_debug=True)

    # Save overlay + debug artifacts
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    overlay = _render_overlay(result.debug["rgb_native"], result.instances)
    cv2.imwrite(str(overlays_dir / f"{image_id}_overlay.png"), overlay)
    _write_debug(result.debug, out_dir, image_id)

    return {
        "image_id": image_id,
        "source_path": str(image_path),
        "pred_total": int(len(result.instances)),
        "runtime_ms": float(result.runtime_ms),
        "instances": result.instances,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fresh peak-first detector on 20 images.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--teacher-counts", type=Path, default=DEFAULT_TEACHER_COUNTS)
    parser.add_argument("--teacher-instances", type=Path, default=DEFAULT_TEACHER_INSTANCES)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--work-scale", type=float, default=0.67)
    parser.add_argument("--peak-threshold", type=float, default=0.22)
    parser.add_argument("--peak-min-distance", type=int, default=10)
    parser.add_argument("--allowed-threshold", type=float, default=0.12)
    parser.add_argument("--detector-backend", type=str, default="fresh_peak_v0")
    parser.add_argument("--instance-source", type=str, default="fresh_peak_v0")
    parser.add_argument(
        "--use-component-split",
        action="store_true",
        help="Enable the v1 per-component peak splitter for touching pupae.",
    )
    parser.add_argument("--component-single-pupa-area", type=float, default=200.0)
    parser.add_argument("--component-area-ratio", type=float, default=1.20)
    parser.add_argument("--component-min-distance", type=int, default=3)
    parser.add_argument("--component-threshold", type=float, default=0.18)
    parser.add_argument("--component-min-area", type=int, default=60)
    parser.add_argument("--component-max-peaks", type=int, default=20)
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        parser.error(f"image dir not found: {args.image_dir}")

    run_name = args.run_name or f"{args.detector_backend}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(exist_ok=True)
    (run_dir / "debug").mkdir(exist_ok=True)

    cfg = DetectorConfig(
        work_scale=args.work_scale,
        peak_abs_score_threshold=args.peak_threshold,
        peak_min_distance_px=args.peak_min_distance,
        allowed_abs_threshold=args.allowed_threshold,
        use_component_split=args.use_component_split,
        component_single_pupa_area_px=args.component_single_pupa_area,
        component_area_ratio_threshold=args.component_area_ratio,
        component_min_peak_distance_px=args.component_min_distance,
        component_abs_score_threshold=args.component_threshold,
        component_min_component_area_px=args.component_min_area,
        component_max_peaks=args.component_max_peaks,
        detector_backend=args.detector_backend,
        instance_source=args.instance_source,
    )

    images = sorted(args.image_dir.glob("*.png"))
    if not images:
        parser.error(f"no PNG images under {args.image_dir}")

    t_start = time.perf_counter()
    per_image: List[dict] = []
    all_instances: List[pd.DataFrame] = []
    for image_path in images:
        print(f"  . {image_path.name}", flush=True)
        row = process_image(image_path, cfg, run_dir)
        per_image.append(row)
        if not row["instances"].empty:
            all_instances.append(row["instances"])
    total_runtime = (time.perf_counter() - t_start) * 1000.0

    # counts.csv
    counts_df = pd.DataFrame(
        [
            {
                "image_id": row["image_id"],
                "source_path": row["source_path"],
                "pred_total": row["pred_total"],
                "runtime_ms": row["runtime_ms"],
            }
            for row in per_image
        ]
    )

    # Join teacher totals
    teacher_counts = pd.read_csv(args.teacher_counts)
    teacher_counts["match_key"] = teacher_counts["image_id"].map(canonical_scan_number)
    counts_df["match_key"] = counts_df["image_id"].map(canonical_scan_number)
    counts_df = counts_df.merge(
        teacher_counts[["match_key", "n_pupa_final"]].rename(
            columns={"n_pupa_final": "teacher_total"}
        ),
        on="match_key",
        how="left",
    )
    counts_df["abs_delta"] = (counts_df["pred_total"] - counts_df["teacher_total"]).abs()
    counts_df.to_csv(run_dir / "counts.csv", index=False)

    # instances.csv
    instances_df = (
        pd.concat(all_instances, ignore_index=True) if all_instances else pd.DataFrame()
    )
    instances_df.to_csv(run_dir / "instances.csv", index=False)

    # Teacher-based disagreement
    teacher_instances = load_teacher_instances(args.teacher_instances)
    summary_df, matches_df, teacher_only_df, pred_only_df = evaluate_disagreement(
        instances_df, teacher_instances, cfg=MatchConfig()
    )
    summary_df.to_csv(run_dir / "run_summary.csv", index=False)
    matches_df.to_csv(run_dir / "matches_vs_teacher.csv", index=False)
    teacher_only_df.to_csv(run_dir / "teacher_only_instances.csv", index=False)
    pred_only_df.to_csv(run_dir / "pred_only_instances.csv", index=False)

    # disagreement_vs_teacher.csv (per-image join of counts + disagreement)
    joined = counts_df.merge(
        summary_df[
            [
                "match_key",
                "teacher_instances",
                "pred_instances",
                "matched",
                "teacher_only",
                "pred_only",
                "mean_centroid_distance_px",
            ]
        ],
        on="match_key",
        how="left",
    )
    joined.to_csv(run_dir / "disagreement_vs_teacher.csv", index=False)

    # meta.json
    meta = {
        "run_name": run_name,
        "detector_backend": cfg.detector_backend,
        "instance_source": cfg.instance_source,
        "config": asdict(cfg),
        "total_runtime_ms": total_runtime,
        "mean_runtime_ms": float(counts_df["runtime_ms"].mean()),
        "n_images": int(len(counts_df)),
        "totals": {
            "teacher": int(pd.to_numeric(counts_df["teacher_total"], errors="coerce").sum()),
            "pred": int(counts_df["pred_total"].sum()),
            "matched": int(summary_df["matched"].sum()),
            "teacher_only": int(summary_df["teacher_only"].sum()),
            "pred_only": int(summary_df["pred_only"].sum()),
            "recall": (
                float(summary_df["matched"].sum() / max(1, summary_df["teacher_instances"].sum()))
            ),
            "precision": (
                float(summary_df["matched"].sum() / max(1, summary_df["pred_instances"].sum()))
            ),
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Terminal summary
    print()
    print(f"== {run_name} ==")
    print(f"images           : {meta['n_images']}")
    print(f"mean runtime_ms  : {meta['mean_runtime_ms']:.1f}")
    print(f"teacher total    : {meta['totals']['teacher']}")
    print(f"pred total       : {meta['totals']['pred']}")
    print(f"matched          : {meta['totals']['matched']}")
    print(f"teacher_only     : {meta['totals']['teacher_only']}")
    print(f"pred_only        : {meta['totals']['pred_only']}")
    print(f"recall           : {meta['totals']['recall']:.3f}")
    print(f"precision        : {meta['totals']['precision']:.3f}")
    print(f"output dir       : {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
