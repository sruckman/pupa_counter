"""Autonomous response-sharpening sweep for the fresh peak-first detector.

Reads a YAML grid of detector configurations, runs ``run_detector`` against
a fixed set of benchmark images (in-process, no subprocess overhead), scores
each configuration against the teacher instances, and picks a winner by F1
under a precision / recall floor.

Output files (under ``--out-dir``):

* ``sweep.csv``  — one row per grid point with all metrics
* ``best.json``  — winner config + metrics + path to the detector run
* ``winning_run/`` — the same detector called one more time with
  ``keep_debug=True`` so overlays + debug images are rendered for the
  visual audit step

Usage::

    python scripts/run_fresh_sweep.py \\
        --grid configs/sweeps/log_dog_v1.yaml \\
        --out-dir /path/to/out \\
        --image-dir /path/to/20-image-probe \\
        --teacher-instances /path/to/teacher_v8_20_instances.csv \\
        --early-stop-recall 0.94

Floors and the acceptance gate come from the grid YAML itself (``floors``
and ``gate`` top-level keys) so each sweep file is self-contained.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

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


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _expand_variant(base: dict, variant: dict) -> list[dict]:
    fixed = variant.get("fixed", {})
    axes = variant.get("axes", {})
    if not axes:
        cfg = {**base, **fixed}
        cfg["_variant"] = variant.get("name", "unnamed")
        return [cfg]

    axis_names = list(axes.keys())
    axis_values = [axes[name] for name in axis_names]
    combos = list(itertools.product(*axis_values))
    result: list[dict] = []
    for combo in combos:
        cfg = {**base, **fixed}
        for name, value in zip(axis_names, combo):
            cfg[name] = value
        cfg["_variant"] = variant.get("name", "unnamed")
        result.append(cfg)
    return result


def expand_grid(grid_yaml: dict) -> list[dict]:
    base = grid_yaml.get("base", {})
    variants = grid_yaml.get("variants", [])
    all_points: list[dict] = []
    for variant in variants:
        all_points.extend(_expand_variant(base, variant))
    return all_points


# ---------------------------------------------------------------------------
# Single run evaluation
# ---------------------------------------------------------------------------


def _config_from_dict(d: dict) -> DetectorConfig:
    """Turn a flat dict into a DetectorConfig, tolerating extra keys."""
    field_names = {f.name for f in dataclasses.fields(DetectorConfig)}
    kwargs = {k: v for k, v in d.items() if k in field_names}
    return DetectorConfig(**kwargs)


def evaluate_config(
    cfg_dict: dict,
    image_paths: list[Path],
    teacher: pd.DataFrame,
    *,
    keep_debug: bool = False,
) -> dict:
    """Run one grid point and return its metrics as a flat dict."""
    cfg = _config_from_dict(cfg_dict)
    rows: list[pd.DataFrame] = []
    total_runtime_ms = 0.0

    for path in image_paths:
        result = run_detector(
            path,
            image_id=canonical_scan_number(path.stem),
            cfg=cfg,
            keep_debug=keep_debug,
        )
        rows.append(result.instances)
        total_runtime_ms += result.runtime_ms

    if rows:
        pred_df = pd.concat(rows, ignore_index=True)
    else:
        pred_df = pd.DataFrame()

    summary_df, *_ = evaluate_disagreement(pred_df, teacher, cfg=MatchConfig())
    teacher_total = int(summary_df["teacher_instances"].sum()) if not summary_df.empty else 0
    pred_total = int(summary_df["pred_instances"].sum()) if not summary_df.empty else 0
    matched = int(summary_df["matched"].sum()) if not summary_df.empty else 0
    teacher_only = int(summary_df["teacher_only"].sum()) if not summary_df.empty else 0
    pred_only = int(summary_df["pred_only"].sum()) if not summary_df.empty else 0

    recall = matched / teacher_total if teacher_total > 0 else 0.0
    precision = matched / pred_total if pred_total > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    mean_runtime_ms = total_runtime_ms / max(1, len(image_paths))

    record = {
        # Identifier
        "variant": cfg_dict.get("_variant", "unnamed"),
        # Metrics
        "teacher_total": teacher_total,
        "pred_total": pred_total,
        "matched": matched,
        "teacher_only": teacher_only,
        "pred_only": pred_only,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "mean_runtime_ms": round(mean_runtime_ms, 2),
        "total_runtime_ms": round(total_runtime_ms, 2),
    }
    # Also log every swept hyperparameter so the CSV is self-describing
    for k, v in cfg_dict.items():
        if k.startswith("_"):
            continue
        # Already logged above or part of DetectorConfig
        if k in record:
            continue
        record[f"cfg.{k}"] = v
    return record


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------


def pick_winner(
    records: list[dict],
    *,
    precision_floor: float,
    recall_floor: float,
) -> dict | None:
    survivors = [
        r for r in records
        if r["precision"] >= precision_floor and r["recall"] >= recall_floor
    ]
    if not survivors:
        return None
    survivors.sort(key=lambda r: (r["f1"], r["recall"], -r["mean_runtime_ms"]), reverse=True)
    return survivors[0]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous v2 response sharpening sweep.")
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/Users/stephenyu/Documents/New project/data/probe_inputs/all_20"),
    )
    parser.add_argument(
        "--teacher-instances",
        type=Path,
        default=Path(
            "/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20_instances.csv"
        ),
    )
    parser.add_argument(
        "--restrict",
        nargs="*",
        default=None,
        help="Optional list of canonical scan keys (e.g. scan_20260313_25). "
        "When provided the sweep only runs on these images — useful for "
        "the 5-hardest-image fast sweep.",
    )
    parser.add_argument(
        "--early-stop-recall",
        type=float,
        default=None,
        help="If provided, exit the loop the moment a survivor reaches this recall "
        "(still under the precision floor).",
    )
    parser.add_argument(
        "--skip-winning-rerun",
        action="store_true",
        help="Do not rerun the winner with overlays. Useful for fast sweeps.",
    )
    args = parser.parse_args()

    grid_yaml = yaml.safe_load(args.grid.read_text())
    base = grid_yaml.get("base", {})
    floors = grid_yaml.get("floors", {})
    gate = grid_yaml.get("gate", {})
    precision_floor = float(floors.get("precision_floor", 0.94))
    recall_floor = float(floors.get("recall_floor", 0.90))
    target_recall = float(gate.get("target_recall", 0.94))
    target_precision = float(gate.get("target_precision", 0.94))
    target_f1 = float(gate.get("target_f1", 0.93))

    grid_points = expand_grid(grid_yaml)
    print(f"grid points: {len(grid_points)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.image_dir.glob("*.png"))
    if args.restrict:
        wanted = set(args.restrict)
        image_paths = [p for p in image_paths if canonical_scan_number(p.stem) in wanted]
        if not image_paths:
            parser.error(f"--restrict filtered out every image in {args.image_dir}")
    print(f"images: {len(image_paths)}")

    teacher = load_teacher_instances(args.teacher_instances)
    # Restrict the teacher frame to the SAME image subset, otherwise the
    # ``teacher_only`` count leaks in rows from images the sweep never ran.
    image_keys = {canonical_scan_number(p.stem) for p in image_paths}
    teacher_keys = teacher["image_id"].map(canonical_scan_number)
    teacher = teacher.loc[teacher_keys.isin(image_keys)].reset_index(drop=True)
    print(f"teacher rows (after image filter): {len(teacher)}")

    records: list[dict] = []
    t0 = time.perf_counter()
    early_stop_hit = False
    for i, cfg_dict in enumerate(grid_points, start=1):
        result = evaluate_config(cfg_dict, image_paths, teacher, keep_debug=False)
        records.append(result)
        label = result["variant"]
        print(
            f"[{i:3d}/{len(grid_points)}] {label:18s} R={result['recall']:.3f} "
            f"P={result['precision']:.3f} F1={result['f1']:.3f} "
            f"m={result['matched']:4d} to={result['teacher_only']:3d} po={result['pred_only']:3d} "
            f"rt={result['mean_runtime_ms']:.0f}ms",
            flush=True,
        )
        if (
            args.early_stop_recall is not None
            and result["recall"] >= args.early_stop_recall
            and result["precision"] >= precision_floor
        ):
            early_stop_hit = True
            print(f"[early-stop] recall {result['recall']:.3f} >= {args.early_stop_recall:.3f}; exiting loop")
            break
    elapsed = time.perf_counter() - t0
    print(f"sweep finished in {elapsed:.1f}s")

    sweep_df = pd.DataFrame(records)
    sweep_csv = args.out_dir / "sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"wrote {sweep_csv}")

    winner = pick_winner(
        records,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
    )

    if winner is None:
        msg = (
            f"no config passed the floors (precision >= {precision_floor}, "
            f"recall >= {recall_floor}). Sweep recorded {len(records)} runs "
            f"in {sweep_csv}. Widen the grid and re-run."
        )
        print(f"[NO WINNER] {msg}")
        (args.out_dir / "best.json").write_text(
            json.dumps(
                {
                    "status": "no_winner",
                    "precision_floor": precision_floor,
                    "recall_floor": recall_floor,
                    "reason": msg,
                    "records": records,
                },
                indent=2,
            )
        )
        return 1

    gate_hit = (
        winner["recall"] >= target_recall
        and winner["precision"] >= target_precision
        and winner["f1"] >= target_f1
    )

    winner_cfg = {k: v for k, v in winner.items() if k.startswith("cfg.")}
    winner_full_cfg = {
        **base,
        **{k[len("cfg."):]: v for k, v in winner_cfg.items()},
    }
    winner_full_cfg["response_mode"] = winner_full_cfg.get("response_mode", "smooth")

    best = {
        "status": "gate_passed" if gate_hit else "gate_missed_but_best",
        "early_stop_hit": early_stop_hit,
        "precision_floor": precision_floor,
        "recall_floor": recall_floor,
        "target_recall": target_recall,
        "target_precision": target_precision,
        "target_f1": target_f1,
        "winner_metrics": {
            "variant": winner["variant"],
            "recall": winner["recall"],
            "precision": winner["precision"],
            "f1": winner["f1"],
            "matched": winner["matched"],
            "teacher_only": winner["teacher_only"],
            "pred_only": winner["pred_only"],
            "mean_runtime_ms": winner["mean_runtime_ms"],
        },
        "winner_config": winner_full_cfg,
        "n_grid_points": len(grid_points),
        "n_evaluated": len(records),
        "image_count": len(image_paths),
        "image_keys": sorted({canonical_scan_number(p.stem) for p in image_paths}),
    }
    (args.out_dir / "best.json").write_text(json.dumps(best, indent=2))

    print()
    print("== winner ==")
    print(json.dumps(best["winner_metrics"], indent=2))
    print(f"variant           : {winner['variant']}")
    print(f"gate passed       : {gate_hit}")
    print(f"early-stop hit    : {early_stop_hit}")
    print(f"config            : {winner_full_cfg}")

    if not args.skip_winning_rerun:
        winning_dir = args.out_dir / "winning_run"
        winning_dir.mkdir(exist_ok=True)
        print()
        print(f"rerunning winner with keep_debug=True into {winning_dir} ...")
        # Overlay rendering lives in run_fresh_peak_detector.py, not here, so
        # we skip it and just stash the instances + summary. Overlays for
        # visual audit are handled by a separate explicit call to
        # scripts/run_fresh_peak_detector.py after the sweep exits.
        rows: list[pd.DataFrame] = []
        cfg = _config_from_dict(winner_full_cfg)
        for path in image_paths:
            result = run_detector(
                path,
                image_id=canonical_scan_number(path.stem),
                cfg=cfg,
                keep_debug=False,
            )
            rows.append(result.instances)
        instances = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        instances.to_csv(winning_dir / "instances.csv", index=False)
        print(f"  wrote {winning_dir / 'instances.csv'}  (n={len(instances)})")

    return 0 if gate_hit else 2


if __name__ == "__main__":
    raise SystemExit(main())
