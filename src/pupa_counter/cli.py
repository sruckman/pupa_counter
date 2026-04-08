"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pupa_counter.config import load_config
from pupa_counter.eval.metrics import evaluate_counts
from pupa_counter.io.discover import discover_inputs, manifest_dataframe
from pupa_counter.pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-first pupa counting pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Build manifest.csv from an input root")
    manifest_parser.add_argument("--input-root", required=True)
    manifest_parser.add_argument("--config", default="configs/base.yaml")
    manifest_parser.add_argument("--output", default=None)

    run_parser = subparsers.add_parser("run", help="Run the full pupa counting pipeline")
    run_parser.add_argument("--input-root", required=True)
    run_parser.add_argument("--config", default="configs/base.yaml")
    run_parser.add_argument("--output-root", default="data/processed/runs")
    run_parser.add_argument("--gold-csv", default=None)
    run_parser.add_argument("--limit", type=int, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a prediction CSV against a gold CSV")
    eval_parser.add_argument("--pred", required=True)
    eval_parser.add_argument("--gold", required=True)

    config_parser = subparsers.add_parser("print-config", help="Print the effective config as JSON")
    config_parser.add_argument("--config", default="configs/base.yaml")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "manifest":
        cfg = load_config(Path(args.config))
        records = discover_inputs(Path(args.input_root), cfg)
        manifest_df = manifest_dataframe(records)
        output_path = Path(args.output) if args.output else Path("data/manifests/manifest.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(output_path, index=False)
        print("manifest_rows=%s" % len(manifest_df))
        print("manifest_path=%s" % output_path)
        return 0

    if args.command == "run":
        cfg = load_config(Path(args.config))
        results = run_pipeline(
            input_root=Path(args.input_root),
            cfg=cfg,
            output_root=Path(args.output_root),
            gold_csv=Path(args.gold_csv) if args.gold_csv else None,
            limit=args.limit,
        )
        print("run_root=%s" % results["run_root"])
        print("images=%s" % len(results["counts_df"]))
        print("review=%s" % len(results["review_df"]))
        if results["metrics"]:
            print("metrics=%s" % json.dumps(results["metrics"], ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "evaluate":
        pred_df = pd.read_csv(args.pred)
        gold_df = pd.read_csv(args.gold)
        metrics = evaluate_counts(pred_df, gold_df)
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if args.command == "print-config":
        cfg = load_config(Path(args.config))
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
