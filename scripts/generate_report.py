#!/usr/bin/env python3
"""Regenerate the markdown/html run summary from saved CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pupa_counter.report.html_report import build_run_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    build_run_report(pd.read_csv(args.counts), pd.read_csv(args.review), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
