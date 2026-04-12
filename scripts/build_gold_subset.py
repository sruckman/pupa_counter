#!/usr/bin/env python3
"""Create a gold-subset CSV template from an existing counts file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--output", default="data/gold/image_level_labels.csv")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    counts_df = pd.read_csv(args.counts).head(args.limit)
    gold_df = counts_df[["image_id", "source_path"]].copy()
    gold_df["true_total"] = ""
    gold_df["true_top"] = ""
    gold_df["true_middle"] = ""
    gold_df["true_bottom"] = ""

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold_df.to_csv(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
