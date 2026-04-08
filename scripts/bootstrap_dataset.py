#!/usr/bin/env python3
"""Create a stable data/raw symlink to an external batch folder."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default="data/raw/input_batch")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
