"""Helpers for writing pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from PIL import Image


def ensure_run_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "run": root,
        "overlays": root / "overlays",
        "intermediate": root / "intermediate",
        "reports": root / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_mask(path: Path, mask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def save_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def serializable_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    drop_cols = [col for col in frame.columns if col in {"mask"}]
    cleaned = frame.drop(columns=drop_cols, errors="ignore").copy()
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: value.tolist() if hasattr(value, "tolist") else value
        )
    return cleaned
