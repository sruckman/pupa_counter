"""Input discovery and manifest helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

import fitz
import pandas as pd
from PIL import Image

from pupa_counter.config import AppConfig
from pupa_counter.types import ImageRecord


def _source_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".pdf":
        return "clean_pdf"
    if suffix == ".pptx":
        return "example"
    if "scan_20260313" in name:
        return "annotated_png"
    return "clean_png"


def _split_for_source(source_type: str) -> str:
    mapping = {
        "annotated_png": "dev_annotated",
        "clean_pdf": "holdout_clean",
        "clean_png": "holdout_clean",
        "example": "example",
        "derived": "derived",
    }
    return mapping.get(source_type, "unknown")


def _stable_id(relative_path: str, page_index: int = None) -> str:
    key = relative_path if page_index is None else "%s#%s" % (relative_path, page_index)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    stem = Path(relative_path).stem.lower().replace(" ", "_").replace("(", "").replace(")", "")
    if page_index is None:
        return "%s_%s" % (stem, digest)
    return "%s_p%03d_%s" % (stem, page_index + 1, digest)


def _iter_supported_paths(input_root: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    normalized = {suffix.lower() for suffix in suffixes}
    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in normalized or path.suffix.lower() == ".pptx":
            yield path


def discover_inputs(input_root: Path, cfg: AppConfig) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for path in _iter_supported_paths(input_root, cfg.input.accepted_suffixes):
        relative_path = str(path.relative_to(input_root))
        source_type = _source_type_for_path(path)
        split = _split_for_source(source_type)
        has_blue_hint = source_type == "annotated_png"

        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as document:
                for page_index in range(document.page_count):
                    page = document.load_page(page_index)
                    width = int(round(page.rect.width * cfg.input.raster_dpi / 72.0))
                    height = int(round(page.rect.height * cfg.input.raster_dpi / 72.0))
                    records.append(
                        ImageRecord(
                            image_id=_stable_id(relative_path, page_index),
                            source_path=path,
                            source_type=source_type,
                            split=split,
                            width=width,
                            height=height,
                            dpi=cfg.input.raster_dpi,
                            has_blue_hint=has_blue_hint,
                            notes="pdf_page=%s" % page_index,
                            relative_path=relative_path,
                            page_index=page_index,
                        )
                    )
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(path) as image:
                width, height = image.size
            records.append(
                ImageRecord(
                    image_id=_stable_id(relative_path),
                    source_path=path,
                    source_type=source_type,
                    split=split,
                    width=width,
                    height=height,
                    dpi=None,
                    has_blue_hint=has_blue_hint,
                    notes="",
                    relative_path=relative_path,
                )
            )
        else:
            records.append(
                ImageRecord(
                    image_id=_stable_id(relative_path),
                    source_path=path,
                    source_type=source_type,
                    split=split,
                    width=None,
                    height=None,
                    dpi=None,
                    has_blue_hint=None,
                    notes="reference_only",
                    relative_path=relative_path,
                )
            )
    return records


def manifest_dataframe(records: List[ImageRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_row() for record in records]).sort_values(
        ["split", "source_type", "relative_path", "page_index"], na_position="last"
    )
