"""Rasterization and image loading."""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageOps

from pupa_counter.types import ImageRecord


def rasterize_record(record: ImageRecord, dpi: int = 300) -> np.ndarray:
    suffix = record.source_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(record.source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(image, dtype=np.uint8)
    if suffix == ".pdf":
        page_index = 0 if record.page_index is None else record.page_index
        with fitz.open(record.source_path) as document:
            page = document.load_page(page_index)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB"
            image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
            return np.asarray(image, dtype=np.uint8)
    raise ValueError("Unsupported rasterization input: %s" % record.source_path)
