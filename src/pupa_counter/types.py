"""Core typed data structures for the pupa counting pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


SourceType = Literal["annotated_png", "clean_pdf", "clean_png", "example", "derived"]
CandidateLabel = Literal["pupa", "artifact", "cluster", "uncertain"]
AnchorMode = Literal["centroid", "bbox_edge"]


@dataclass
class ImageRecord:
    image_id: str
    source_path: Path
    source_type: SourceType
    split: str
    width: Optional[int] = None
    height: Optional[int] = None
    dpi: Optional[int] = None
    has_blue_hint: Optional[bool] = None
    notes: str = ""
    relative_path: str = ""
    page_index: Optional[int] = None

    def to_row(self) -> Dict[str, object]:
        row = asdict(self)
        row["source_path"] = str(self.source_path)
        return row


@dataclass
class BandGeometry:
    top_y: float
    bottom_y: float
    upper_five_pct_y: float
    upper_middle_y: float
    lower_middle_y: float
    anchor_mode: AnchorMode = "centroid"

    def to_row(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ReviewFlag:
    code: str
    severity: Literal["low", "medium", "high"]
    message: str

    def to_row(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class CountSummary:
    image_id: str
    source_path: str
    split: str
    n_candidates_raw: int
    n_pupa_final: int
    n_top_5pct: int
    n_top: int
    n_middle: int
    n_bottom: int
    top_y: Optional[float]
    bottom_y: Optional[float]
    upper_five_pct_y: Optional[float]
    upper_middle_y: Optional[float]
    lower_middle_y: Optional[float]
    mean_confidence: Optional[float]
    unresolved_clusters: int
    blue_pixel_ratio: Optional[float]
    needs_review: bool
    review_reason: str
    config_version: str
    model_version: Optional[str]
    runtime_ms: Optional[float]
    extra: Dict[str, object] = field(default_factory=dict)

    def to_row(self) -> Dict[str, object]:
        row = asdict(self)
        extra = row.pop("extra", {}) or {}
        row.update(extra)
        return row


def flags_to_reason(flags: List[ReviewFlag]) -> str:
    if not flags:
        return ""
    return " | ".join("%s: %s" % (flag.code, flag.message) for flag in flags)
