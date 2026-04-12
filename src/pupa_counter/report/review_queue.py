"""Review flag generation and review queue export helpers."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.types import CountSummary, ReviewFlag, flags_to_reason


def _anchor_row(instances_df: pd.DataFrame, role: str, fallback: str) -> pd.Series:
    if "anchor_role" in instances_df.columns:
        candidates = instances_df.loc[instances_df["anchor_role"] == role]
        if not candidates.empty:
            return candidates.iloc[0]
    if fallback == "top":
        return instances_df.loc[instances_df["centroid_y"].idxmin()]
    return instances_df.loc[instances_df["centroid_y"].idxmax()]


def build_review_flags(
    summary: CountSummary,
    instances_df: pd.DataFrame,
    candidate_df: Optional[pd.DataFrame] = None,
    previous_row: Optional[pd.Series] = None,
    cfg: AppConfig = None,
) -> List[ReviewFlag]:
    cfg = cfg or AppConfig()
    flags: List[ReviewFlag] = []

    if summary.n_pupa_final < cfg.counting.min_final_instances:
        flags.append(
            ReviewFlag(
                code="too_few_detections",
                severity="high",
                message="Final accepted pupa count is below the minimum threshold.",
            )
        )

    if summary.top_y is not None and summary.bottom_y is not None:
        span = summary.bottom_y - summary.top_y
        if span < cfg.counting.min_span_px:
            flags.append(
                ReviewFlag(
                    code="small_anchor_span",
                    severity="high",
                    message="Anchor span is too small for stable middle-band counting.",
                )
            )

    if cfg.review.flag_low_anchor_confidence and not instances_df.empty:
        top_row = _anchor_row(instances_df, "top", "top")
        bottom_row = _anchor_row(instances_df, "bottom", "bottom")
        anchor_confidence = min(float(top_row["anchor_confidence"]), float(bottom_row["anchor_confidence"]))
        if anchor_confidence < cfg.review.low_anchor_confidence_threshold:
            flags.append(
                ReviewFlag(
                    code="low_confidence_anchors",
                    severity="high",
                    message="Top or bottom anchor confidence is low.",
                )
            )

    if cfg.review.flag_border_anchor and not instances_df.empty:
        top_row = _anchor_row(instances_df, "top", "top")
        bottom_row = _anchor_row(instances_df, "bottom", "bottom")
        if (
            float(top_row.get("border_touch_ratio", 0.0)) >= cfg.review.border_anchor_threshold
            or float(bottom_row.get("border_touch_ratio", 0.0)) >= cfg.review.border_anchor_threshold
            or bool(top_row.get("touches_image_border", False))
            or bool(bottom_row.get("touches_image_border", False))
        ):
            flags.append(
                ReviewFlag(
                    code="border_anchor",
                    severity="medium",
                    message="Top or bottom anchor touches the image border.",
                )
            )

    if cfg.review.flag_unresolved_cluster and summary.unresolved_clusters > 0:
        unresolved_threshold = max(
            cfg.review.unresolved_cluster_min_count,
            int(round(summary.n_pupa_final * cfg.review.unresolved_cluster_ratio_threshold)),
        )
        if summary.unresolved_clusters >= unresolved_threshold:
            flags.append(
                ReviewFlag(
                    code="unresolved_clusters",
                    severity="high",
                    message="Merged cluster candidates could not be split cleanly.",
                )
            )

    if summary.blue_pixel_ratio is not None and summary.blue_pixel_ratio > cfg.review.flag_high_blue_ratio_threshold:
        flags.append(
            ReviewFlag(
                code="high_blue_overlap",
                severity="medium",
                message="Blue annotation ratio is unusually high.",
            )
        )

    trusted_middle_disagreement = summary.extra.get("trusted_middle_disagreement")
    if trusted_middle_disagreement is not None and trusted_middle_disagreement >= cfg.review.flag_blue_trust_disagreement_threshold:
        flags.append(
            ReviewFlag(
                code="blue_trust_disagreement",
                severity="high",
                message="Predicted middle-band count differs substantially from trusted blue annotation supervision.",
            )
        )

    if not instances_df.empty:
        mean_color_score = float(instances_df["color_score"].mean())
        if mean_color_score < cfg.review.suspicious_color_low:
            flags.append(
                ReviewFlag(
                    code="suspicious_color_distribution",
                    severity="medium",
                    message="Accepted instances have an unusually weak brown color score.",
                )
            )

    if previous_row is not None:
        previous_middle = int(previous_row["n_middle"])
        if abs(summary.n_middle - previous_middle) >= cfg.review.flag_large_run_diff_threshold:
            flags.append(
                ReviewFlag(
                    code="large_disagreement_vs_previous",
                    severity="medium",
                    message="Middle-band count differs substantially from the previous run.",
                )
            )

    summary.needs_review = bool(flags)
    summary.review_reason = flags_to_reason(flags)
    return flags


def build_review_queue_frame(
    summaries: List[CountSummary],
    flags_by_image: Dict[str, List[ReviewFlag]],
    overlay_dir: str,
) -> pd.DataFrame:
    rows = []
    severity_order = {"high": 3, "medium": 2, "low": 1}
    for summary in summaries:
        flags = flags_by_image.get(summary.image_id, [])
        if not summary.needs_review:
            continue
        highest = max((severity_order[flag.severity] for flag in flags), default=0)
        rows.append(
            {
                "image_id": summary.image_id,
                "source_path": summary.source_path,
                "split": summary.split,
                "n_middle": summary.n_middle,
                "n_pupa_final": summary.n_pupa_final,
                "unresolved_clusters": summary.unresolved_clusters,
                "severity_rank": highest,
                "flag_codes": ",".join(flag.code for flag in flags),
                "review_reason": summary.review_reason,
                "overlay_path": "%s/%s.png" % (overlay_dir.rstrip("/"), summary.image_id),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "image_id",
                "source_path",
                "split",
                "n_middle",
                "n_pupa_final",
                "unresolved_clusters",
                "severity_rank",
                "flag_codes",
                "review_reason",
                "overlay_path",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["severity_rank", "unresolved_clusters", "n_middle"], ascending=[False, False, False]
    )
