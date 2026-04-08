"""Run summary report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def build_run_report(
    counts_df: pd.DataFrame,
    review_df: pd.DataFrame,
    output_dir: Path,
    metrics: Dict[str, float] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {}
    lines = ["# Run Summary", ""]
    lines.append("## Aggregate")
    lines.append("")
    lines.append("- Images processed: %s" % len(counts_df))
    lines.append("- Images needing review: %s" % int(review_df.shape[0]))
    if not counts_df.empty:
        lines.append("- Mean middle count: %.2f" % counts_df["n_middle"].mean())
        lines.append("- Mean total count: %.2f" % counts_df["n_pupa_final"].mean())
    if metrics:
        lines.append("")
        lines.append("## Metrics")
        lines.append("")
        for key, value in sorted(metrics.items()):
            lines.append("- %s: %.4f" % (key, value))
    lines.append("")
    lines.append("## Top Review Cases")
    lines.append("")
    if review_df.empty:
        lines.append("No flagged images.")
    else:
        for _, row in review_df.head(20).iterrows():
            lines.append("- %s | %s" % (row["image_id"], row["review_reason"]))

    markdown = "\n".join(lines) + "\n"
    (output_dir / "run_summary.md").write_text(markdown, encoding="utf-8")
    html = "<html><body><pre>%s</pre></body></html>" % markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    (output_dir / "run_summary.html").write_text(html, encoding="utf-8")
