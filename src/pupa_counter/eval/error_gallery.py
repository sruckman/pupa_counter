"""Simple error gallery generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_error_gallery(merged_df: pd.DataFrame, output_path: Path) -> None:
    if merged_df.empty:
        output_path.write_text("# Error Gallery\n\nNo matched evaluation rows.\n", encoding="utf-8")
        return

    ranked = merged_df.sort_values("abs_error_middle", ascending=False).head(25)
    lines = ["# Error Gallery", ""]
    for _, row in ranked.iterrows():
        lines.append(
            "- %s | pred_middle=%s | true_middle=%s | abs_error_middle=%s"
            % (
                row.get("image_id", "unknown"),
                row.get("n_middle_pred", row.get("n_middle")),
                row.get("true_middle", row.get("n_middle_gold")),
                row.get("abs_error_middle", "n/a"),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
