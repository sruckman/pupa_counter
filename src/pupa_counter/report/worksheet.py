"""Running totals worksheet export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_running_totals_frame(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Create a shareable per-image worksheet from the run counts table."""
    if counts_df.empty:
        return pd.DataFrame(
            columns=[
                "image_order",
                "image_id",
                "source_name",
                "source_path",
                "n_top_5pct",
                "n_top",
                "n_middle",
                "n_bottom",
                "n_pupa_final",
                "cumulative_total",
                "cumulative_middle",
                "needs_review",
                "review_reason",
                "runtime_s",
            ]
        )

    frame = counts_df.copy().reset_index(drop=True)
    frame.insert(0, "image_order", range(1, len(frame) + 1))
    frame["source_name"] = frame["source_path"].apply(lambda value: Path(str(value)).name)
    frame["runtime_s"] = frame["runtime_ms"].astype(float) / 1000.0
    frame["cumulative_total"] = frame["n_pupa_final"].astype(int).cumsum()
    frame["cumulative_middle"] = frame["n_middle"].astype(int).cumsum()

    preferred_columns = [
        "image_order",
        "image_id",
        "source_name",
        "source_path",
        "n_top_5pct",
        "n_top",
        "n_middle",
        "n_bottom",
        "n_pupa_final",
        "cumulative_total",
        "cumulative_middle",
        "needs_review",
        "review_reason",
        "runtime_s",
    ]
    remaining_columns = [column for column in frame.columns if column not in preferred_columns]
    return frame.loc[:, preferred_columns + remaining_columns]


def export_running_totals_workbook(
    run_root: Path,
    counts_df: pd.DataFrame,
    review_df: pd.DataFrame,
) -> Path:
    """Write CSV + XLSX worksheet artifacts for sharing outside the codebase."""
    run_root.mkdir(parents=True, exist_ok=True)
    running_totals_df = build_running_totals_frame(counts_df)
    running_totals_csv = run_root / "running_totals.csv"
    running_totals_df.to_csv(running_totals_csv, index=False)

    workbook_path = run_root / "running_totals.xlsx"
    summary_rows = [
        {"metric": "images_processed", "value": int(len(counts_df))},
        {"metric": "images_needing_review", "value": int(len(review_df))},
        {
            "metric": "mean_total_count",
            "value": None if counts_df.empty else float(counts_df["n_pupa_final"].astype(float).mean()),
        },
        {
            "metric": "mean_middle_count",
            "value": None if counts_df.empty else float(counts_df["n_middle"].astype(float).mean()),
        },
        {
            "metric": "mean_runtime_s",
            "value": None if counts_df.empty else float((counts_df["runtime_ms"].astype(float) / 1000.0).mean()),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        running_totals_df.to_excel(writer, index=False, sheet_name="running_totals")
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        if review_df is not None and not review_df.empty:
            review_df.to_excel(writer, index=False, sheet_name="review_queue")

        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells:
                    if cell.value is None:
                        continue
                    max_length = max(max_length, len(str(cell.value)))
                worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 48)

    return workbook_path
