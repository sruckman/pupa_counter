from __future__ import annotations

from pathlib import Path

import pandas as pd

from pupa_counter.report.worksheet import build_running_totals_frame, export_running_totals_workbook


def test_build_running_totals_frame_adds_cumulative_columns():
    counts_df = pd.DataFrame(
        [
            {
                "image_id": "img_a",
                "source_path": "/tmp/a.png",
                "n_top_5pct": 1,
                "n_top": 2,
                "n_middle": 10,
                "n_bottom": 3,
                "n_pupa_final": 15,
                "needs_review": False,
                "review_reason": "",
                "runtime_ms": 2500.0,
            },
            {
                "image_id": "img_b",
                "source_path": "/tmp/b.png",
                "n_top_5pct": 0,
                "n_top": 1,
                "n_middle": 8,
                "n_bottom": 2,
                "n_pupa_final": 11,
                "needs_review": True,
                "review_reason": "border_anchor",
                "runtime_ms": 1500.0,
            },
        ]
    )

    frame = build_running_totals_frame(counts_df)

    assert frame["image_order"].tolist() == [1, 2]
    assert frame["source_name"].tolist() == ["a.png", "b.png"]
    assert frame["cumulative_total"].tolist() == [15, 26]
    assert frame["cumulative_middle"].tolist() == [10, 18]
    assert frame["runtime_s"].tolist() == [2.5, 1.5]


def test_export_running_totals_workbook_writes_csv_and_xlsx(tmp_path: Path):
    counts_df = pd.DataFrame(
        [
            {
                "image_id": "img_a",
                "source_path": "/tmp/a.png",
                "n_top_5pct": 1,
                "n_top": 2,
                "n_middle": 10,
                "n_bottom": 3,
                "n_pupa_final": 15,
                "needs_review": False,
                "review_reason": "",
                "runtime_ms": 2500.0,
            }
        ]
    )
    review_df = pd.DataFrame([{"image_id": "img_a", "reason": "ok"}])

    workbook_path = export_running_totals_workbook(tmp_path, counts_df, review_df)

    assert (tmp_path / "running_totals.csv").exists()
    assert workbook_path.exists()

    workbook = pd.ExcelFile(workbook_path)
    assert set(workbook.sheet_names) >= {"running_totals", "summary", "review_queue"}
