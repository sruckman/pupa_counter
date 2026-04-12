"""Run the fresh peak-first detector on the 20-image benchmark.

Outputs (under ``--out-dir`` / a dated subdirectory):

* ``counts.csv`` — one row per image: teacher total, pred total, runtime
* ``instances.csv`` — every accepted peak, native coords, per-image export
* ``run_summary.csv`` — per-image matched / teacher_only / pred_only + totals
* ``disagreement_vs_teacher.csv`` — per-image disagreement counts alongside
  the peak totals so the user can see where the gains/regressions came from
* ``teacher_only_instances.csv``, ``pred_only_instances.csv`` — the actual
  disagreement rows for mining
* ``overlays/<image_id>_overlay.png`` — original + accepted peak centers
* ``debug/<image_id>/{blue_mask,response_map,allowed_mask,peak_map}.png``
* ``meta.json`` — detector config + total runtime
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pupa_counter_fresh import DetectorConfig, run_detector  # noqa: E402
from pupa_counter_fresh.eval_instances import (  # noqa: E402
    MatchConfig,
    canonical_scan_number,
    evaluate_disagreement,
    load_teacher_instances,
)


DEFAULT_IMAGE_DIR = Path("/Users/stephenyu/Documents/New project/data/probe_inputs/all_20")
DEFAULT_TEACHER_COUNTS = Path(
    "/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20.csv"
)
DEFAULT_TEACHER_INSTANCES = Path(
    "/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20_instances.csv"
)
DEFAULT_OUT_DIR = Path(
    "/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs"
)


# ---------------------------------------------------------------------------
# Debug / overlay rendering
# ---------------------------------------------------------------------------


def _save_mask(path: Path, mask: np.ndarray) -> None:
    if mask.dtype != np.uint8:
        mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), mask)


def _save_response_map(path: Path, response: np.ndarray) -> None:
    scaled = np.clip(response * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), color)


def _compute_band_y_values(instances: pd.DataFrame) -> dict[str, float] | None:
    """Return the y-positions for the band divider lines, teacher semantics.

    Returns None when there are no instances or the span is zero. **Teacher
    semantics:** the scanned paper's physical top appears at the visual
    BOTTOM of the image (large y). So "top 5%" is the 5% closest to the
    image bottom, and the "top 25%" band occupies the bottom-most quarter
    of the visual image. The "bottom 25%" band sits at the visual image top.
    """
    if instances.empty or "centroid_y" not in instances.columns:
        return None
    y = pd.to_numeric(instances["centroid_y"], errors="coerce").dropna()
    if y.empty:
        return None
    img_top_y = float(y.min())       # visual top of image
    img_bottom_y = float(y.max())    # visual bottom of image
    span = img_bottom_y - img_top_y
    if span <= 0:
        return None
    return {
        "img_top_y": img_top_y,
        "img_bottom_y": img_bottom_y,
        # Lines placed at the teacher's band boundaries:
        "y_top_5pct": img_bottom_y - 0.05 * span,    # top 5% line (near visual image bottom)
        "y_top_25pct": img_bottom_y - 0.25 * span,   # separates middle from teacher's "top"
        "y_bottom_25pct": img_top_y + 0.25 * span,   # separates teacher's "bottom" from middle
    }


def _draw_band_lines(bgr: np.ndarray, band_ys: dict[str, float]) -> None:
    """Draw the 3 horizontal band dividers in cyan, in place.

    Teacher semantics — the ``top 5%`` and ``top 25%`` lines sit near the
    VISUAL image bottom (that's where the physical top of the paper shows
    up after scanning), and the ``bottom 25%`` line sits near the visual
    image top. Lines span the full image width and have a right-edge label.
    Cyan (BGR 255,255,0) is used so the lines don't clash with the yellow
    detection circles.
    """
    h, w = bgr.shape[:2]
    line_color = (255, 255, 0)  # cyan in BGR
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (band_ys["y_bottom_25pct"], "bottom 25%"),
        (band_ys["y_top_25pct"], "top 25%"),
        (band_ys["y_top_5pct"], "top 5%"),
    ]
    for y_f, label in lines:
        y = int(round(y_f))
        if y < 0 or y >= h:
            continue
        cv2.line(bgr, (0, y), (w - 1, y), line_color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
        text_x = w - tw - 14
        text_y = y - 8
        if text_y < th + 4:
            text_y = y + th + 10
        cv2.putText(bgr, label, (text_x, text_y), font, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(bgr, label, (text_x, text_y), font, 0.8, line_color, 2, cv2.LINE_AA)


def _compute_band_counts(instances: pd.DataFrame) -> dict[str, int]:
    """Return the total + per-band counts from an instances frame."""
    if instances.empty:
        return {"n_total": 0, "n_top": 0, "n_middle": 0, "n_bottom": 0, "n_top_5pct": 0}
    bands = instances.get("band", pd.Series("unassigned", index=instances.index)).astype(str)
    top5 = (
        instances.get("is_top_5pct", pd.Series(False, index=instances.index))
        .fillna(False)
        .astype(bool)
    )
    return {
        "n_total": int(len(instances)),
        "n_top": int((bands == "top").sum()),
        "n_middle": int((bands == "middle").sum()),
        "n_bottom": int((bands == "bottom").sum()),
        "n_top_5pct": int(top5.sum()),
    }


def _draw_stats_header(
    bgr: np.ndarray, title: str, band_counts: dict[str, int]
) -> None:
    """Stamp a two-line stats header at the top-left of the image in place.

    Line 1: ``{title}  n={total}``
    Line 2: ``top 25%: A   middle 50%: B   bottom 25%: C   top 5%: D``
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    line1 = f"{title}  n={band_counts['n_total']}"
    line2 = (
        f"top 25%: {band_counts['n_top']}   "
        f"middle 50%: {band_counts['n_middle']}   "
        f"bottom 25%: {band_counts['n_bottom']}   "
        f"top 5%: {band_counts['n_top_5pct']}"
    )
    for y, text in ((42, line1), (82, line2)):
        cv2.putText(bgr, text, (18, y), font, 0.95, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(bgr, text, (18, y), font, 0.95, (0, 255, 255), 2, cv2.LINE_AA)


def _render_overlay(rgb_native: np.ndarray, instances: pd.DataFrame) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_native, cv2.COLOR_RGB2BGR).copy()
    band_ys = _compute_band_y_values(instances)
    if band_ys is not None:
        _draw_band_lines(bgr, band_ys)
    for _, row in instances.iterrows():
        cx = int(round(float(row["centroid_x"])))
        cy = int(round(float(row["centroid_y"])))
        cv2.circle(bgr, (cx, cy), 14, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.circle(bgr, (cx, cy), 2, (0, 180, 255), -1, cv2.LINE_AA)
    _draw_stats_header(bgr, "fresh_peak", _compute_band_counts(instances))
    return bgr


def _render_side_by_side(
    rgb_native: np.ndarray,
    overlay_bgr: np.ndarray,
    instances: pd.DataFrame,
) -> np.ndarray:
    """Stack the (band-lined) original (left) alongside the marked overlay (right).

    Both panels get the 5% / 25% / 75% divider lines and the stats header so
    the user can see which pupae fall into which band. The left panel has no
    detection circles so it's still countable; the right panel has both
    circles and dividers.
    """
    bgr_original = cv2.cvtColor(rgb_native, cv2.COLOR_RGB2BGR).copy()
    band_ys = _compute_band_y_values(instances)
    if band_ys is not None:
        _draw_band_lines(bgr_original, band_ys)
    _draw_stats_header(bgr_original, "original", _compute_band_counts(instances))
    if bgr_original.shape != overlay_bgr.shape:
        raise ValueError(
            f"shape mismatch between original and overlay: {bgr_original.shape} vs {overlay_bgr.shape}"
        )
    return np.hstack([bgr_original, overlay_bgr])


def _write_run_report_xlsx(
    path: Path,
    *,
    joined: pd.DataFrame,
    summary_df: pd.DataFrame,
    cfg: "DetectorConfig",
    run_name: str,
    total_runtime_ms: float,
) -> None:
    """Write the teacher-facing Excel report.

    Three sheets:

    * ``per_image`` — teacher_total, pred_total, per-band counts, matched /
      teacher_only / pred_only, and per-image runtime.
    * ``totals`` — one-row aggregate: recall / precision / F1 and the band
      sums.
    * ``config`` — detector config snapshot so a future reader can tell
      which knobs were active.

    The teacher's final use case is real-time scan → aggregated spreadsheet,
    so this file is the artifact that will actually land on their desk.
    """
    per_image_cols = [
        "image_id",
        "teacher_total",
        "pred_total",
        "abs_delta",
        "n_top",
        "n_middle",
        "n_bottom",
        "n_top_5pct",
        "matched",
        "teacher_only",
        "pred_only",
        "mean_centroid_distance_px",
        "runtime_ms",
    ]
    present_cols = [c for c in per_image_cols if c in joined.columns]
    per_image_sheet = joined.loc[:, present_cols].copy()

    teacher_total = int(
        pd.to_numeric(joined["teacher_total"], errors="coerce").fillna(0).sum()
    )
    pred_total = int(joined["pred_total"].sum())
    matched = int(summary_df["matched"].sum())
    teacher_only = int(summary_df["teacher_only"].sum())
    pred_only = int(summary_df["pred_only"].sum())
    teacher_instances_sum = int(summary_df["teacher_instances"].sum())
    pred_instances_sum = int(summary_df["pred_instances"].sum())
    recall = matched / max(1, teacher_instances_sum)
    precision = matched / max(1, pred_instances_sum)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    def _band_sum(col: str) -> int:
        return int(per_image_sheet[col].sum()) if col in per_image_sheet.columns else 0

    totals_sheet = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "images": int(len(joined)),
                "teacher_total": teacher_total,
                "pred_total": pred_total,
                "matched": matched,
                "teacher_only": teacher_only,
                "pred_only": pred_only,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "n_top_sum": _band_sum("n_top"),
                "n_middle_sum": _band_sum("n_middle"),
                "n_bottom_sum": _band_sum("n_bottom"),
                "n_top_5pct_sum": _band_sum("n_top_5pct"),
                "mean_runtime_ms": round(float(joined["runtime_ms"].mean()), 1),
                "total_runtime_ms": round(float(total_runtime_ms), 1),
            }
        ]
    )

    config_sheet = pd.DataFrame(
        [{"key": k, "value": str(v)} for k, v in asdict(cfg).items()]
    )

    # Teacher-facing flat summary — the sheet the teacher actually opens.
    # Chinese headers, one row per image + one total row at the bottom.
    teacher_rows: list[dict[str, object]] = []
    for _, row in per_image_sheet.iterrows():
        teacher_rows.append(
            {
                "图片编号": row["image_id"],
                "总数": int(row.get("pred_total", 0) or 0),
                "顶部25%": int(row.get("n_top", 0) or 0),
                "中部50%": int(row.get("n_middle", 0) or 0),
                "底部25%": int(row.get("n_bottom", 0) or 0),
                "顶部5%": int(row.get("n_top_5pct", 0) or 0),
                "处理时间ms": round(float(row.get("runtime_ms", 0.0) or 0.0), 1),
            }
        )
    if teacher_rows:
        teacher_rows.append(
            {
                "图片编号": f"总计 ({len(teacher_rows)}张)",
                "总数": int(sum(r["总数"] for r in teacher_rows)),
                "顶部25%": int(sum(r["顶部25%"] for r in teacher_rows)),
                "中部50%": int(sum(r["中部50%"] for r in teacher_rows)),
                "底部25%": int(sum(r["底部25%"] for r in teacher_rows)),
                "顶部5%": int(sum(r["顶部5%"] for r in teacher_rows)),
                "处理时间ms": round(
                    float(sum(float(r["处理时间ms"]) for r in teacher_rows)), 1
                ),
            }
        )
    teacher_summary_sheet = pd.DataFrame(
        teacher_rows,
        columns=[
            "图片编号",
            "总数",
            "顶部25%",
            "中部50%",
            "底部25%",
            "顶部5%",
            "处理时间ms",
        ],
    )

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        teacher_summary_sheet.to_excel(xw, sheet_name="teacher_summary", index=False)
        per_image_sheet.to_excel(xw, sheet_name="per_image", index=False)
        totals_sheet.to_excel(xw, sheet_name="totals", index=False)
        config_sheet.to_excel(xw, sheet_name="config", index=False)

    # CSV sibling so the teacher sheet is also openable without Excel /
    # Numbers / LibreOffice. UTF-8 BOM so Excel on Windows reads Chinese
    # headers correctly if someone does end up opening it there.
    csv_path = path.with_name("teacher_summary.csv")
    teacher_summary_sheet.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Self-contained HTML sibling — opens in any browser, no spreadsheet
    # software required. This is the actual "user double-clicks and sees a
    # nice table" artifact for machines that don't have Numbers / Excel /
    # LibreOffice installed. Styled inline so the file is a single download.
    html_path = path.with_name("teacher_summary.html")
    _write_teacher_summary_html(
        html_path,
        teacher_summary_sheet=teacher_summary_sheet,
        run_name=run_name,
        f1=f1,
        recall=recall,
        precision=precision,
        mean_runtime_ms=float(joined["runtime_ms"].mean()),
        total_runtime_ms=total_runtime_ms,
    )


def _write_teacher_summary_html(
    path: Path,
    *,
    teacher_summary_sheet: pd.DataFrame,
    run_name: str,
    f1: float,
    recall: float,
    precision: float,
    mean_runtime_ms: float,
    total_runtime_ms: float,
) -> None:
    """Render the teacher summary as a self-contained styled HTML page.

    Output is one file, no external CSS / JS / fonts — opens in any browser
    (Safari / Chrome / Firefox) by double-clicking. Chinese font fallbacks
    are in the font-stack so headers always render correctly even without
    specific fonts installed.
    """
    # Split off the totals row (last row, has "总计" in image_id column)
    if len(teacher_summary_sheet) > 0 and "总计" in str(
        teacher_summary_sheet.iloc[-1].get("图片编号", "")
    ):
        data_rows = teacher_summary_sheet.iloc[:-1]
        totals_row = teacher_summary_sheet.iloc[-1]
    else:
        data_rows = teacher_summary_sheet
        totals_row = None

    headers = list(teacher_summary_sheet.columns)

    def _fmt_cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    def _esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    thead_cells = "".join(f"<th>{_esc(str(h))}</th>" for h in headers)
    body_rows_html: list[str] = []
    for _, row in data_rows.iterrows():
        cells = "".join(f"<td>{_esc(_fmt_cell(row[h]))}</td>" for h in headers)
        body_rows_html.append(f"<tr>{cells}</tr>")
    if totals_row is not None:
        totals_cells = "".join(
            f"<td>{_esc(_fmt_cell(totals_row[h]))}</td>" for h in headers
        )
        body_rows_html.append(f'<tr class="totals">{totals_cells}</tr>')
    tbody_html = "\n      ".join(body_rows_html)

    n_images = int(len(data_rows))
    total_pupae = (
        int(pd.to_numeric(data_rows.get("总数", pd.Series([])), errors="coerce").fillna(0).sum())
    )
    total_runtime_s = total_runtime_ms / 1000.0

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pupa Counter 汇总 — {_esc(run_name)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue",
                 Arial, sans-serif;
    background: #f4f5f7;
    margin: 0;
    padding: 40px 20px;
    color: #1f2328;
  }}
  .container {{
    max-width: 960px;
    margin: 0 auto;
    background: #ffffff;
    padding: 36px 44px;
    border-radius: 10px;
    box-shadow: 0 2px 18px rgba(15, 23, 42, 0.08);
  }}
  h1 {{
    margin: 0 0 6px 0;
    font-size: 22px;
    color: #0f172a;
    font-weight: 600;
  }}
  .subtitle {{
    color: #64748b;
    margin-bottom: 24px;
    font-size: 13px;
  }}
  .stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 22px;
    padding: 18px 22px;
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    margin-bottom: 28px;
  }}
  .stat-item {{
    display: flex;
    flex-direction: column;
    min-width: 96px;
  }}
  .stat-label {{
    color: #64748b;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 4px;
  }}
  .stat-value {{
    font-size: 18px;
    font-weight: 600;
    color: #0f172a;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  thead th {{
    background: #3b82f6;
    color: #ffffff;
    text-align: right;
    padding: 11px 14px;
    font-weight: 500;
    white-space: nowrap;
  }}
  thead th:first-child {{ text-align: left; border-top-left-radius: 6px; }}
  thead th:last-child  {{ border-top-right-radius: 6px; }}
  tbody td {{
    padding: 9px 14px;
    border-bottom: 1px solid #e5e7eb;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  tbody td:first-child {{
    text-align: left;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    color: #334155;
  }}
  tbody tr:nth-child(even) td {{ background: #f9fafb; }}
  tbody tr:hover td {{ background: #eff6ff; }}
  tbody tr.totals td {{
    background: #fef3c7 !important;
    font-weight: 600;
    color: #78350f;
    border-top: 2px solid #f59e0b;
    border-bottom: none;
  }}
  .footer {{
    margin-top: 28px;
    padding-top: 18px;
    border-top: 1px solid #e5e7eb;
    font-size: 11px;
    color: #94a3b8;
    line-height: 1.6;
  }}
  .footer code {{
    background: #f1f5f9;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 10px;
    color: #475569;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Pupa Counter 汇总报告</h1>
  <div class="subtitle">运行 <b>{_esc(run_name)}</b> · 每张图片分带数据(顶部25% / 中部50% / 底部25% / 顶部5%) · 方向按物理扫描纸: 顶部 = 图片视觉下方</div>
  <div class="stats">
    <div class="stat-item"><span class="stat-label">图片数</span><span class="stat-value">{n_images}</span></div>
    <div class="stat-item"><span class="stat-label">pupa 总数</span><span class="stat-value">{total_pupae}</span></div>
    <div class="stat-item"><span class="stat-label">平均 / 张</span><span class="stat-value">{mean_runtime_ms:.1f} ms</span></div>
    <div class="stat-item"><span class="stat-label">总耗时</span><span class="stat-value">{total_runtime_s:.2f} s</span></div>
    <div class="stat-item"><span class="stat-label">F1</span><span class="stat-value">{f1:.3f}</span></div>
    <div class="stat-item"><span class="stat-label">Recall</span><span class="stat-value">{recall:.3f}</span></div>
    <div class="stat-item"><span class="stat-label">Precision</span><span class="stat-value">{precision:.3f}</span></div>
  </div>
  <table>
    <thead><tr>{thead_cells}</tr></thead>
    <tbody>
      {tbody_html}
    </tbody>
  </table>
  <div class="footer">
    数据源同目录下的 <code>run_report.xlsx</code> · sheet <code>teacher_summary</code><br>
    检测器: fresh_peak_v2 (adaptive response + component split) · 字段 <code>顶部5%</code> 即 <code>顶部25%</code> 中最顶端的 5%
  </div>
</div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_debug(
    debug: dict[str, np.ndarray], out_dir: Path, image_id: str
) -> None:
    image_debug_dir = out_dir / "debug" / image_id
    image_debug_dir.mkdir(parents=True, exist_ok=True)
    if "blue_mask" in debug:
        _save_mask(image_debug_dir / "blue_mask.png", debug["blue_mask"])
    if "allowed_mask" in debug:
        _save_mask(image_debug_dir / "allowed_mask.png", debug["allowed_mask"])
    if "peak_map" in debug:
        _save_mask(image_debug_dir / "peak_map.png", debug["peak_map"])
    if "response_map" in debug:
        _save_response_map(image_debug_dir / "response_map.png", debug["response_map"])


# ---------------------------------------------------------------------------
# Per-image driver
# ---------------------------------------------------------------------------


def process_image(
    image_path: Path,
    cfg: DetectorConfig,
    out_dir: Path,
) -> dict:
    stem = image_path.stem
    image_id = canonical_scan_number(stem) or stem
    result = run_detector(image_path, image_id=image_id, cfg=cfg, keep_debug=True)

    # Save overlay + side-by-side + debug artifacts
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_dir = out_dir / "side_by_side"
    side_by_side_dir.mkdir(parents=True, exist_ok=True)
    overlay = _render_overlay(result.debug["rgb_native"], result.instances)
    cv2.imwrite(str(overlays_dir / f"{image_id}_overlay.png"), overlay)
    side_by_side = _render_side_by_side(result.debug["rgb_native"], overlay, result.instances)
    cv2.imwrite(str(side_by_side_dir / f"{image_id}_side_by_side.png"), side_by_side)
    _write_debug(result.debug, out_dir, image_id)

    return {
        "image_id": image_id,
        "source_path": str(image_path),
        "pred_total": int(len(result.instances)),
        "runtime_ms": float(result.runtime_ms),
        "instances": result.instances,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fresh peak-first detector on 20 images.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--teacher-counts", type=Path, default=DEFAULT_TEACHER_COUNTS)
    parser.add_argument("--teacher-instances", type=Path, default=DEFAULT_TEACHER_INSTANCES)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--work-scale", type=float, default=0.67)
    parser.add_argument("--peak-threshold", type=float, default=0.22)
    parser.add_argument("--peak-min-distance", type=int, default=10)
    parser.add_argument("--allowed-threshold", type=float, default=0.12)
    parser.add_argument("--detector-backend", type=str, default="fresh_peak_v0")
    parser.add_argument("--instance-source", type=str, default="fresh_peak_v0")
    parser.add_argument(
        "--use-component-split",
        action="store_true",
        help="Enable the v1 per-component peak splitter for touching pupae.",
    )
    parser.add_argument("--component-single-pupa-area", type=float, default=200.0)
    parser.add_argument("--component-area-ratio", type=float, default=1.20)
    parser.add_argument("--component-min-distance", type=int, default=3)
    parser.add_argument("--component-threshold", type=float, default=0.18)
    parser.add_argument("--component-min-area", type=int, default=60)
    parser.add_argument("--component-max-peaks", type=int, default=20)
    parser.add_argument(
        "--response-mode",
        type=str,
        default="smooth",
        choices=["smooth", "log", "dog", "adaptive"],
        help="Response map computation strategy (v2 response-sharpening).",
    )
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--log-sigma", type=float, default=1.0)
    parser.add_argument("--dog-sigma-low", type=float, default=0.8)
    parser.add_argument("--dog-sigma-high", type=float, default=2.0)
    parser.add_argument("--adaptive-small-sigma", type=float, default=0.6)
    parser.add_argument("--adaptive-large-sigma", type=float, default=1.4)
    parser.add_argument("--adaptive-area-threshold", type=int, default=500)
    parser.add_argument(
        "--use-paper-roi",
        action="store_true",
        help="Enable paper ROI masking to suppress scanner-edge false positives.",
    )
    parser.add_argument("--paper-roi-brightness-threshold", type=int, default=180)
    parser.add_argument("--paper-roi-close-kernel", type=int, default=15)
    parser.add_argument("--paper-roi-erode-margin", type=int, default=6)
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        parser.error(f"image dir not found: {args.image_dir}")

    run_name = args.run_name or f"{args.detector_backend}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(exist_ok=True)
    (run_dir / "debug").mkdir(exist_ok=True)

    cfg = DetectorConfig(
        work_scale=args.work_scale,
        smooth_sigma=args.smooth_sigma,
        peak_abs_score_threshold=args.peak_threshold,
        peak_min_distance_px=args.peak_min_distance,
        allowed_abs_threshold=args.allowed_threshold,
        use_component_split=args.use_component_split,
        component_single_pupa_area_px=args.component_single_pupa_area,
        component_area_ratio_threshold=args.component_area_ratio,
        component_min_peak_distance_px=args.component_min_distance,
        component_abs_score_threshold=args.component_threshold,
        component_min_component_area_px=args.component_min_area,
        component_max_peaks=args.component_max_peaks,
        response_mode=args.response_mode,
        log_sigma=args.log_sigma,
        dog_sigma_low=args.dog_sigma_low,
        dog_sigma_high=args.dog_sigma_high,
        adaptive_small_sigma=args.adaptive_small_sigma,
        adaptive_large_sigma=args.adaptive_large_sigma,
        adaptive_area_threshold_px=args.adaptive_area_threshold,
        use_paper_roi=args.use_paper_roi,
        paper_roi_brightness_threshold=args.paper_roi_brightness_threshold,
        paper_roi_close_kernel_px=args.paper_roi_close_kernel,
        paper_roi_erode_margin_px=args.paper_roi_erode_margin,
        detector_backend=args.detector_backend,
        instance_source=args.instance_source,
    )

    images = sorted(args.image_dir.glob("*.png"))
    if not images:
        parser.error(f"no PNG images under {args.image_dir}")

    t_start = time.perf_counter()
    per_image: List[dict] = []
    all_instances: List[pd.DataFrame] = []
    for image_path in images:
        print(f"  . {image_path.name}", flush=True)
        row = process_image(image_path, cfg, run_dir)
        per_image.append(row)
        if not row["instances"].empty:
            all_instances.append(row["instances"])
    total_runtime = (time.perf_counter() - t_start) * 1000.0

    # instances.csv (built first so per-band counts can be folded into counts.csv)
    instances_df = (
        pd.concat(all_instances, ignore_index=True) if all_instances else pd.DataFrame()
    )
    instances_df.to_csv(run_dir / "instances.csv", index=False)

    # Per-image band aggregation (top 25 / middle 50 / bottom 25 / top 5%)
    if not instances_df.empty:
        bands = instances_df["band"].astype(str)
        band_per_image = (
            pd.DataFrame(
                {
                    "image_id": instances_df["image_id"].astype(str),
                    "_is_top": bands.eq("top").astype(int),
                    "_is_middle": bands.eq("middle").astype(int),
                    "_is_bottom": bands.eq("bottom").astype(int),
                    "_is_top5": instances_df["is_top_5pct"].fillna(False).astype(bool).astype(int),
                }
            )
            .groupby("image_id", as_index=False)[
                ["_is_top", "_is_middle", "_is_bottom", "_is_top5"]
            ]
            .sum()
            .rename(
                columns={
                    "_is_top": "n_top",
                    "_is_middle": "n_middle",
                    "_is_bottom": "n_bottom",
                    "_is_top5": "n_top_5pct",
                }
            )
        )
    else:
        band_per_image = pd.DataFrame(
            columns=["image_id", "n_top", "n_middle", "n_bottom", "n_top_5pct"]
        )

    # counts.csv (teacher totals + per-band breakdown)
    counts_df = pd.DataFrame(
        [
            {
                "image_id": row["image_id"],
                "source_path": row["source_path"],
                "pred_total": row["pred_total"],
                "runtime_ms": row["runtime_ms"],
            }
            for row in per_image
        ]
    )

    teacher_counts = pd.read_csv(args.teacher_counts)
    teacher_counts["match_key"] = teacher_counts["image_id"].map(canonical_scan_number)
    counts_df["match_key"] = counts_df["image_id"].map(canonical_scan_number)
    counts_df = counts_df.merge(
        teacher_counts[["match_key", "n_pupa_final"]].rename(
            columns={"n_pupa_final": "teacher_total"}
        ),
        on="match_key",
        how="left",
    )
    counts_df = counts_df.merge(band_per_image, on="image_id", how="left")
    for col in ("n_top", "n_middle", "n_bottom", "n_top_5pct"):
        counts_df[col] = counts_df[col].fillna(0).astype(int)
    counts_df["abs_delta"] = (counts_df["pred_total"] - counts_df["teacher_total"]).abs()
    counts_df.to_csv(run_dir / "counts.csv", index=False)

    # Teacher-based disagreement
    teacher_instances = load_teacher_instances(args.teacher_instances)
    summary_df, matches_df, teacher_only_df, pred_only_df = evaluate_disagreement(
        instances_df, teacher_instances, cfg=MatchConfig()
    )
    summary_df.to_csv(run_dir / "run_summary.csv", index=False)
    matches_df.to_csv(run_dir / "matches_vs_teacher.csv", index=False)
    teacher_only_df.to_csv(run_dir / "teacher_only_instances.csv", index=False)
    pred_only_df.to_csv(run_dir / "pred_only_instances.csv", index=False)

    # disagreement_vs_teacher.csv (per-image join of counts + disagreement)
    joined = counts_df.merge(
        summary_df[
            [
                "match_key",
                "teacher_instances",
                "pred_instances",
                "matched",
                "teacher_only",
                "pred_only",
                "mean_centroid_distance_px",
            ]
        ],
        on="match_key",
        how="left",
    )
    joined.to_csv(run_dir / "disagreement_vs_teacher.csv", index=False)

    # run_report.xlsx — teacher-facing summary (per_image / totals / config)
    _write_run_report_xlsx(
        run_dir / "run_report.xlsx",
        joined=joined,
        summary_df=summary_df,
        cfg=cfg,
        run_name=run_name,
        total_runtime_ms=total_runtime,
    )

    # meta.json
    meta = {
        "run_name": run_name,
        "detector_backend": cfg.detector_backend,
        "instance_source": cfg.instance_source,
        "config": asdict(cfg),
        "total_runtime_ms": total_runtime,
        "mean_runtime_ms": float(counts_df["runtime_ms"].mean()),
        "n_images": int(len(counts_df)),
        "totals": {
            "teacher": int(pd.to_numeric(counts_df["teacher_total"], errors="coerce").sum()),
            "pred": int(counts_df["pred_total"].sum()),
            "matched": int(summary_df["matched"].sum()),
            "teacher_only": int(summary_df["teacher_only"].sum()),
            "pred_only": int(summary_df["pred_only"].sum()),
            "recall": (
                float(summary_df["matched"].sum() / max(1, summary_df["teacher_instances"].sum()))
            ),
            "precision": (
                float(summary_df["matched"].sum() / max(1, summary_df["pred_instances"].sum()))
            ),
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Terminal summary
    print()
    print(f"== {run_name} ==")
    print(f"images           : {meta['n_images']}")
    print(f"mean runtime_ms  : {meta['mean_runtime_ms']:.1f}")
    print(f"teacher total    : {meta['totals']['teacher']}")
    print(f"pred total       : {meta['totals']['pred']}")
    print(f"matched          : {meta['totals']['matched']}")
    print(f"teacher_only     : {meta['totals']['teacher_only']}")
    print(f"pred_only        : {meta['totals']['pred_only']}")
    print(f"recall           : {meta['totals']['recall']:.3f}")
    print(f"precision        : {meta['totals']['precision']:.3f}")
    print(f"output dir       : {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
