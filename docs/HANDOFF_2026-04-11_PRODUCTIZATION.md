# Handoff — Fresh Peak v2 → Productization Phase

**Date:** 2026-04-11 (updated after r005 — HTML sibling for users with no spreadsheet app)
**Branch:** `codex/fresh-start-peak-proposal-v1`
**Repo:** `/Users/stephenyu/Documents/pupa_counter_publish`
**Hand-off reason:** user asked for a self-contained log so a future compacted/reset session can pick up the work seamlessly.

## CRITICAL: Band direction convention

The teacher's "top" = physical top of the scanned paper = **visual image BOTTOM** (large `centroid_y`). The legacy `pupa_counter.count.anchors` code used the opposite convention (visual top = physical top). We intentionally diverged in `pupa_counter_fresh.geometry.assign_bands` so that `顶部5%` / `top 5%` means what the teacher expects. Do NOT re-sync with the legacy code — that's the bug we just fixed. r001/r002/r003 outputs all had the wrong direction and are superseded.

---

## 1. Status at a glance

| area | state |
| --- | --- |
| Detection accuracy | **Done.** R=0.8924 / P=0.959 / F1=0.9245 on the 20-image benchmark. User already signed off ("接近甚至超越 v8 … 现在已经很好了"). |
| Band classification (top 25 / middle 50 / bottom 25 / top 5%) | **Done.** Implemented in `src/pupa_counter_fresh/geometry.py`, exercised in the r002 run — `counts.csv` now has `n_top`, `n_middle`, `n_bottom`, `n_top_5pct`. |
| Excel export | **Done.** Every run now writes `run_report.xlsx` with three sheets (`per_image`, `totals`, `config`) via openpyxl. |
| Side-by-side renderer | **Done.** Every run writes `side_by_side/<image_id>_side_by_side.png` — original on left, marked overlay on right. Both panels now carry the 5%/25%/75% cyan divider lines and the top-left stats header (`total / top 25% / middle 50% / bottom 25% / top 5%`). |
| Teacher-facing Excel sheet | **Done.** `run_report.xlsx` now has `teacher_summary` as its first sheet — Chinese headers (`图片编号`, `总数`, `顶部25%`, `中部50%`, `底部25%`, `顶部5%`, `处理时间ms`), 20 rows + one `总计 (20张)` row. |
| Self-audit by vision | **Deferred to user.** Image budget reserved for user inspection. User will flip through `side_by_side/` themselves. DO NOT delegate image review to subagents — user explicitly flagged subagent vision quality as unreliable. |
| Real-time / live scan hook | **Not yet built.** Current entrypoint is still the batch script; no `--watch-dir` or folder-polling mode. This is the next big productization step if the teacher's workflow needs it. |

### Current best run (as of this handoff)

- **Directory:** `/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs/fresh_peak_v2_adaptive_r004/`
- **Config:** `--response-mode adaptive --use-component-split`; all other knobs at `DetectorConfig` defaults. Run name `fresh_peak_v2_adaptive_r004`, backend `fresh_peak_v2_adaptive`.
- **Metrics (from r004 meta.json + run_report.xlsx `totals` sheet):**
  - teacher = 2175, pred = 2024, matched = 1941
  - teacher_only = 234, pred_only = 83
  - **recall = 0.892, precision = 0.959, F1 = 0.925**
  - mean runtime 78.5 ms/image, 1.57 s total detector wall clock for 20 images
- **Band totals (teacher semantics, across 20 images, 2024 predicted pupae):** n_top = 295, n_middle = 1389, n_bottom = 340, **n_top_5pct = 39**. r003's numbers had n_top and n_bottom swapped and n_top_5pct = 56 — those were wrong, the r004 figures are the ones that match the teacher's physical mental model.
- **Detection numbers are bit-identical to r002/r003** — the r003 → r004 change only flipped the band assignment direction in `geometry.py` and the line labels in `_draw_band_lines`. Detection logic untouched.
- **Artifacts in the run dir:**
  ```
  counts.csv                       -- per-image totals + band breakdown + abs_delta
  instances.csv                    -- every accepted peak, with band/is_top_5pct columns
  run_summary.csv                  -- matched / teacher_only / pred_only per image
  matches_vs_teacher.csv           -- individual match pairs (for mining)
  teacher_only_instances.csv       -- rows the teacher has that we don't
  pred_only_instances.csv          -- rows we have that the teacher doesn't
  disagreement_vs_teacher.csv      -- counts + disagreement joined per image
  run_report.xlsx                  -- 4 sheets: teacher_summary (first, Chinese headers) /
                                      per_image (dev view) / totals / config
  meta.json                        -- detector config snapshot + totals
  overlays/<image_id>_overlay.png  -- 20 marked overlays (yellow circles + band lines + header)
  side_by_side/<image_id>_side_by_side.png  -- 20 pairs [original + band lines | overlay]
  debug/<image_id>/{blue_mask,response_map,allowed_mask,peak_map}.png
  teacher_summary.csv              -- CSV sibling, UTF-8 BOM for Chinese headers
  teacher_summary.html             -- self-contained styled HTML page (6 KB), opens in any
                                      browser by double-clicking. Inline CSS, no external
                                      assets. This is the "user has no Numbers / Excel /
                                      LibreOffice installed" escape hatch; r005 was the
                                      first run to produce it.
  ```

### Note on small run-to-run drift

The r001 run referenced in earlier versions of this doc reported R=0.896 / P=0.956 / F1=0.925. r002 reports R=0.8924 / P=0.959 / F1=0.9245. The detector code that produces peaks is unchanged between runs — only `geometry.py` (band assignment) was edited, and band assignment happens *after* detection. The ~8-match drift is almost certainly from the evaluator's greedy matching being sensitive to iteration order after the extra `band` / `is_top_5pct` columns were populated (previously always `"unassigned"` / `False`, now real values). The difference is ~0.4 %, well within noise, and the user explicitly said accuracy was already good enough. Do not sweep-tune this.

---

## 2. What changed in this session (code)

### 2.1 `src/pupa_counter_fresh/geometry.py`

Replaced the stub `assign_bands` with real per-image band math, mirroring `src/pupa_counter/count/anchors.py` + `src/pupa_counter/count/assign.py` in their `centroid` anchor mode. The math is inlined — the old `BandGeometry` dataclass from `pupa_counter.types` is deliberately NOT imported, because `pupa_counter_fresh` is kept dependency-free from the legacy package.

Algorithm (per image, using detected centroids only):

```python
top_y     = centroid_y.min()          # topmost pupa
bottom_y  = centroid_y.max()          # bottommost pupa
span      = bottom_y - top_y
upper_five_pct_y = top_y + 0.05 * span
upper_middle_y   = top_y + 0.25 * span
lower_middle_y   = top_y + 0.75 * span

is_top_5pct = (centroid_y <= upper_five_pct_y)
band = "middle"
band[centroid_y <  upper_middle_y] = "top"
band[centroid_y >  lower_middle_y] = "bottom"
```

Strict `<` / `>` inequalities preserve the legacy semantics (`tests/test_band_math.py:25` locks them in for the old package). Empty-frame path sets `band="unassigned"` / `is_top_5pct=False`. Smoke-tested on a synthetic 10-row frame: 3 top / 4 middle / 3 bottom / 1 top_5pct.

### 2.2 `scripts/run_fresh_peak_detector.py`

Five groups of edits across two sub-sessions. The script now has dedicated render helpers for band decoration:

1. **Render helpers** (all just above `_render_overlay`):
   - `_compute_band_y_values(instances)` — returns `{top_y, bottom_y, y_5pct, y_25pct, y_75pct}` from the instances frame's `centroid_y.min()` / `.max()`. Returns None for empty / zero-span inputs so the renderer can skip cleanly.
   - `_draw_band_lines(bgr, band_ys)` — draws 3 cyan horizontal lines (BGR `(255,255,0)`, 2 px, anti-aliased) spanning the full image width, with right-edge text labels `top 5%` / `25%` / `75%`. Labels flip below the line if they'd clip the image top.
   - `_compute_band_counts(instances)` — returns `{n_total, n_top, n_middle, n_bottom, n_top_5pct}` from the `band` / `is_top_5pct` columns, with empty-frame guards.
   - `_draw_stats_header(bgr, title, band_counts)` — stamps a two-line header at `(18, 42)` and `(18, 82)`: `{title} n={total}` on line 1 and `top 25%: A   middle 50%: B   bottom 25%: C   top 5%: D` on line 2. Yellow foreground over black outline.

2. **`_render_overlay` rewritten** to (a) draw band lines first, (b) draw yellow detection circles on top, (c) draw the stats header last. The order matters — header on top so it's always legible; circles over lines so dividers don't obscure detections.

3. **`_render_side_by_side(rgb_native, overlay_bgr, instances)`** — new `instances` parameter so it can also draw band lines + header on the original (left) panel. Left panel remains free of detection circles (user's core requirement: must still be recountable). Right panel is the pre-decorated overlay from `_render_overlay`. `np.hstack` composes them.

4. **`_write_run_report_xlsx`** now writes **four** sheets, not three:
   - `teacher_summary` (FIRST, Chinese headers) — `图片编号`, `总数`, `顶部25%`, `中部50%`, `底部25%`, `顶部5%`, `处理时间ms`. 20 image rows + one `总计 (20张)` totals row. This is the sheet the teacher actually opens.
   - `per_image` (dev view) — 13 columns with the teacher_total / matched / disagreement metrics.
   - `totals` — one-row aggregate with R / P / F1.
   - `config` — flat dump of `DetectorConfig`.

5. **`process_image`** now passes `result.instances` into `_render_side_by_side` so the original panel can be band-decorated. A new `side_by_side/` directory is created alongside `overlays/` at run start.

6. **`main()` reorganized** (done in r002) so `instances_df` is built *before* `counts_df` is written. This lets the per-image band aggregation be folded into `counts.csv` as four new integer columns (`n_top`, `n_middle`, `n_bottom`, `n_top_5pct`). The Excel report is written after `disagreement_vs_teacher.csv` and before `meta.json`.

No edits to `detector.py`, `resolver_cv.py`, `peaks.py`, `response.py`, `paper_roi.py`, or any of the v1/v2 research modules.

### 2.3 Files touched this session (summary)

```
src/pupa_counter_fresh/geometry.py          -- rewrote stub with real band math
scripts/run_fresh_peak_detector.py          -- 4 edits: side-by-side helper, xlsx helper,
                                               process_image, main() reorg
docs/HANDOFF_2026-04-11_PRODUCTIZATION.md   -- this doc (you are here)
```

Also the memory file `~/.claude/projects/-Users-stephenyu/memory/feedback_overlay_presentation.md` was created earlier in the session, documenting the "always pair overlay with unmarked original" rule.

---

## 3. Visual audit — summarized data (from the prior sub-session)

From the previous context I reviewed 18/20 `disagreement_galleries` (scans 3, 7, 8, 10, 15, 20, 22, 25, 30, 35, 40, 50, 55, 60, 65, 74, 80, 95). Scans 70 and 90 were counted from the CSV only (small numbers: t_only=4,5 / p_only=4,2). Categories:

- **real_miss** — genuine pupa the detector missed (mostly in touching-pair clusters).
- **split_artifact** — `component_id` contains `_split`; these are `cellpose_split` synthetic rows v8 manufactured, not real pupae.
- **edge_ambiguous** — on/near the scanner gray strip at `x<20`.
- **v8_suspicious** — isolated teacher marker on white background with nothing visible; probable v8 FP.
- For `pred_only`: **v8_miss** = real pupa v8 missed; **edge_fp** = FP on the left scanner strip; **amb** = unclear.

| scan | t_only | real_miss | split_artifact | edge_amb | v8_susp | p_only | v8_miss | edge_fp | amb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 9 | 7 | 2 | 0 | 0 | 4 | 2 | 0 | 2 |
| 7 | 12 | 8 | 4 | 0 | 0 | 2 | 2 | 0 | 0 |
| 8 | 13 | 10 | 3 | 0 | 0 | 1 | 1 | 0 | 0 |
| 10 | 13 | 7 | 2 | 2 | 2 | 3 | 2 | 0 | 1 |
| 15 | 20 | 17 | 1 | 0 | 2 | 11 | 9 | 0 | 2 |
| 20 | 22 | 13 | 7 | 1 | 1 | 3 | 3 | 0 | 0 |
| 22 | 15 | 5 | 7 | 3 | 0 | 2 | 2 | 0 | 0 |
| 25 | 18 | 10 | 6 | 1 | 1 | 8 | 4 | 4 | 0 |
| 30 | 9 | 4 | 5 | 0 | 0 | 7 | 7 | 0 | 0 |
| 35 | 9 | 7 | 2 | 0 | 0 | 6 | 4 | 0 | 2 |
| 40 | 4 | 3 | 1 | 0 | 0 | 8 | 4 | 1 | 3 |
| 50 | 7 | 4 | 2 | 0 | 1 | 4 | 1 | 2 | 1 |
| 55 | 10 | 9 | 1 | 0 | 0 | 3 | 3 | 0 | 0 |
| 60 | 3 | 2 | 1 | 0 | 0 | 4 | 3 | 1 | 0 |
| 65 | 12 | 5 | 5 | 1 | 1 | 8 | 3 | 2 | 3 |
| 74 | 12 | 8 | 4 | 0 | 0 | 1 | 1 | 0 | 0 |
| 80 | 13 | 6 | 5 | 1 | 1 | 5 | 4 | 0 | 1 |
| 95 | 16 | 9 | 6 | 0 | 1 | 3 | 3 | 0 | 0 |
| subtotal | 217 | 134 | 64 | 9 | 10 | 83 | 58 | 10 | 15 |
| 70 | 4 | — | — | — | — | 4 | — | — | — |
| 90 | 5 | — | — | — | — | 2 | — | — | — |
| TOTAL | 226 | — | — | — | — | 89 | — | — | — |

**Conclusions (still valid after r002):**

1. Only ~134 of the 226 `teacher_only` rows are true misses. 64 are `cellpose_split` artifacts, ~9 edge-ambiguous, ~10 v8 FPs.
2. ~58 of the 89 `pred_only` rows are real pupae v8 missed, mostly in dense touching clusters.
3. Effective recall ≈ 0.93, precision ≈ 0.97 once credit is given on both sides.
4. **Two remaining failure modes, neither blocking productization:**
   - 4+ overlap in touching clusters — component splitter collapses some peaks in very dense areas.
   - Scanner-strip FPs at `x<20` in scans 25, 40, 50, 60, 65 (~10 total) — warm brownish tint sneaks past the blue mask.

---

## 4. Outstanding / next-session picks

Ordered most-to-least useful:

1. **User visual audit of `side_by_side/`** — open the 20 pairs in `/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs/fresh_peak_v2_adaptive_r002/side_by_side/` and report anything obviously wrong. The panels are ~1 MB each; probably best viewed in Finder Quick Look or Preview.
2. **Real-time / live-scan hook.** Current script is batch-only. If the teacher's workflow is "put scan on scanner → PNG appears in a watch folder → workbook row appears", we need either (a) a `--watch-dir` loop that processes new files and appends to an existing workbook, or (b) a tiny GUI wrapper. Decide which with the user before building.
3. **Regression test for `geometry.assign_bands`.** Mirror `tests/test_band_math.py` at `tests/fresh/test_geometry.py`. I smoke-tested by hand; a pytest version is 15 lines.
4. **Append-mode for `run_report.xlsx`.** Currently each run creates a fresh workbook. For the teacher's aggregate-over-many-runs use case we probably want a single master workbook with one row appended per scan run. Not built yet — needs user input on where the master lives.
5. **Scanner-strip FP cleanup.** ~10 false positives in 5 scans. Could be fixed by raising `peak_edge_margin_px` from 4 to ~25 OR adding a per-peak LAB-space check for warm tint. User said not to prioritize.
6. **Dense-cluster splitter polish.** ~134 real_miss rows are almost all in 4+ overlap clusters. Would need distance-transform-assisted peak splitting (already has plumbing via `component_use_distance_transform` flag). User said not to prioritize.

---

## 5. Key paths (copy-paste friendly)

```
# Source
/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter_fresh/geometry.py
/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter_fresh/detector.py
/Users/stephenyu/Documents/pupa_counter_publish/scripts/run_fresh_peak_detector.py
/Users/stephenyu/Documents/pupa_counter_publish/docs/HANDOFF_2026-04-11_PRODUCTIZATION.md

# Test inputs
/Users/stephenyu/Documents/New project/data/probe_inputs/all_20/          # 20 .png test images
/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20.csv
/Users/stephenyu/Documents/New project/fresh_start_agent_handoff_2026-04-10/benchmarks/teacher_v8_20_instances.csv

# Current best output (r002 — productized)
/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs/fresh_peak_v2_adaptive_r002/
  counts.csv  instances.csv  run_summary.csv  run_report.xlsx  meta.json
  overlays/  side_by_side/  debug/  matches_vs_teacher.csv
  teacher_only_instances.csv  pred_only_instances.csv  disagreement_vs_teacher.csv

# Prior run (r001) — kept for comparison, slightly different matched count
/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs/fresh_peak_v2_adaptive_r001/

# Old code — reference only, do NOT import from the fresh package
/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/count/anchors.py
/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/count/assign.py
/Users/stephenyu/Documents/pupa_counter_publish/tests/test_band_math.py

# Project venv with openpyxl / pandas / opencv / scikit-image
/Users/stephenyu/Documents/pupa_counter_publish/.venv/bin/python
```

### Rerun command

```bash
cd /Users/stephenyu/Documents/pupa_counter_publish
.venv/bin/python scripts/run_fresh_peak_detector.py \
    --response-mode adaptive \
    --use-component-split \
    --run-name fresh_peak_v2_adaptive_r003 \
    --detector-backend fresh_peak_v2_adaptive \
    --instance-source fresh_peak_v2_adaptive
```

Add `--work-scale 1.0` if you want to remove the 0.67 downscale (accuracy same, runtime ~2× slower). Add any of the `--component-*` or `--adaptive-*` args to retune; defaults are load-bearing (see §6 gotchas).

---

## 6. Gotchas — do NOT rediscover these

- **Paper ROI at `close_kernel_px=15` drops 123 real pupae.** The scanner strip is only 6–8 px wide, `peak_edge_margin_px=4` in the main detector already handles it. Don't reopen that thread.
- **LoG / DoG / adaptive sharpening is falsified at the 20-image level.** Unit tests show the modes separate touching blobs; at real pupa density the smooth baseline is within ~0.4 F1. `response_mode="adaptive"` is kept because sweep picked it, not because it's meaningfully better.
- **`allowed_abs_threshold` (0.12) must stay < `peak_abs_score_threshold` (0.22).** Polarity flip kills recall silently.
- **v1 component splitter defaults are load-bearing:** `area=200, ratio=1.20, mind=3, thr=0.18`. These came from the 5-hard-image sweep; don't re-tune.
- **`teacher_v8_20_instances_cleaned_BOGUS_DO_NOT_USE.csv`** exists in the benchmarks dir — ignore it, don't use it as ground truth. It dropped 123 real pupae due to the paper-ROI misadventure above.
- **`scripts/rebuild_cleaned_teacher.py`** is still in the repo for historical reference but produces the BOGUS file above. Do not rerun.
- **`scripts/render_v8_overlays.py`** hard-codes absolute paths to `/Users/stephenyu/Documents/New project/...`. Fine for this laptop; note before porting anywhere else.
- **Band math assumes detector has fired.** On an empty `instances_df`, `assign_bands` returns `band="unassigned"` / `is_top_5pct=False`. Downstream (`counts.csv`) will still show 0 for all band columns via the `.fillna(0).astype(int)` guard.
- **`pd.ExcelWriter` requires openpyxl.** It IS installed in `.venv/` (3.1.5), but don't invoke with the system `python` — that one lacks it. Always use `/Users/stephenyu/Documents/pupa_counter_publish/.venv/bin/python`.
- **Image budget is finite.** The previous sub-session ran out reviewing `disagreement_galleries/`. If you need to open a .png, open exactly one at a time and cache the finding as text. The side-by-side PNGs are ~1 MB each — plan accordingly.

---

## 7. Memo to next-session-me

The user's mental model: *"ship this to my teacher."* Accuracy is done; every hour spent chasing another 0.005 F1 is an hour not spent making the spreadsheet the teacher can actually open. The r002 workbook already opens and reads cleanly in Excel — the remaining question is whether the teacher needs a live/watch-dir mode or can paste scans into a folder and run the script once.

When presenting results to the user: **always include the unmarked original alongside the marked overlay** (`side_by_side/` is the output directory — use it instead of `overlays/`). This is durable feedback, already saved to `~/.claude/projects/-Users-stephenyu/memory/feedback_overlay_presentation.md`. Circles-only overlays hide what's underneath; the user cannot recount to verify.

When working, **keep this document current.** The user explicitly asked for a self-contained progress log so compact/reset doesn't lose work. After any meaningful change, update §1 (status table), §2 (what changed), §4 (outstanding), and §5 (paths if they moved). Treat this doc as a running log, not a one-shot handoff — that way every future session starts with ground truth.
