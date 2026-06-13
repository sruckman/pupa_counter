# Agent Handoff 2026-04-09

This document is the detailed handoff for the `pupa_counter` project as of
`2026-04-09`.

It is intentionally verbose. The goal is that another agent can read only this
file and quickly understand:

- which directory is the real repo
- which branch/commit is the current stable starting point
- what the pipeline does
- what was tried, what worked, what failed, and what was reverted
- where all important code, configs, docs, outputs, and human-review packs live
- what problems are still open
- what should be done next, and what should **not** be repeated

## 1. Executive Summary

The project is a pupa-counting pipeline for scanned sheets.

The best practical state right now is:

- the code base lives in the Git repo at
  `/Users/stephenyu/Documents/pupa_counter_publish`
- the best current branch is
  `codex/publish-current-pipeline`
- the current branch head is commit `02e44ff`, which is a **revert** of a failed
  overcount experiment and should be treated as the current stable starting
  point
- functionally, `02e44ff` is back to the same *behavior class* as the earlier
  `v8`-style stable probe runs
- the user's preferred visual/stability checkpoint is the local output set
  `annotated_probe_2026-04-08_v8`

Current high-level status:

- the pipeline is much better than the original unusable prototype
- it is good enough that many annotated PNGs are only off by a small number
- the dominant remaining error mode is **undercount** in dense / touching areas
- there are still some false positives possible near edges / blue dots / stains,
  but these are no longer the main issue
- a later experiment to fix small touching pairs (`d54dfdd`) caused severe
  overcount and was reverted; do **not** reuse it blindly

## 2. Directory Map

### 2.1 Canonical Git Repo

The canonical repo is:

`/Users/stephenyu/Documents/pupa_counter_publish`

This is the directory that is connected to GitHub and should be treated as the
authoritative codebase.

### 2.2 Mirror / Working Data Directory

There is also a second directory:

`/Users/stephenyu/Documents/New project`

Important:

- this directory contains many generated outputs, review packs, and also a
  mirror copy of code from earlier steps
- it is **not** the canonical Git repo for the latest work
- many older docs were written while operating inside `New project`, so they
  reference absolute paths under `/Users/stephenyu/Documents/New project/...`
- for outputs and review packs, those paths are still valid and important

### 2.3 Raw Input Data

Primary raw dataset:

`/Users/stephenyu/Downloads/pupate_batch`

Examples inside:

- `Scan_20260313 (7).png`
- `Scan_20260313 (10).png`
- `Scan_20260313 (22).png`
- `Scan_20260313 (25).png`
- `Scan_20260313 (74).png`
- `scan.pdf`
- `scan0001.pdf`
- `scan0002.pdf`
- `scan0003.pdf`
- `scan0004.pdf`
- `scan0005.pdf`

## 3. Branch / Commit Status

### 3.1 Current Branches

In `/Users/stephenyu/Documents/pupa_counter_publish`:

- `codex/publish-current-pipeline`
- `main`

As of handoff:

- local + remote `codex/publish-current-pipeline` head: `02e44ff`
- remote `main` head: `dacc192`

### 3.2 What This Means

`main` is **not** the most up-to-date stable work branch.

The branch that currently matters is:

`codex/publish-current-pipeline`

Why:

- many later iterations were experimental
- some later iterations were clearly worse
- it was safer to keep experiments off `main`
- the failed overcount experiment was committed as `d54dfdd`
- that experiment was then reverted by `02e44ff`

### 3.3 Important Commits

Key commits in reverse order:

| commit | meaning | status |
|---|---|---|
| `02e44ff` | Revert failed pairlike over-splitting experiment | current stable starting point |
| `d54dfdd` | Refine pairlike overlap splitting | failed, severe overcount, reverted |
| `14653b9` | Tighten interior merge rescue for annotated probe | user-liked `v8`-class checkpoint |
| `ef1d0fa` | Improve annotated edge filtering and reference detection | useful precursor to `v8` |
| `dacc192` | Improve annotated recall with reference view and rescue supplements | current `main` head, behind stable branch |
| `c9eba72` | Add annotated dual-path touching-pair rescue | major recall milestone |
| `05785e2` | Add dense patch rescue and top 5 percent counts | major feature milestone |
| `df71aaa` | Add numbered middle overlays for QA | human review quality-of-life milestone |
| `2ae2fa0` | Add conservative Cellpose overlap split | early touching-mask milestone |
| `f63940e` | publish current pupa counting pipeline | repo publish milestone |

### 3.4 Recommended Branch for Next Agent

Start from:

- repo: `/Users/stephenyu/Documents/pupa_counter_publish`
- branch: `codex/publish-current-pipeline`
- commit: `02e44ff`

## 4. What the Pipeline Does

### 4.1 Core Goal

Count pupae on scanned sheets and report:

- `n_pupa_final`
- `n_top_5pct`
- `n_top`
- `n_middle`
- `n_bottom`

The geometry rule is:

1. detect accepted pupa instances
2. find the top-most and bottom-most accepted instance
3. define total height span from those anchors
4. draw 5%, 25%, and 75% guide lines
5. count instances by centroid relative to those lines

### 4.2 Key Design Principle

This is intended to be a **generic pipeline**, not a one-image hack.

The algorithm is built around generic signals:

- white paper ROI
- blue annotation masking
- original-like reference view
- clean view for main detection
- Cellpose instance proposals
- generic postprocess / split / supplement modules
- centroid-based band counting

It is **not** supposed to be:

- hardcoded to one image
- based on one hand-drawn number
- a giant lookup table of exceptions

## 5. Code Map

### 5.1 Entry Points

- CLI:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/cli.py`
- Main pipeline:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/pipeline.py`

### 5.2 Config

Primary config:

- `/Users/stephenyu/Documents/pupa_counter_publish/configs/base.yaml`

Other configs:

- `/Users/stephenyu/Documents/pupa_counter_publish/configs/bench_clean_pdfs_cellpose.yaml`
- `/Users/stephenyu/Documents/pupa_counter_publish/configs/bench_stripped_pngs_cellpose.yaml`
- `/Users/stephenyu/Documents/pupa_counter_publish/configs/classifier_v1.yaml`
- `/Users/stephenyu/Documents/pupa_counter_publish/configs/gemini_audit.yaml`
- `/Users/stephenyu/Documents/pupa_counter_publish/configs/gemini_probe.yaml`
- `/Users/stephenyu/Documents/pupa_counter_publish/configs/gemini_vision.yaml`
- `/Users/stephenyu/Documents/pupa_counter_publish/configs/thresholds_v1.yaml`

Config dataclasses:

- `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/config.py`

### 5.3 Preprocess Modules

- crop:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/preprocess/crop.py`
- normalize:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/preprocess/normalize.py`
- blue mask:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/preprocess/blue_mask.py`
- inpaint:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/preprocess/inpaint.py`
- paper ROI:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/preprocess/paper_region.py`

### 5.4 Detection Modules

- Cellpose backend:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/cellpose_backend.py`
- Cellpose postprocess:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/cellpose_postprocess.py`
- Cellpose dense patch rescue:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/cellpose_dense_patch.py`
- Cellpose dual path:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/cellpose_dual_path.py`
- Cellpose overlap split:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/cellpose_split.py`
- classical brown mask:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/brown_mask.py`
- connected components:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/components.py`
- features:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/features.py`
- rule filter:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/rule_filter.py`
- cluster splitting:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/split_clusters.py`
- cluster fallback:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/cluster_fallback.py`
- candidate classifier:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/detect/classifier.py`

### 5.5 Counting Modules

- anchors:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/count/anchors.py`
- band assignment:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/count/assign.py`
- summary:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/count/summarize.py`

### 5.6 Reporting / QA

- overlays:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/report/overlay.py`
- review queue:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/report/review_queue.py`
- HTML report:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/report/html_report.py`
- blue supervision extraction:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/annotate/blue_supervision.py`

### 5.7 Evaluation

- metrics:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/eval/metrics.py`
- compare:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/eval/compare.py`
- error gallery:
  `/Users/stephenyu/Documents/pupa_counter_publish/src/pupa_counter/eval/error_gallery.py`

### 5.8 Scripts

- bootstrap dataset:
  `/Users/stephenyu/Documents/pupa_counter_publish/scripts/bootstrap_dataset.py`
- build gold subset:
  `/Users/stephenyu/Documents/pupa_counter_publish/scripts/build_gold_subset.py`
- generate report:
  `/Users/stephenyu/Documents/pupa_counter_publish/scripts/generate_report.py`
- run pipeline:
  `/Users/stephenyu/Documents/pupa_counter_publish/scripts/run_pipeline.py`
- train candidate classifier:
  `/Users/stephenyu/Documents/pupa_counter_publish/scripts/train_candidate_classifier.py`
- tune thresholds:
  `/Users/stephenyu/Documents/pupa_counter_publish/scripts/tune_thresholds.py`

### 5.9 Tests

Important tests:

- `/Users/stephenyu/Documents/pupa_counter_publish/tests/test_cellpose_split.py`
- `/Users/stephenyu/Documents/pupa_counter_publish/tests/test_cellpose_dense_patch.py`
- `/Users/stephenyu/Documents/pupa_counter_publish/tests/test_cellpose_dual_path.py`
- `/Users/stephenyu/Documents/pupa_counter_publish/tests/test_cellpose_postprocess.py`
- `/Users/stephenyu/Documents/pupa_counter_publish/tests/test_overlay.py`
- `/Users/stephenyu/Documents/pupa_counter_publish/tests/test_paper_region.py`
- `/Users/stephenyu/Documents/pupa_counter_publish/tests/test_cli_smoke.py`

Current test status at handoff:

- `45 passed`

## 6. Algorithm Mechanism (Detailed)

### 6.1 Why Cellpose

`Cellpose` is the main instance segmentation backbone.

It is used to propose individual pupa masks.

It is **not** trusted blindly. The pipeline wraps it with additional logic.

### 6.2 Main Flow

1. Discover image / PDF / PPTX inputs.
2. Rasterize PDF/PPTX if needed.
3. Crop scanner borders.
4. Find the white paper area.
5. Detect blue pen lines / dots / numbers.
6. Build multiple views:
   - original-like / stage0
   - normalized
   - clean / inpainted
   - reference
7. Run the main detector (usually Cellpose).
8. Compute component features:
   - area
   - eccentricity
   - aspect ratio
   - extent
   - border touch
   - color statistics
9. Apply source-aware postprocess:
   - suppress edge junk
   - suppress blue-dot overlaps
   - suppress stain-like bright artifacts
10. Run selective rescue modules:
   - dense patch rescue
   - dual-path rescue
   - conservative touching-mask split
11. Accept final instances.
12. Compute top/bottom anchors.
13. Compute 5%, 25%, 75% lines.
14. Count by centroid.
15. Write overlays / candidate table / review queue / reports.

### 6.3 Why Multiple Views Exist

This was a major discovery during iteration.

`clean_stage1` helps because:

- it removes blue annotation clutter
- it gives stronger foreground contrast

But it hurts because:

- it can darken pupae
- it can visually merge two nearby pupae
- dense clusters can become more blob-like

So the final design evolved into:

- keep the clean image for the primary full-image pass
- also keep a lighter reference/original-like view
- use the lighter view for rescue logic and human QA

### 6.4 Why the Algorithm Is Generic

The current modules are generic because they trigger on **shape and local image
structure**, not on a specific file name.

Examples:

- dense rescue triggers on dense local occupancy / clustering
- overlap split triggers on large or suspicious masks
- edge filtering triggers on border contact and ROI logic
- blue suppression triggers on blue overlap, not on image ID

The failed experiments are useful evidence that the team repeatedly tried to
improve the algorithm **generally**, not by one-off per-image hacks.

## 7. Human QA Workflow Used During Development

This project was not tuned by looking only at one scalar metric.

The effective workflow became:

1. Run a small targeted batch.
2. Open:
   - `original.png`
   - `overlay_numbered.png`
   - `clean_stage1.png`
   - `reference_stage0.png`
3. Zoom into suspicious local patches.
4. Compare:
   - what the model counted
   - what the human sees
   - whether the error is undercount or overcount
5. Use `candidate_table.csv` to identify the specific component row.
6. Use bbox and component ID to inspect whether:
   - two pupae became one mask
   - a blue dot became a pupa
   - a stain/edge fragment survived filtering
7. If necessary, build a mini run on only 2–5 images instead of rerunning 106.

This “small targeted probe” pattern is critical and should continue.

### 7.1 Important Practical Review Artifacts

Human-review packs were assembled repeatedly under:

`/Users/stephenyu/Documents/New project/data/manual_review`

Especially:

- `problem_pack_2026-04-08_v2`
- `annotated_bulk_review_2026-04-08_v1`
- `annotated_bulk_review_2026-04-08_v3`
- `annotated_bulk_review_2026-04-08_v4`
- `annotated_bulk_review_2026-04-08_v5`
- `annotated_bulk_review_2026-04-08_v6`
- `annotated_probe_2026-04-08_v7`
- `annotated_probe_2026-04-08_v8`

These folders usually contain:

- `original.png`
- `overlay_numbered.png`
- `clean_stage1.png`
- `normalized_stage0.png`
- `reference_stage0.png`
- `blue_mask.png`
- `review_manifest.csv`
- `README.md`

### 7.2 Important Human Conclusions

The user repeatedly provided the following high-value feedback:

- `ABC`-style annotated PNGs are much more relevant than the clean PDF path
- remaining important errors are mostly **missed detections**, not wild false
  positives everywhere
- some local touching pairs are still counted as one
- `clean_stage1` darkening/merging is real and harmful
- footer digits should not be treated as trusted labels

This feedback materially shaped the pipeline.

## 8. Existing Documentation (Read These)

Inside the canonical repo:

- `/Users/stephenyu/Documents/pupa_counter_publish/README.md`
- `/Users/stephenyu/Documents/pupa_counter_publish/MAINTAINER_NOTES.md`
- `/Users/stephenyu/Documents/pupa_counter_publish/PROJECT_STATUS_2026-04-08.md`
- `/Users/stephenyu/Documents/pupa_counter_publish/AGENT_HANDOFF_DETAILED_2026-04-09.md` (this file)

In the mirror directory:

- `/Users/stephenyu/Documents/New project/PROJECT_HANDOFF_2026-04-07.md`
- `/Users/stephenyu/Documents/New project/PROJECT_STATUS_2026-04-08.md`
- `/Users/stephenyu/Documents/New project/MAINTAINER_NOTES.md`

Note:

- older docs in `New project` are still useful for path discovery
- they are not authoritative for branch/commit state

## 9. Results / Artifact Directory Map

### 9.1 Earliest Broad Baseline

Full batch baseline:

`/Users/stephenyu/Documents/New project/data/processed/runs/baseline_v1`

This is the original broad run over the whole sample set and is useful as a
historical low baseline.

Examples from that early run:

- `10`: `55`
- `22`: `85`
- `25`: `70`

This is far below later annotated-focused versions and should mainly be used for
historical comparison.

### 9.2 Clean Truth Evaluation

Clean truth evaluation:

`/Users/stephenyu/Documents/New project/data/processed/stage2_clean_eval/baseline_v1`

Known clean truth numbers from that run:

| image | pred |
|---|---:|
| `scan` | 64 |
| `scan0001` | 57 |
| `scan0002` | 40 |
| `scan0003` | 65 |
| `scan0004` | 62 |
| `scan0005` | 53 |

This branch of evaluation matters less than annotated PNGs for current use,
because the user said the real production setup will look more like the cleaner
annotated PNG capture style rather than those six difficult clean-PDF cases.

### 9.3 Human Review Packs

Key human review packs:

- `/Users/stephenyu/Documents/New project/data/manual_review/problem_pack_2026-04-08_v2`
- `/Users/stephenyu/Documents/New project/data/manual_review/problem_pack_2026-04-08_v2.zip`
- `/Users/stephenyu/Documents/New project/data/manual_review/review_pack_2026-04-08`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_bulk_review_2026-04-08_v1`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_bulk_review_2026-04-08_v3`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_bulk_review_2026-04-08_v4`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_bulk_review_2026-04-08_v5`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_bulk_review_2026-04-08_v6`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v7`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8`

### 9.4 Recommended User-Facing Stable Visual Set

The user repeatedly referred back to `v8` as a better/stabler checkpoint.

Those assets are here:

- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8.zip`
- `/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-08_v8/baseline_v1`
- `/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-08_v8/baseline_v1/counts.csv`

The five `v8` probe cases are:

- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8/01_scan_20260313_10`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8/02_scan_20260313_22`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8/03_scan_20260313_25`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8/04_scan_20260313_7`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8/05_scan_20260313_74`

### 9.5 Pair-Only Debug Run

The local 2-image pair split debug run is here:

- `/Users/stephenyu/Documents/New project/data/processed/annotated_pair_probe_2026-04-09_v2/baseline_v1`

This run is useful because it proves the later pair-splitting experiment could
fix specific local doublets:

- in `10`, `cp_00059` split into two children
- in `74`, `cp_00053` split into two children

However, that logic was **not accepted globally**, because the same experiment
caused severe overcount on other images.

## 10. Version Timeline: What Changed, What Worked, What Broke

This section is the most important part for the next agent.

### 10.1 Broad Early Baseline (`full_batch_baseline`)

Run:

`/Users/stephenyu/Documents/New project/data/processed/runs/baseline_v1`

Characteristics:

- early broad baseline
- clearly undercounted dense annotated images
- useful only as a historical floor

### 10.2 Bulk Review Iterations on 2026-04-08

#### `bulk_v1`

Run:

`/Users/stephenyu/Documents/New project/data/processed/annotated_bulk_review_eval_2026-04-08_v1/baseline_v1`

Key totals:

- `10`: 79
- `22`: 119
- `25`: 114
- `7`: 106
- `74`: 103

Read:

- better than earliest full baseline
- still visibly undercounting dense cases

#### `bulk_v3`

Run:

`/Users/stephenyu/Documents/New project/data/processed/annotated_bulk_review_eval_2026-04-08_v3/baseline_v1`

Key totals:

- `10`: 79
- `22`: 128
- `25`: 118
- `7`: 107
- `74`: 112

Interpretation:

- recall improved in dense annotated cases
- especially `22`, `25`, and `74`

#### `bulk_v4`

Run:

`/Users/stephenyu/Documents/New project/data/processed/annotated_bulk_review_eval_2026-04-08_v4/baseline_v1`

Key totals:

- `10`: 81
- `22`: 128
- `25`: 121
- `7`: 111
- `74`: 119

Interpretation:

- more recall gains
- touching-pair rescue and annotated dual-path logic were helping

#### `bulk_v5`

Run:

`/Users/stephenyu/Documents/New project/data/processed/annotated_bulk_review_eval_2026-04-08_v5/baseline_v1`

Key totals:

- `10`: 86
- `22`: 143
- `25`: 140
- `7`: 121
- `74`: 119

Interpretation:

- stronger recall
- but beginning to flirt with over-aggression in some cases

#### `bulk_v6`

Run:

`/Users/stephenyu/Documents/New project/data/processed/annotated_bulk_review_eval_2026-04-08_v6/baseline_v1`

Key totals:

- `10`: 84
- `22`: 142
- `25`: 129
- `7`: 112
- `74`: 115

Interpretation:

- cleaned up some false positives
- but also became a bit more conservative again

User reaction at this stage:

- quality was much better than early versions
- remaining errors were mostly “漏数” rather than wild false positives

### 10.3 Probe Iterations (`v7` and `v8`)

#### `probe_v7`

Run:

`/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-08_v7/baseline_v1`

Key totals:

- `10`: 94
- `22`: 135
- `25`: 130
- `7`: 111
- `74`: 108

Key middle counts:

- `10`: 62
- `22`: 117
- `25`: 103
- `7`: 77
- `74`: 75

Interpretation:

- edge errors improved
- clean/reference split became more mature
- still undercount in touching clusters

#### `probe_v8` (user-liked stable checkpoint)

Run:

`/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-08_v8/baseline_v1`

Key totals:

- `10`: 96
- `22`: 138
- `25`: 134
- `7`: 116
- `74`: 111

Key middle counts:

- `10`: 64
- `22`: 119
- `25`: 105
- `7`: 79
- `74`: 77

Key `top 5%` counts:

- `10`: 1
- `22`: 2
- `25`: 5
- `7`: 4
- `74`: 4

Interpretation:

- this is the last checkpoint the user clearly described as “very good” /
  “much better”
- still has undercount in `25`
- still misses some local touching pairs in `10` and `74`
- but does not exhibit the later catastrophic overcount

This is why `v8` is the recommended visual / qualitative stable reference.

### 10.4 2026-04-09 Pairlike Experiment Series

#### `probe_0409_v1`

Run:

`/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-09_v1/baseline_v1`

Key totals:

- `10`: 96
- `22`: 138
- `25`: 134
- `7`: 116
- `74`: 111

This was effectively the same neighborhood as `v8`.

#### `probe_0409_v3` / `probe_0409_v4`

Runs:

- `/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-09_v3/baseline_v1`
- `/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-09_v4/baseline_v1`

These were increasingly aggressive attempts to repair local touching pairs.

Totals moved upward:

- `v3`: `10/22/25/7/74 = 103 / 149 / 147 / 124 / 118`
- `v4`: `10/22/25/7/74 = 106 / 157 / 155 / 131 / 123`

Interpretation:

- some local pair fixes were real
- but global totals were already drifting too high

#### `d54dfdd` / `probe_0409_v5` / `probe_0409_v6` (FAILED)

This was the failed experiment.

Commit:

`d54dfdd Refine pairlike overlap splitting`

Runs:

- `/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-09_v5/baseline_v1`
- `/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-09_v6/baseline_v1`

Key totals:

- `10`: 122
- `22`: 173
- `25`: 167
- `7`: 143
- `74`: 133

Key middle counts:

- `10`: 85
- `22`: 151
- `25`: 136
- `7`: 106
- `74`: 96

Why it failed:

- it solved some specific “two counted as one” examples
- but globally it over-split too many masks
- the user explicitly called this version a failure and asked for rollback

#### `02e44ff` Revert

Current branch head:

`02e44ff Revert "Refine pairlike overlap splitting"`

Meaning:

- the severe-overcount experiment was removed
- branch state returned to the earlier stable class

## 11. Specific Known Problems

### 11.1 `25` Still Undercounts

This is the most important remaining open issue.

The user repeatedly pointed out that `25` still misses pupae in dense local
regions.

Important nuance:

- this is **not** the same issue as a clean touching pair
- it is more like dense-cluster recall / local merge behavior
- trying to solve it by globally splitting more aggressively caused overcount

Conclusion:

- `25` should be attacked with a **local dense patch strategy**, not a global
  split-everything strategy

### 11.2 Touching Pairs in `10` and `74`

In stable `v8`, the user identified local cases where two touching pupae were
still counted as one.

Later pair-probe experiments proved these local cases can be split.

However:

- the global implementation of that idea caused unacceptable overcount
- a future fix must preserve the local win while staying far more selective

### 11.3 `clean_stage1` Can Merge Pupae

This was repeatedly observed and confirmed.

The question the user asked directly was:

> Why does removing blue change the pupa color?

Correct answer:

- it does not *need* to
- current implementation side effects made some clean images darker
- darker/cleaner is useful for blue suppression
- but it can merge close pupae visually and hurt recall

Conclusion:

- do not rely on `clean_stage1` alone when debugging misses
- always compare with `reference_stage0` / original

### 11.4 `main` Branch Is Not the Most Useful Branch

This is a project-organization problem, not an algorithm problem.

Current situation:

- `main` is behind
- the stable latest work is on `codex/publish-current-pipeline`

This should be cleaned up later, but is not the top algorithmic priority.

### 11.5 Reproducibility Is Good but Not Perfect

The project is close to reproducible, but not perfectly frozen.

Why not perfect:

- dependencies are version ranges, not exact locks
- Cellpose weights are not committed into the repo
- first Cellpose run may download weights
- large result directories and raw data are not in GitHub

## 12. Explicit “Do Not Repeat” Lessons

The next agent should avoid these traps:

1. **Do not** assume bigger totals mean better results.
   - This caused the failed `d54dfdd` experiment.

2. **Do not** keep pushing global split aggressiveness.
   - It fixes local pairs and destroys global counts.

3. **Do not** treat footer digits as truth labels.
   - Blue dots and divider lines were the more trusted weak supervision.

4. **Do not** trust only `clean_stage1` when judging a miss.
   - The missing pupa may still be visible in the original/reference view.

5. **Do not** assume the `New project` code tree is the canonical repo.
   - Use `pupa_counter_publish` for Git work.

6. **Do not** merge current branch to `main` without intentionally deciding that
   `main` should now become the stable branch.

## 13. Recommended Next Steps for the Next Agent

### 13.1 Start Point

Start from:

- repo:
  `/Users/stephenyu/Documents/pupa_counter_publish`
- branch:
  `codex/publish-current-pipeline`
- commit:
  `02e44ff`

### 13.2 First Technical Target

Focus on `25` undercount without increasing global overcount.

Recommended shape:

- work locally on dense patches
- do not increase global pair-splitting aggressiveness
- compare original/reference/clean on the same patch
- require plausibility gates before adding any new instance

### 13.3 Suggested Debug Method

1. Use a **2–5 image probe set** only:
   - `10`
   - `22`
   - `25`
   - `7`
   - `74`
2. If trying a local pair fix, also verify:
   - did `10` improve?
   - did `74` improve?
   - did `22/25/7` explode?
3. Prefer local crop inspection over whole-dataset reruns.

### 13.4 Suggested Review Assets

Primary stable human-review set:

- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_probe_2026-04-08_v8`

Failed overcount contrast set:

- `/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-09_v5/baseline_v1`
- `/Users/stephenyu/Documents/New project/data/processed/annotated_probe_eval_2026-04-09_v6/baseline_v1`

Pair-fix proof set:

- `/Users/stephenyu/Documents/New project/data/processed/annotated_pair_probe_2026-04-09_v2/baseline_v1`

## 14. Reproduction Notes

### 14.1 Install

Current documented install flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 14.2 Typical Commands

Manifest:

```bash
python -m pupa_counter.cli manifest \
  --input-root /path/to/input \
  --config configs/base.yaml
```

Run:

```bash
python -m pupa_counter.cli run \
  --input-root /path/to/input \
  --config configs/base.yaml \
  --output-root data/processed/runs
```

Evaluate:

```bash
python -m pupa_counter.cli evaluate \
  --pred data/processed/runs/baseline_v1/counts.csv \
  --gold data/gold/image_level_labels.csv
```

Print config:

```bash
python -m pupa_counter.cli print-config --config configs/base.yaml
```

### 14.3 Important Caveats

- if running from outside the repo, ensure `PYTHONPATH` or editable install is
  correct
- during development, one repeated pitfall was accidentally running the mirror
  package from `New project/src` instead of the real repo source in
  `pupa_counter_publish/src`
- if behavior seems inconsistent, verify which package path Python is using

## 15. Miscellaneous Work Done During This Project

These were done and may matter contextually:

- generated many numbered overlays to make human QA possible
- added `top 5%` counting support
- prepared a Gmail draft update to Sarah summarizing progress and asking about
  cluster access / other lab work
- experimentally wired Gemini audit/probe paths, but did **not** keep them as
  the main production counting path

## 16. Final Practical Guidance

If another agent takes over immediately, the safest mindset is:

- treat `v8` as the last user-approved behavior checkpoint
- treat `02e44ff` as the current stable code checkpoint
- keep the good parts:
  - edge cleanup
  - reference view
  - numbered overlays
  - top 5% count
  - dense rescue framework
- do **not** continue the reverted `d54dfdd` strategy
- attack `25` with local dense reasoning, not global over-splitting

If only one thing should be looked at next, it is:

**How to recover missed pupae in dense local regions like `25` without pushing
`22/7/74` into overcount.**

