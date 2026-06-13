# Pupa Counter

`pupa-counter` is a reproducible pipeline for counting drosophila pupae from scanned `PNG / JPG / PDF / PPTX` sheets.

It implements the counting rule used in this project:

1. detect final pupa instances
2. find the top-most and bottom-most accepted pupa
3. use that span as the canonical height
4. draw the 5%, 25%, and 75% guide lines
5. report `n_top_5pct`, `top`, `middle`, `bottom`, and `n_pupa_final`

The current best default is a `Cellpose`-first detector with source-aware cleanup for annotated PNGs and clean scans.

## What This Repo Contains

- a CLI for manifest building, batch runs, config printing, and evaluation
- a full preprocessing stack for crop, normalization, blue-mark masking, and clean-image export
- two detection backends:
  - `cellpose` (default, best current backend)
  - `classical` brown-mask + component rules + cluster splitting
- review artifacts for every run:
  - overlays
  - clean images
  - masks
  - candidate tables
  - review queue
  - markdown/html reports
- tests for band math, mask logic, cluster handling, CLI smoke, and Cellpose postprocess

## Current Best Read

This repository is not "finished perfect automation". It is the best working version so far.

What is solid right now:

- annotated PNG sheets are handled much better than the older classical-only pipeline
- the `Cellpose` backend is currently the strongest default choice
- review artifacts are detailed enough to debug per-image failures
- overlays now number the accepted `middle`-band detections in green, which makes human QA easier
- outputs now also include a `top 5%` count derived from the same top-to-bottom anchor span
- each run now also exports a shareable `running_totals.xlsx` workbook and
  `running_totals.csv` for lab-facing review
- the clean-scan truth subset improved to `MAE = 6.00` in the latest iteration
- a lighter `reference` view is now saved for annotated PNG review, so human QA
  can compare the dark `clean_stage1` image against an original-like view

What is still open:

- some dense overlap regions still over-split or under-merge
- some raw PDFs still undercount compared with their cleaner PNG exports
- one important failure mode is that `clean_stage1` can sometimes darken or visually merge touching pupae, which may make a tight cluster harder to separate downstream

That last point is important: if a result looks suspicious, always inspect the saved `*_clean_stage1.png` together with the overlay.

The current code does **not** blindly switch the whole detector to the original image. Instead it uses a hybrid rule:

- the main full-image pass still runs on the cleaned image, because that suppresses blue annotations well
- dense annotated patches are re-checked on higher-recall views
- a lighter `reference` view is saved and used as a secondary rescue path for touching pupae
- that dense-patch rescue exists specifically to recover touching pupae that became too merged after cleaning

## Repository Layout

```text
configs/
  base.yaml
scripts/
  build_gold_subset.py
  run_pipeline.py
  tune_thresholds.py
src/pupa_counter/
  cli.py
  pipeline.py
  preprocess/
  detect/
  count/
  report/
  eval/
tests/
MAINTAINER_NOTES.md
PROJECT_STATUS_2026-04-08.md
```

## Installation

The project is written to run locally on macOS with Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Notes:

- the default detector backend is `cellpose`, so `cellpose` is included in the dependencies
- if you need a lighter fallback, switch `configs/base.yaml` to `detector.backend: classical`
- no API keys are stored in this repo; optional vision fallback only reads keys from environment variables

## Quick Start

Build a manifest:

```bash
python -m pupa_counter.cli manifest \
  --input-root /path/to/input \
  --config configs/base.yaml
```

Run the full pipeline:

```bash
python -m pupa_counter.cli run \
  --input-root /path/to/input \
  --config configs/base.yaml \
  --output-root data/processed/runs
```

Evaluate against a gold CSV:

```bash
python -m pupa_counter.cli evaluate \
  --pred data/processed/runs/baseline_v1/counts.csv \
  --gold data/gold/image_level_labels.csv
```

Print the resolved config:

```bash
python -m pupa_counter.cli print-config --config configs/base.yaml
```

## Webcam Research Notes

There is now a first-pass webcam capture prototype at:

```bash
python scripts/webcam_capture_loop.py --output-dir data/webcam_captures
```

This is **not** full live counting yet. It is a capture-and-process stepping
stone that lets the lab test camera placement, framing, and ROI conventions.

The current live-counting direction is documented in:

```text
WEBCAM_ROADMAP_2026-04-09.md
```

## Pipeline Overview

The end-to-end flow in `src/pupa_counter/pipeline.py` is:

1. discover supported input files
2. rasterize PDFs / PPTX pages
3. crop scanner borders
4. normalize background
5. detect blue marks
6. remove or ignore blue marks
7. run the configured detector backend
8. compute component features
9. postprocess detections
10. pick final accepted instances
11. derive top/bottom anchors and 5% / 25% / 75% lines
12. assign `top / middle / bottom`
13. write overlays, tables, review flags, and reports

### Detection Backends

`cellpose`:

- best current backend
- strongest on dense annotated PNGs
- uses `src/pupa_counter/detect/cellpose_backend.py`
- includes a conservative second-pass split for a small number of large,
  blob-like Cellpose masks that likely contain two touching pupae
- includes a dense-patch rescue pass in `src/pupa_counter/detect/cellpose_dense_patch.py`
  that re-runs Cellpose on alternate annotated views for only the most crowded
  regions
- includes a recall-biased annotated supplement that can add a small number of
  strong classical-only detections when the learned path misses a touching pair
- then runs source-aware cleanup in `src/pupa_counter/detect/cellpose_postprocess.py`

`classical`:

- brown mask detection
- connected components
- feature extraction
- rule filtering
- watershed splitting for clusters

### Optional Vision Fallback

There is an optional cluster-audit path in `src/pupa_counter/vision/openai_cluster_counter.py`.

It is disabled by default in `configs/base.yaml`.

If enabled, it only reads credentials from environment variables such as:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`

## Outputs

Each run writes to:

```text
data/processed/runs/<config_version>/
```

Important artifacts:

- `manifest.csv`
- `counts.csv`
- `running_totals.csv`
- `running_totals.xlsx`
- `review_queue.csv`
- `candidate_table.csv`
- `blue_supervision.csv`
- `vision_cluster_counts.csv`
- `intermediate/*.png`
- `overlays/*.png`
- `reports/run_summary.md`
- `reports/run_summary.html`

Overlay convention:

- orange contour = `top`
- green contour = `middle`
- red contour = `bottom`
- gold horizontal line = the `5%` guide line
- green center dots = accepted instances
- middle-instance numbering is available as an option, but the default overlays
  now leave labels off to make visual QA easier

Annotated intermediate convention:

- `*_clean_stage1.png` = the main cleaned view used by the primary full-image detector
- `*_normalized_stage0.png` = a lighter, original-like review view for annotated sheets
- `*_reference_stage0.png` = the same reference-style rescue view kept under an explicit name

During long runs, partial progress is also written:

- `counts.partial.csv`
- `candidate_table.partial.csv`
- `review_queue.partial.csv`
- `progress.json`

## How To Review A Suspicious Result

For one image, inspect these files together:

1. original image
2. `intermediate/<image_id>_clean_stage1.png`
3. `intermediate/<image_id>_brown_mask.png`
4. `overlays/<image_id>.png`
5. `candidate_table.csv`

This usually tells you which failure mode happened:

- missed detections in a dense band
- oversplitting of a touching cluster
- bright stain / border artifact
- clean-stage merge after normalization or inpainting

## Current Validation Snapshot

Latest local validation from `PROJECT_STATUS_2026-04-08.md` plus the annotated
`v5` review subset:

- automated tests: `23 passed`
- clean truth subset:
  - `scan` -> `64 vs 82`
  - `scan0001` -> `57 vs 54`
  - `scan0002` -> `40 vs 42`
  - `scan0003` -> `65 vs 55`
  - `scan0004` -> `62 vs 64`
  - `scan0005` -> `53 vs 54`
- clean-set `MAE = 6.00`
- annotated hard cases:
  - `scan_20260313_7` review subset: `111 -> 121`
  - `scan_20260313_22` review subset: `128 -> 143`
  - `scan_20260313_25` review subset: `121 -> 140`
  - `scan_20260313_74` review subset: total stayed `119`, but middle-band count is still not fully recovered

## Gold Labels And Manual Review

Gold CSV templates are included:

- `data/gold/image_level_labels.csv`
- `data/gold/object_level_labels.csv`

You can bootstrap an image-level sheet from an existing run:

```bash
python scripts/build_gold_subset.py \
  --counts data/processed/runs/baseline_v1/counts.csv \
  --output data/gold/image_level_labels.csv \
  --limit 15
```

## Known Limitations

- the clean-image export can sometimes make a touching group look darker or more merged than in the original image
- dense overlap is still the hardest regime
- raw PDFs and cleaner PNG exports do not behave identically
- old cluster-estimate logic is intentionally kept out of the main count path because it created systematic errors

## Maintainer Docs

More detailed internal notes live here:

- `MAINTAINER_NOTES.md`
- `PROJECT_STATUS_2026-04-08.md`

## License / Data

This repo is the codebase and documentation for the pipeline.

Large processed outputs, manual review packs, temporary debug assets, raw input data, and local test image directories are intentionally ignored from version control.
