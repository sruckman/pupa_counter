# Project Status 2026-04-08

## This Iteration

This pass added a **source-aware postprocess** on top of the Cellpose backend.

Files changed:

- `/Users/stephenyu/Documents/New project/src/pupa_counter/detect/cellpose_postprocess.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/pipeline.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/config.py`
- `/Users/stephenyu/Documents/New project/configs/base.yaml`
- `/Users/stephenyu/Documents/New project/tests/test_cellpose_postprocess.py`

What changed:

- Cellpose detections are no longer stamped as unconditional `pupa`.
- On `clean_png` / `clean_pdf`, extremely bright + weak-color Cellpose masks are re-labeled as `artifact`.
- On `clean_png` only, a **very conservative classical supplement** can add a few tiny, strong, non-overlapping detections that Cellpose missed.
- The aggressive old classical rule filter is still **not** applied to Cellpose outputs, because it wrecks recall on dense annotated images.

## Validation

### Automated tests

- `23 passed`

### Clean truth set

Run root:

- `/Users/stephenyu/Documents/New project/data/processed/stage2_clean_eval/baseline_v1`

Image-level results against the existing 6-image clean truth set:

| image | pred | truth | abs err |
|---|---:|---:|---:|
| scan | 64 | 82 | 18 |
| scan0001 | 57 | 54 | 3 |
| scan0002 | 40 | 42 | 2 |
| scan0003 | 65 | 55 | 10 |
| scan0004 | 62 | 64 | 2 |
| scan0005 | 53 | 54 | 1 |

- Clean-set `MAE = 6.00`
- Previous baseline comment in `base.yaml` was `MAE = 6.17`
- So this iteration is a **small but real improvement**, not a regression.

### Annotated hard cases

Run root:

- `/Users/stephenyu/Documents/New project/data/processed/stage2_debug_annotated/baseline_v1`

Key outcomes:

- `scan_20260313_7_78157780df`: `105` vs trusted blue total `109`
- `scan_20260313_74_f670a25628`: `100` vs trusted blue total `105`
- `scan_20260313_22_74a4d07f84`: still `119` vs trusted blue total `30`

Important interpretation:

- The new postprocess **did not break** the good high-recall behavior on `7` and `74`.
- The remaining big failure is still `22`.
- `22` is **not** a simple bright-noise problem. It survives the clean-only artifact gate almost unchanged.

## Important Observation About Footer Digits

I manually inspected rotated footer crops for:

- `Scan_20260313 (22).png`
- `Scan_20260313 (7).png`
- `Scan_20260313 (74).png`

Artifacts created during this check:

- `/Users/stephenyu/Documents/New project/tmp/footer_debug`
- `/Users/stephenyu/Documents/New project/tmp/footer_debug_rot`

The large blue edge digits do **not** line up with the trusted blue-dot totals on images like `7` and `74`, so for this subset they appear more like **page / sample IDs** than reliable count labels.

That means the currently trustworthy weak supervision remains:

- blue dots
- blue divider lines

and **not automatically the large handwritten footer digits**.

## Current Best Read

What is now solid:

- Cellpose is still the right primary backend for dense annotated PNGs.
- The old `estimate x3/x4` style cluster reasoning should stay out of the main path.
- A light, source-aware clean-image postprocess helps a bit on clean truth without harming the annotated wins.
- A conservative Cellpose overlap split now exists for a small number of large,
  blob-like masks that likely contain two touching pupae.

### Follow-up overlap split check

On the annotated strong subset (`7`, `22`, `74`), the overlap split pass produced:

- `scan_20260313_7`: `105 -> 106`
- `scan_20260313_74`: `100 -> 103`
- `scan_20260313_22`: unchanged at `119`

Interpretation:

- the new split pass is doing real work on some obvious merged masks
- it is currently conservative enough not to destabilize the whole image
- `22` remains a harder case and is not just a simple "one fat mask should become two" problem

What is still open:

- overlay QA is easier now because accepted `middle` detections are numbered in green on saved overlays
- `scan_20260313_22_74a4d07f84` style overcount / oversplitting
- large undercount on some raw PDF inputs from `/Users/stephenyu/Downloads/pupate_batch/*.pdf`
- the gap between raw PDFs and the cleaner PNG truth set in `/Users/stephenyu/Documents/New project/data/test_inputs/clean_pdfs`
- a newly confirmed review risk: `clean_stage1` can sometimes darken or merge
  touching pupae relative to the original image, so clean artifacts must be
  checked alongside the source image during QA

## Best Next Steps

1. Build a **manual review subset specifically for Cellpose false positives** on annotated PNGs like `22`, instead of using the old classical overlays.
2. Split evaluation more clearly into:
   - annotated PNG weak supervision
   - clean PNG truth set
   - raw PDF exploratory set
3. If we keep improving the current codebase, the next likely win is:
   - a merge / dedupe pass for oversplit Cellpose instances in dense overlap regions
   - not more classical rejection
