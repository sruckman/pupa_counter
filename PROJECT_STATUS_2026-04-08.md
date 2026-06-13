# Project Status 2026-04-08

## Follow-up Dense Patch Rescue

After the earlier source-aware Cellpose pass, a new issue became clearer on
annotated PNGs such as `22`, `25`, and `74`:

- the main failure was often **local undercount in crowded regions**
- `clean_stage1` sometimes made neighboring pupae look darker and more merged
- switching the entire detector to the original image was not safe enough

So this follow-up iteration added a **dense-patch rescue** instead of a global
backend change.

Files changed in this follow-up:

- `/Users/stephenyu/Documents/New project/src/pupa_counter/detect/cellpose_backend.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/detect/cellpose_dense_patch.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/pipeline.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/detect/cellpose_postprocess.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/config.py`
- `/Users/stephenyu/Documents/New project/configs/base.yaml`
- `/Users/stephenyu/Documents/New project/tests/test_cellpose_dense_patch.py`

What changed:

- the primary full-image Cellpose pass still runs on `clean_stage1`
- only dense `annotated_png` regions are selected for a second local pass
- that second pass reruns Cellpose on the `normalized` image patch, not the
  cleaned patch
- replacement is gated tightly, so only modest and plausible gains are accepted
- accepted rescue detections are tagged as `cellpose_dense_patch` in
  `candidate_table.csv`

Why this shape:

- full-image `clean -> normalized` switching did not produce a reliable win
- local dense rescue improved recall without destabilizing the calmer parts of
  the sheet
- this matches the observed human failure mode better: a few crowded clusters
  are the real problem, not the whole image

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

### Dense rescue check on the 15-image annotated review subset

Run root:

- `/Users/stephenyu/Documents/New project/data/processed/annotated_bulk_review_eval_2026-04-08_v2/baseline_v1`

Key count deltas versus the earlier review batch:

- `scan_20260313_22_74a4d07f84`: `119 -> 128`
- `scan_20260313_25_db0b949a97`: `114 -> 118`
- `scan_20260313_74_f670a25628`: `103 -> 112`
- `scan_20260313_7_78157780df`: `106 -> 107`

Interpretation:

- the new pass is behaving like a **recall supplement**, not a global rewrite
- several dense annotated cases now recover extra instances
- the gain is strongest on `22`, `25`, and `74`, which were repeatedly called
  out during visual review
- this still needs human QA, because some crowded regions may now be closer to
  correct while others may have become slightly too aggressive

Automated test status after this follow-up:

- `27 passed`

## Added 5 Percent Count

Based on the latest feedback, the pipeline now also reports a **top 5% count**
derived from the same top-to-bottom anchor span that already defines the
25% / 75% middle band.

What this means in practice:

- `upper_five_pct_y = top_y + 0.05 * (bottom_y - top_y)`
- any accepted instance with centroid at or above that line contributes to
  `n_top_5pct`
- the existing `top / middle / bottom` buckets are unchanged

Where it appears:

- `counts.csv` now includes `n_top_5pct`
- `counts.csv` now includes `upper_five_pct_y`
- overlays show a gold `5%` guide line
- overlays show `top5%=...` in the stats box

This is intentionally additive:

- it does not change how the main middle-band count is computed
- it gives Sarah one extra, narrower top-of-span statistic without breaking
  any of the existing output columns that downstream scripts already use

## Best Next Steps

1. Build a **manual review subset specifically for Cellpose false positives** on annotated PNGs like `22`, instead of using the old classical overlays.
2. Split evaluation more clearly into:
   - annotated PNG weak supervision
   - clean PNG truth set
   - raw PDF exploratory set
3. If we keep improving the current codebase, the next likely win is:
   - a merge / dedupe pass for oversplit Cellpose instances in dense overlap regions
   - not more classical rejection

## Latest Annotated Recall Pass

The newest follow-up focused on one user-confirmed property of the current
errors:

- the existing counted pupae are usually correct
- the dominant error is **undercount**
- the hardest misses are touching / merged local groups

So this pass intentionally biased the annotated PNG path toward **higher
recall**.

Files changed in this pass:

- `/Users/stephenyu/Documents/New project/src/pupa_counter/preprocess/normalize.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/detect/cellpose_split.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/detect/cellpose_postprocess.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/pipeline.py`
- `/Users/stephenyu/Documents/New project/src/pupa_counter/config.py`
- `/Users/stephenyu/Documents/New project/configs/base.yaml`
- `/Users/stephenyu/Documents/New project/tests/test_reference_view.py`
- `/Users/stephenyu/Documents/New project/tests/test_cellpose_split.py`
- `/Users/stephenyu/Documents/New project/tests/test_cellpose_postprocess.py`

What changed:

- annotated runs now build a lighter `reference` view that stays closer to the
  original crop and avoids the strong darkening from the main normalized path
- that lighter view is saved as both `reference_stage0` and
  `normalized_stage0` for human review, so the review artifact is no longer
  visually identical to `clean_stage1`
- the Cellpose overlap split on annotated PNGs now considers any sufficiently
  large mask, not only fat/round masks, so elongated dumbbell-like touching
  pairs can be split
- a new annotated-only classical supplement can add back a small number of
  strong unmatched detections when the learned route misses a real touching
  pair entirely

Automated tests after this pass:

- `17 passed` on the focused postprocess / split / overlay subset

Annotated review subset comparison (`v4 -> v5`):

- improved total count on `9 / 15`
- worsened total count on `3 / 15`
- improved middle-band count on `8 / 15`
- worsened middle-band count on `5 / 15`

Largest total-count gains:

- `22`: `128 -> 143`
- `25`: `121 -> 140`
- `79`: `110 -> 122`
- `7`: `111 -> 121`

Important caveat:

- the new pass helps recall in several user-flagged dense cases, but it is not
  a universal win yet
- `74` keeps its total (`119`) but still shifts some accepted detections out of
  the middle band
- `12`, `19`, and `97` are good examples of why this still needs human QA

Newest review pack:

- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_bulk_review_2026-04-08_v5`
- `/Users/stephenyu/Documents/New project/data/manual_review/annotated_bulk_review_2026-04-08_v5.zip`
