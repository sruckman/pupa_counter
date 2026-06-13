# Maintainer Notes

## What This Baseline Does Well

- Handles annotated PNG and clean PDF in one CLI.
- Uses deterministic run folders keyed by `config_version`.
- Saves enough intermediate artifacts to debug each phase.
- Keeps anchor selection more robust by preferring non-border, higher-confidence
  accepted instances before falling back to all final instances.

## Known Failure Modes

- Dense cluster regions still produce many unresolved cluster flags.
- Clean PDFs are currently much sparser than annotated PNGs and need a dedicated
  threshold pass or a small clean-only gold subset.
- Handwritten blue markings are masked conservatively; overlap-heavy regions are
  more likely to undercount than overcount.
- `clean_stage1` can sometimes darken or visually merge touching pupae compared
  with the original scan. When a result looks suspicious, inspect the original
  image next to `*_clean_stage1.png` before tuning detector thresholds.

## Where To Tune First

1. `configs/base.yaml`
2. `src/pupa_counter/detect/brown_mask.py`
3. `src/pupa_counter/detect/rule_filter.py`
4. `src/pupa_counter/detect/split_clusters.py`
5. `src/pupa_counter/report/review_queue.py`

## Suggested Next Iteration

1. Create `data/gold/image_level_labels.csv` for 10-15 representative images.
2. Use the existing `scripts/build_gold_subset.py` to bootstrap the sheet.
3. Compare predicted `n_middle` against the gold counts.
4. Tune the brown score and cluster threshold separately for:
   - annotated PNG
   - clean PDF
5. Only enable the optional classifier after a basic gold set exists.

## Phase 9 TODO

- `scripts/train_candidate_classifier.py` currently exports a clear TODO marker.
- `scripts/tune_thresholds.py` is ready as the next place to automate sweeps.
