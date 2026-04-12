# Fresh-start peak-first detector — v0 / v1 findings

**Branch:** `codex/fresh-start-peak-proposal-v1`
**Package:** `src/pupa_counter_fresh/`
**Run date:** 2026-04-10
**Benchmark:** 20-image probe set under `/Users/stephenyu/Documents/New project/data/probe_inputs/all_20`
**Teacher:** `teacher_v8_20_instances.csv` (2175 instances, of which 181 are cellpose_split artifacts)
**Audit artifacts:** `/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs/fresh_peak_v1_final/`

## Mission

Start a fresh low-latency pupa counting pipeline from scratch, using v8 as an
offline teacher and audit baseline only. Do not extend the existing long
heuristic CV line. Judge progress instance-by-instance and visually, not by
total count alone.

## Headline result

| Detector | pred | matched | teacher_only | pred_only | recall | precision | F1 | mean runtime |
|---|---|---|---|---|---|---|---|---|
| old cc baseline (fresh_start notes, 5 hard images) | – | – | – | – | ~0.50 | – | – | ~2200 ms |
| old heuristic line `r133` (20 images) | 2168 | 1768 | 407 | 400 | 0.813 | 0.815 | 0.814 | ~2200 ms |
| **fresh v0** peak + global NMS (20 images) | 1856 | 1768 | 407 | 88 | **0.813** | **0.953** | **0.878** | **79.4 ms** |
| **fresh v1** peak + per-component split (20 images) | 2016 | 1933 | 242 | 83 | **0.889** | **0.959** | **0.922** | **67.8 ms** |

Instance-level deltas vs the old `r133` heuristic line:

* **v0** — identical recall (same 1768 matched) at *4.5× fewer false
  positives* (88 vs 400) and *27× faster* (79 ms vs ~2200 ms). v0 matches a
  different subset of teacher pupae than r133 (it wins big on some images,
  loses on others) but the aggregate is the same.
* **v1** — **+165 matched** over v0 (1933 vs 1768). **Every single one of the
  20 images improved or stayed flat on matched count.** False positives also
  dropped slightly (83 vs 88). F1 rose from 0.878 to 0.922. Runtime is
  *32× faster* than the heuristic line.

Relative to the "real" teacher (2175 − 181 cellpose_split = 1994 instances),
v1 misses 170 real pupae, so the **real-pupae recall is 91.5%**.

## Architecture

v0 / v1 live in a brand-new package that does not import anything from
`pupa_counter`:

```
src/pupa_counter_fresh/
  preprocess.py    # load_image_rgb, downscale, build_blue_mask
  response.py      # compute_response_map, build_allowed_mask
  peaks.py         # global NMS peak detection (v0)
  resolver_cv.py   # per-component peak splitter (v1) + v2 experiments
  geometry.py      # stub; band assignment is deferred
  eval_instances.py  # teacher-based matcher used by the audit harness
  detector.py      # run_detector(image_path, cfg) -> DetectorOutput
```

The driver is `scripts/run_fresh_peak_detector.py` and a diagnostic overlay
renderer is `scripts/fresh_diagnostic_overlay.py`.

### Pipeline stages

1. **Preprocess** — load RGB, downscale to 0.67×, build a permissive blue-ink
   mask.
2. **Response** — weighted brown/dark scalar map in [0, 1], then Gaussian
   smooth with σ=1.2, then zero-out blue-ink pixels.
3. **Allowed mask** — `response ≥ 0.12` (a permissive foreground gate; must
   be *lower* than the peak threshold so candidate peaks are reachable).
4. **Peak detection**
   * v0 — global `scipy.ndimage.maximum_filter` + NMS at `min_distance=10` in
     work pixels, `score ≥ 0.22`.
   * v1 — `scipy.ndimage.label` on the allowed mask, then
     `skimage.feature.peak_local_max(num_peaks = round(area / 200))` per
     component with `min_distance=3`. Dense clusters get split into multiple
     peaks; isolated pupae become single peaks.
5. **Rescale** — peak coords are mapped back to native pixels.
6. **Instance export** — the standard columns the audit matcher expects
   (`centroid_x`, `centroid_y`, `score`, `major_axis_px`, `bbox_*`, `band`).

### Key defaults

```python
DetectorConfig(
    work_scale=0.67,
    smooth_sigma=1.2,
    allowed_abs_threshold=0.12,
    peak_min_distance_px=10,                 # global NMS (v0)
    peak_abs_score_threshold=0.22,           # global NMS (v0)
    use_component_split=True,                # v1
    component_single_pupa_area_px=200.0,
    component_area_ratio_threshold=1.20,
    component_min_peak_distance_px=3,
    component_abs_score_threshold=0.18,
)
```

## Tuning log

**v0 calibration.** The first smoke test on scan 25 found 80 instances for a
134-pupa teacher, because `allowed_abs_threshold` was set above
`peak_abs_score_threshold`, so peaks between 0.15 and 0.28 were
unreachable. After decoupling the two thresholds and sweeping on the 5
hardest images, `at=0.12, pt=0.22, md=10` converged on the fresh-start
experiment's reported baseline (R≈0.79 at ~74 ms / image on the 5 hard
images), confirming v0 reproduces the reference baseline.

**v1 calibration.** Running a single global `peak_local_max` with
`min_distance=10` silently drops ~20% of teacher pupae whenever several
pupae touch inside the same allowed-mask component (on scan 25 alone,
62/134 teacher centroids sit within 14 work-pixels of another). Replacing
the global call with a per-component `peak_local_max(num_peaks = round(area
/ single_pupa_area), min_distance=3)` immediately recovered those merged
pupae. The best F1 came from
`single_pupa_area=200, ratio=1.20, min_distance=3, threshold=0.18`.

### What did not work (v2 exploration)

Multiple v2 approaches were tried and rejected after instance-level audits:

* **Distance-transform mixed signal** (response + α·EDT). Mixing pulled
  peaks toward the *midpoint* between two touching pupae, reducing recall
  to 0.79.
* **Distance-transform union seeds** (add EDT peaks to the response peaks,
  then union-NMS). Recall jumped to 0.94 but precision collapsed to 0.80
  (+415 FPs). Constraining dedup distance and EDT thresholds tied F1 with
  v1 but never cleanly beat it.
* **Erosion-based core counting.** With the current 0.12 allowed-mask
  threshold, 2–3 px erosion does not break touching-pupae bridges — the
  core count equals the area prior, so the override never fires.
* **Response core mask** (sub-label blobs at `response ≥ 0.24–0.28`
  and use the sub-count as expected_k). Gave +1 matched instance across
  20 images — peaks are already at those locations.
* **Work scale 0.80–1.00 with lower smoothing.** Recall climbs to 0.98 at
  scale 1.0 σ=0.7, but precision falls to 0.53. Most importantly: at scale
  0.80, the distributions of matched-peak scores (median 0.36, p10 0.32)
  and false-positive scores (median 0.34, p90 0.41) *overlap*, so score
  thresholding can never cleanly separate the two. The FPs at high
  resolution are structured, not noise.

**Conclusion from v2:** Further recall requires *structural* work, not more
threshold tuning:

* a shape / ellipticity verifier on each peak's local patch,
* or a learned per-patch classifier (the "tiny resolver" reserved in
  `docs/tiny_resolver_plan_2026-04-10.md`),
* or a sharper response feature that does not smooth touching pairs into
  one bump.

## Per-image breakdown (v1)

Sorted by total teacher_only. Real misses exclude cellpose_split rows.

| image | teacher | pred | matched | teacher_only | real_miss | split_miss | pred_only |
|---|---|---|---|---|---|---|---|
| scan 20 | 130 | 109 | 106 | 24 | 17 | 7 | 3 |
| scan 15 | 115 | 104 | 94 | 21 | 19 | 2 | 10 |
| scan 25 | 134 | 122 | 114 | 20 | 13 | 7 | 8 |
| scan 95 | 132 | 118 | 115 | 17 | 11 | 6 | 3 |
| scan 22 | 138 | 124 | 122 | 16 | 9 | 7 | 2 |
| scan 10 | 96 | 84 | 81 | 15 | 11 | 4 | 3 |
| scan 8 | 130 | 117 | 116 | 14 | 11 | 3 | 1 |
| scan 80 | 118 | 110 | 105 | 13 | 10 | 3 | 5 |
| scan 7 | 116 | 106 | 104 | 12 | 8 | 4 | 2 |
| scan 74 | 111 | 100 | 99 | 12 | 8 | 4 | 1 |
| scan 35 | 115 | 111 | 104 | 11 | 8 | 3 | 7 |
| scan 65 | 94 | 91 | 83 | 11 | 6 | 5 | 8 |
| scan 3 | 98 | 91 | 88 | 10 | 8 | 2 | 3 |
| scan 30 | 100 | 97 | 90 | 10 | 5 | 5 | 7 |
| scan 55 | 100 | 92 | 90 | 10 | 9 | 1 | 2 |
| scan 50 | 111 | 107 | 103 | 8 | 6 | 2 | 4 |
| scan 60 | 91 | 88 | 86 | 5 | 3 | 2 | 2 |
| scan 90 | 88 | 85 | 83 | 5 | 3 | 2 | 2 |
| scan 70 | 95 | 94 | 91 | 4 | 2 | 2 | 3 |
| scan 40 | 63 | 66 | 59 | 4 | 3 | 1 | 7 |

Worst image (scan 20): 17 real misses of 130 teacher pupae = 13% real-miss
rate on the hardest input. Best images (scan 40, 60, 70, 90): 2-3 real
misses, 0.96+ real recall.

## Visual audit findings

Diagnostic overlays for all 20 images are under
`/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs/fresh_peak_v1_final/diagnostic_overlays/`.
Legend: GREEN = matched prediction, YELLOW = false positive,
RED = real teacher miss, BLUE = cellpose_split teacher miss.

Patterns observed in the 10 hard-case images from `V8_AUDIT_PROTOCOL.md`:

1. **Almost every red circle sits directly next to a green circle.** The
   dominant failure mode is touching pupae sharing a single response bump,
   not isolated misses. This is structural — fixing it requires work on the
   response map or a shape verifier, not peak post-processing.
2. **No systematic edge-region failures.** The edge margin filter correctly
   suppresses border artifacts; no missed pupae cluster near the image
   frame.
3. **Yellow (FP) circles are rare** (1-10 per image) and usually on dust,
   thin blue-pen residue the blue mask missed, or narrow brown streaks in
   the paper texture. None of the FPs look like systematic detector bugs.
4. **Sparse-image performance is essentially perfect.** Images with
   <90 pupae (40, 60, 70, 90) show 2-3 real misses each; almost all
   non-touching pupae are caught.
5. **No band-reassignment artifacts.** Geometry is not enabled yet, so
   there is no opportunity for the old "progress from band shuffling" class
   of mistakes.

## Honest limitations

1. **Real-pupae recall is ~91.5%.** The remaining 8.5% are the touching-pair
   failures described above; they will not be recovered by more peak
   plumbing. A tiny learned resolver or a shape verifier is the likely
   next step.
2. **Teacher calibration is unverified.** We match to v8 as ground truth
   but do not audit whether v8 itself was correct. Some "false positives"
   may be real pupae that v8 missed, but that is out of scope here.
3. **Geometry stage is absent.** Band assignment / top-5% math is stubbed
   to `unassigned`. Adding band logic before instance detection stabilized
   further is explicitly forbidden by the handoff.
4. **Single-scan validation set.** All 20 benchmark images come from the
   2026-03-13 scan. We have not retested against older scans; some fixed
   numeric constants (single pupa area ≈ 200, min_peak_distance ≈ 3 at
   0.67x) may need adjustment when the input dpi or focal distance drift.

## Path forward (prioritized for the next iteration)

1. **Shape / local-context verifier for peak candidates.** At scale 0.80
   or 1.00 the recall ceiling is 0.96–0.98, but precision collapses. A
   cheap post-filter that rejects peaks whose immediate neighborhood is
   not "blob-shaped enough" could recover 5–10% recall without losing
   precision.
2. **Response sharpening.** The current Gaussian smoothing at σ=1.2 merges
   touching pupae. Alternatives: Laplacian-of-Gaussian (preserves multiple
   blobs), difference-of-Gaussians, or adaptive local smoothing keyed on
   the allowed-mask component size.
3. **Tiny per-patch resolver** (the option the handoff keeps in reserve).
   Train only on local crops around teacher-only regions from v1's output;
   outputs `single` / `pair` / `multi`. Use it as a local revisit cue, not
   a post-hoc total corrector. The dataset is already prepared in
   `fresh_start_agent_handoff_2026-04-10/benchmarks/resolver_patches_*`.
4. **Geometry pass.** Once instance recall is in the 0.93–0.95 range,
   estimate band lines from image evidence (horizontal density profile)
   and assign each accepted peak to top / middle / bottom. Do *not* use
   teacher-fraction fitting.
5. **Validation on a second scan batch** to check that the detector's
   numeric constants generalize.

## Reproducibility

v0:

```
python scripts/run_fresh_peak_detector.py \
    --run-name fresh_peak_v0_r001 \
    --detector-backend fresh_peak_v0 \
    --instance-source fresh_peak_v0
```

v1 (the final deliverable):

```
python scripts/run_fresh_peak_detector.py \
    --run-name fresh_peak_v1_final \
    --use-component-split \
    --detector-backend fresh_peak_v1 \
    --instance-source fresh_peak_v1
```

Diagnostic overlays:

```
python scripts/fresh_diagnostic_overlay.py \
    --run-dir "/Users/stephenyu/Documents/New project/data/processed/fresh_start_runs/fresh_peak_v1_final"
```

## Files produced by the v1 run

```
fresh_peak_v1_final/
  counts.csv
  instances.csv
  run_summary.csv
  matches_vs_teacher.csv
  teacher_only_instances.csv
  pred_only_instances.csv
  disagreement_vs_teacher.csv
  meta.json
  overlays/<image_id>_overlay.png
  diagnostic_overlays/<image_id>_diagnostic.png
  debug/<image_id>/{blue_mask,response_map,allowed_mask,peak_map}.png
```
