# Response sharpening iteration — honest negative results + infrastructure

**Branch:** `codex/fresh-response-sharpen-v1` (this iteration)
**Parent:** `codex/fresh-start-peak-proposal-v1` (v1 final, commit `ce1c0ab`)
**Goal of the ultraplan:** Close the touching-pair recall gap without
sacrificing latency. Acceptance gate: real-pupae recall ≥ **0.94**,
precision ≥ **0.94**, F1 ≥ **0.93**, and ``scan_20260313_25`` teacher_only
must drop from v1's 20 to ≤ **14**.

**Result:** **Gate not met.** The response-sharpening mechanism
(``smooth`` → ``log`` / ``dog`` / ``adaptive``) is real but its effect on
real pupae is +0.3 F1 points at best. Paper ROI was also investigated and
found ineffective. The test harness, sweep driver, and paper_roi module
are kept because they are genuinely reusable, but the primary hypothesis
is falsified.

This document records what was tried, what the data said, and why the
next iteration should *not* be more peak-detection plumbing.

## What was tried

### 1. Test scaffold with regression gate (`tests/fresh/`)

New hermetic test package. Five test files:

* `conftest.py` — `make_synthetic_pupa_image` paints elliptical Gaussian
  blobs at specified centers, `make_blue_ink_image` paints a blue stroke.
  No test reads from the user's local probe data.
* `test_eval_instances.py` — matcher unit tests (perfect match, one miss,
  one FP, cross-image join via canonical scan number). Prevents silent
  drift in the metrics pipeline.
* `test_detector_smoke.py` — end-to-end `run_detector` on a synthetic
  4-pupa image. Verifies the detector still returns the right shape, the
  debug dict is populated, and the blue mask suppresses overlapping pupae.
* `test_response_sharpen.py` — the core regression gate for this
  iteration. Every mode must find 1 peak on an isolated pupa, and
  `log` / `dog` / `adaptive` must find 2 peaks on a touching pair where
  `smooth` finds only 1. The fixture uses a Gaussian blob at
  σ_major=4, σ_minor=3, pair separation=10 — the setting empirically
  tuned to exactly differentiate the modes.
* `test_paper_roi.py` — paper ROI regression gate. A painted "scanner
  bar" on the left edge must be excluded from the detected mask; a pupa
  painted on top of the bar must not appear in the detector instances
  when `use_paper_roi=True`.

All 32 tests pass in ~0.5 s. Run with `pytest tests/fresh -q`.

### 2. Response mode dispatcher (`pupa_counter_fresh/response.py`)

`compute_response_map` now takes `response_mode: Literal["smooth","log","dog","adaptive"]`:

* `smooth` — plain Gaussian blur with `smooth_sigma` (the v1 default,
  bit-compatible when the other sharpening params are untouched).
* `log` — `Gauss(σ) - Gauss(2σ)`, re-clipped and renormalized.
  Single-sigma DoG approximation to Laplacian-of-Gaussian.
* `dog` — `Gauss(σ_low) - Gauss(σ_high)` with independent sigmas.
* `adaptive` — permissive gate → connected components → per-component
  sigma (small sigma for components below `area_threshold_px`, large
  sigma otherwise) → composite.

Blue-mask zeroing runs *after* sharpening in every mode so LoG/DoG
ringing inside a blue stroke cannot leak into real regions.

### 3. Detector / CLI wiring (`pupa_counter_fresh/detector.py`, `scripts/run_fresh_peak_detector.py`)

`DetectorConfig` gained `response_mode`, `log_sigma`, `dog_sigma_low`,
`dog_sigma_high`, `adaptive_small_sigma`, `adaptive_large_sigma`,
`adaptive_area_threshold_px`, and the paper-ROI fields. The CLI driver
exposes matching `--response-mode`, `--log-sigma`, `--dog-sigma-low`,
etc. flags.

**Also fixed a latent bug:** `DetectorConfig`'s default
`component_*` values were pre-v1 values (`area=230, mind=5, thr=0.20`).
I updated them to the v1-tuned values (`area=200, mind=3, thr=0.18`) so
`DetectorConfig()` with no arguments now produces v1 baseline behavior.
This caught a reproduction-failure early in the iteration and is worth
calling out — the class defaults had diverged from the CLI defaults.

### 4. Autonomous sweep driver (`scripts/run_fresh_sweep.py`)

In-process sweep loop (no subprocess overhead). Reads a YAML grid,
expands each `variant` into the cartesian product of its `axes`, builds
a `DetectorConfig`, calls `run_detector` on a fixed image set, evaluates
against `teacher_v8_20_instances.csv`, and writes:

* `sweep.csv` — one row per grid point with full metrics + all swept
  hyperparameters
* `best.json` — winner picked by F1 under `precision_floor` / `recall_floor`
* optional `--early-stop-recall` to exit the loop the moment a config
  crosses a target recall while still inside the precision floor

Grid file `configs/sweeps/log_dog_v1.yaml` defines the first iteration:

* baseline: `smooth_baseline` (v1 reference point)
* log: 48 points (log_sigma × allowed_threshold × single_pupa_area)
* dog: 54 points (sigma_low × sigma_high × allowed_threshold × single_pupa_area)
* adaptive: 36 points (small_sigma × large_sigma × allowed_threshold × single_pupa_area)

**First-run bug fixed:** the sweep was running on 5 hardest images but
loading the full 20-image teacher, so 15 unrelated teacher images showed
up as `teacher_only` and recall scored as 0.20. Fix: filter the teacher
frame to the same scan keys as the image subset.

### 5. Paper ROI module (`pupa_counter_fresh/paper_roi.py`)

`detect_paper_roi(rgb, cfg)` — largest bright connected component with a
configurable morphological close to fill internal holes and an optional
inward erosion margin. `apply_paper_roi_to_response(response, mask)` —
zero the response outside the mask. Wired into `run_detector` via
`DetectorConfig.use_paper_roi` (default off, backward compatible).

### 6. Cleaned teacher script (`scripts/rebuild_cleaned_teacher.py`)

Takes `teacher_v8_20_instances.csv` and drops rows whose centroid sits
outside the paper ROI, for offline teacher cleaning. **This script
produced a bogus cleaned teacher on first run** (see pitfall below) and
the output was renamed with a `_BOGUS_DO_NOT_USE` suffix. The script is
kept for reference but its current default parameters are wrong for
native-resolution inputs; see below.

## What the data said

### Response-mode sweep on the 5 hardest images

139 grid points ran in ~8 minutes in-process. Best F1 per variant,
sorted by F1:

| variant | best F1 | best recall | best precision | vs v1 F1 |
|---|---|---|---|---|
| smooth (v1 baseline) | 0.900 | 0.851 | 0.955 | reference |
| **adaptive** | **0.904** | **0.860** | **0.954** | **+0.004** |
| log | 0.878 | 0.891 | 0.865 | **−0.022** |
| dog | 0.864 | 0.843 | 0.887 | **−0.036** |

Log and dog cleanly **lost** on these hard images despite passing the
synthetic regression gate. The unit-test touching-pair case (Gaussian
blob, σ=(4,3), sep=10) is not representative of real pupae, which have
more irregular shapes and internal texture that LoG/DoG sharpening picks
up as spurious secondary peaks.

Adaptive mode is the only variant that beats v1, and only by +0.4 F1
points on these 5 images — within what I would call noise at the
instance level.

### Adaptive winner promoted to full 20 images

Config: `response_mode=adaptive, small_sigma=0.8, large_sigma=1.2,
allowed_abs_threshold=0.12` (everything else v1 default).

| run | matched | teacher_only | pred_only | R | P | F1 | rt |
|---|---|---|---|---|---|---|---|
| v1 smooth | 1933 | 242 | 82 | 0.889 | 0.959 | 0.923 | 68 ms |
| adaptive | 1949 | 226 | 89 | 0.896 | 0.956 | 0.925 | 77 ms |
| **Δ** | **+16** | **−16** | **+7** | **+0.007** | **−0.003** | **+0.002** | **+9 ms** |

That is real but tiny. The acceptance gate (R ≥ 0.94) is nowhere close.

### Paper ROI

Tried two configurations on the 20-image benchmark:

| run | matched | teacher_only | pred_only | R | P | F1 |
|---|---|---|---|---|---|---|
| v1 smooth | 1933 | 242 | 82 | 0.889 | 0.959 | 0.923 |
| v1 smooth + paper_roi (erode 3) | 1930 | 245 | 77 | 0.887 | 0.962 | 0.923 |
| adaptive + paper_roi (erode 3) | 1946 | 229 | 84 | 0.895 | 0.959 | 0.926 |

Paper ROI gets rid of ~5 false positives but also loses ~3 matches — the
ROI's inward erosion margin clips pupae that sit right at the paper
edge. Net effect on F1 is zero.

Swept `erode_margin_px` from 0 through 20. Best configuration still only
cuts 9 FPs from the 173 v1 pred_only count at erode=15, mostly because
the scanner bar on these scans is extremely narrow (6-8 native pixels)
and v1's existing `peak_edge_margin_px=4` already suppresses peaks
inside it.

### Column profile diagnosis on scan 20

I probed the intensity profile of scan 20 directly to understand the
scanner bar's geometry:

```
x=  0: mean=33.5
x=  5: mean=55.6
x= 10: mean=208.7   <-- scanner bar ends here
x= 15: mean=250.7
x= 20: mean=251.6
```

The scanner bar is only **6-8 pixels wide** at the left margin, and v1's
existing `peak_edge_margin_px=4` already kills peaks inside it. My
research doc from 2026-04-10 overestimated this failure mode by taking
low-resolution crop tiles at face value — the RGB values at the
"scanner-edge FP" teacher positions I identified earlier show **brown**
`(185, 150, 110)`, not dark chrome. Those teacher rows might actually
be real small pupae or brown debris right at the paper edge, not
scanner artifacts.

## The pitfall: bogus cleaned teacher

`scripts/rebuild_cleaned_teacher.py` with its default
`close_kernel_px=15` dropped **123 teacher rows** (5.7% of v8's labels),
which I initially read as "v8 has a huge scanner-edge problem". Wrong.

The actual bug: the kernel was too small. A pupa's major axis at native
resolution is ~40 pixels. `cv2.morphologyEx(MORPH_CLOSE, 15×15)` only
fills holes up to ~15 px across. So the paper mask had **holes at every
pupa** and `rebuild_cleaned_teacher.py` dropped random real pupae.

Probed on scan 25:

```
k=15: paper_frac=0.988  teacher_inside=106/114   <-- kernel too small,
                                                     drops 8 real pupae
k=31: paper_frac=1.000  teacher_inside=114/114   <-- kernel large enough
                                                     to fill pupa holes
```

But `k=31` is also too big — at that size the close bridges the paper
region to the 6-8 px scanner bar and the "paper" mask covers the entire
image, including the bar. There is **no kernel size** that both fills
pupa holes AND keeps the scanner bar separate, because the bar is
narrower than a pupa.

The cleaned teacher file has been renamed
`teacher_v8_20_instances_cleaned_BOGUS_DO_NOT_USE.csv` to prevent
accidental use. The rebuild script is kept for reference but its default
parameters must be reconsidered before the next attempt — and most
likely the whole approach (morphological paper mask) is wrong for this
image format.

## Why response sharpening failed in aggregate

The ultraplan bet that Gaussian smoothing was the bottleneck. The
synthetic test clearly shows that LoG and DoG preserve touching-Gaussian
bimodality that pure Gauss smoothing destroys — and that test passes
deterministically.

The problem is that real pupae are **not** pairs of clean Gaussian
bumps. At the pixel level they are irregular brown patches with internal
shading and anti-aliased edges. LoG / DoG sharpening treats that
internal structure as *also* being worth sharpening, so every pupa grows
secondary peaks at its own edges. The net result is:

* slightly more true peaks recovered (from touching pairs)
* **many more false peaks introduced** (from edge ringing on every
  single pupa)

You can see this directly in the sweep data: log mode's best
configuration gets **+40 extra matches** vs smooth but pays for it with
**−90 extra FPs**. F1 goes down.

Adaptive mode is the only sharpening variant that helps, because it
selectively applies a smaller sigma only to components below a size
threshold — effectively "slightly sharpen small isolated blobs, leave
everything else alone". The size threshold acts as a filter that keeps
the sharpening localized. But even adaptive can only recover +0.4 F1
points because most of the touching-pair failures are in *dense*
clusters, not *small* components.

## What this means for the next iteration

1. **Drop the "sharpen the response" direction.** LoG and DoG are
   falsified for this data. Adaptive gives a tiny win that is not worth
   keeping unless it is combined with something else.

2. **Drop paper ROI as the primary win.** The scanner bar is 6-8 px,
   and v1's `peak_edge_margin_px=4` already handles it. My earlier
   research doc overcounted this failure mode by trusting low-resolution
   crop tiles.

3. **Real recall ceiling without structural change ≈ F1 0.925.** Across
   5 variant modes × hundreds of hyperparameter combinations, the ceiling
   on 20 images is F1 ≈ 0.925 / R ≈ 0.90 / P ≈ 0.96. Further gains
   require *structural* work — not another sweep.

4. **The remaining failure mode is still the touching pair inside
   dense clusters.** Nothing I tried changed that. The disagreement
   diagnostic overlay on scan 25 still shows red circles tucked next to
   green matches in the middle band.

### Recommended next direction

Two options, ranked by plausible expected value:

**A. Shape-based peak verifier (medium effort, no training).**
For each peak candidate, extract a ~32×32 crop at native resolution,
compute a small set of shape features (aspect ratio, solidity, axis
lengths from central moments, local contrast), and reject candidates
whose shape doesn't match a "pupa prior". This would let us run
detection at a sharper (noisier) response map, catch more touching pairs
via their individual local maxima, and use the shape verifier to drop
the extra noise peaks that killed log/dog in the sweep.

**B. Tiny per-patch classifier on ambiguous blobs only (higher effort,
one-time training).** The option the handoff reserves. Only classify
patches where v1's splitter reports an ambiguous component. Inference
cost ~1-3 ms per patch, 50 patches per image, well inside the latency
budget. Dataset already exists at
`fresh_start_agent_handoff_2026-04-10/benchmarks/resolver_patches_*`.
Highest expected recall gain but highest engineering cost.

**Do not try another peak-detection reshuffle before (A).** The
infrastructure this iteration built (tests, sweep driver, paper_roi as
optional) is the right substrate for (A) — shape features can be a new
axis in the sweep grid.

## Files

New:
- `tests/fresh/__init__.py`
- `tests/fresh/conftest.py`
- `tests/fresh/test_eval_instances.py`
- `tests/fresh/test_detector_smoke.py`
- `tests/fresh/test_response_sharpen.py`
- `tests/fresh/test_paper_roi.py`
- `src/pupa_counter_fresh/paper_roi.py`
- `scripts/run_fresh_sweep.py`
- `scripts/rebuild_cleaned_teacher.py` (pitfall documented above)
- `configs/sweeps/log_dog_v1.yaml`
- `docs/FRESH_START_V2_RESEARCH_2026-04-10.md` (from previous turn, now
  superseded by this doc for the paper ROI parts — the visual-audit
  findings for scans 15/20/25 remain correct; the paper-ROI fix proposal
  does not)

Modified:
- `src/pupa_counter_fresh/response.py` — response_mode dispatcher
- `src/pupa_counter_fresh/detector.py` — new config fields, paper_roi
  wiring, v1-tuned defaults (the latent bug fix)
- `scripts/run_fresh_peak_detector.py` — new CLI flags

## Reproducibility

Run the full test suite:
```
pytest tests/fresh -q       # 32 tests, ~0.5 s
```

Rerun v1 baseline (bit-compatible with the original v1 run):
```
python scripts/run_fresh_peak_detector.py --run-name v1_check --use-component-split
```

Rerun the adaptive winner on 20 images:
```
python scripts/run_fresh_peak_detector.py \
    --run-name adaptive_check \
    --use-component-split \
    --response-mode adaptive \
    --adaptive-small-sigma 0.8 \
    --adaptive-large-sigma 1.2 \
    --allowed-threshold 0.12
```

Rerun the response-sharpening sweep on the 5 hardest images:
```
python scripts/run_fresh_sweep.py \
    --grid configs/sweeps/log_dog_v1.yaml \
    --out-dir /tmp/sweep_repro \
    --restrict scan_20260313_20 scan_20260313_15 scan_20260313_25 \
               scan_20260313_95 scan_20260313_22
```

## Commit strategy

This iteration is committed on `codex/fresh-response-sharpen-v1` branched
off `codex/fresh-start-peak-proposal-v1` (not on the v1 branch directly,
to keep the PR history clean and isolate the negative result). The v1
baseline commit `ce1c0ab` remains the authoritative state for anyone
who wants to reproduce the v1 numbers from the earlier doc.
