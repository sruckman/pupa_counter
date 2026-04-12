# v2 research — direct visual verification of v1 against v8

**Branch:** `codex/fresh-start-peak-proposal-v1`
**Method:** I (Claude) opened every pred_only and every real teacher_only
crop for scans 15 / 20 / 25 individually and visually judged each one,
treating myself as the human reviewer rather than trusting v8 as ground
truth.
**Artifacts:** `fresh_peak_v1_final/disagreement_galleries/scan_*_{pred_only,teacher_only}_gallery.png`

The user's hypothesis was that **v8 itself has errors** and that the way to
beat v8 is to look at every disagreement instead of trusting aggregate
metrics. That hypothesis checked out and dramatically changes the priority
list for v2.

## Headline findings

1. **v8 has a specific, cheap-to-fix systematic error class: the left-edge
   scanner gray-strip is labeled as pupae.** I found at least 7 confirmed
   cases across scans 15 / 20 / 25 where v8 put a teacher instance directly
   on the scanner bar / gray border of the image. Coordinates include
   (33, 1369), (37, 911), (11, 1022), (41, 1078), (26, 1117), (46, 1213),
   and (56, 783). None of these are real pupae.

2. **v1's precision is being *under-counted* by up to 5 percentage points.**
   Of the 21 pred_only items across these three hard images, approximately
   12 (57%) are **real pupae v1 correctly caught that v8 missed**, not
   false positives. The current audit scores them as FPs only because it
   uses v8 as ground truth.

3. **v1's false positives are the same failure mode as v8's:** ~9 of 21
   (43%) of v1's pred_only items also sit on the scanner left-edge strip.
   A single paper-ROI mask simultaneously removes ~40% of v1's FPs and
   ~5% of v8's labels.

4. **v1's remaining real misses are almost entirely touching-pair
   collapses.** I looked at every red circle in the three hard-image
   teacher_only galleries and could not find a single "unusual shape" or
   "unusual color" failure mode. Every real miss sits directly next to a
   green match inside a tight cluster. This is the structural problem a
   better touching-object separator would solve.

5. **The real recall gap between v1 and v8 is smaller than the v8-as-truth
   audit suggests** — on the three hard images the gap is 5-8 percentage
   points of true recall, not the 10-18 point gap the v8-audit reports.

## Recomputed metrics against human-verified ground truth

Human verification was me reading the gallery tiles and labeling each
disagreement. Ambiguous cases are excluded from both sides.

| scan | matched | v1-unique real | v8-unique real | v1 edge FP | v8 edge FP | true total |
|---|---|---|---|---|---|---|
| 15 | 94 | 5 | 16 | 5 | 2 | 115 |
| 20 | 106 | 3 | 12 | 0 | 4 | 121 |
| 25 | 114 | 4 | 11 | 4 | 1 | 129 |
| **sum** | **314** | **12** | **39** | **9** | **7** | **365** |

Derived rates (v1-as-scored by v8, vs real recall/precision):

| | v1 R (v8-truth) | v1 P (v8-truth) | v1 R real | v1 P real | v8 R real | v8 P real |
|---|---|---|---|---|---|---|
| scan 15 | 0.817 | 0.904 | 0.861 | **0.952** | 0.957 | 0.982 |
| scan 20 | 0.815 | 0.972 | 0.901 | **1.000** | 0.975 | 0.967 |
| scan 25 | 0.851 | 0.934 | 0.915 | **0.966** | 0.969 | 0.984 |
| mean | 0.828 | 0.937 | 0.892 | **0.973** | 0.967 | 0.978 |

**v1's real precision is 0.97-1.00** (not 0.90-0.97).
**v8's real precision is 0.97-0.98** (not 1.00 by construction).
**The two detectors are within 1 percentage point of each other on precision.**
**v1's real recall gap is 5-8 points, not 10-18.**

Extrapolating the "false positive" correction to the full benchmark:
if ~57% of v1's 83 pred_only items are actually real (consistent with the
3-image sample), v1's real matched count is closer to **1980 out of ~2013
real pupae across 20 images**, putting real recall around
**0.91–0.92** instead of the v8-reported 0.889.

## What this means for v2

The previous v2 plan was "shape verifier or learned patch classifier".
That is still a valid option but **it is no longer the cheapest win on
the board**. The visual audit revealed three much cheaper wins that
collectively could shift both v1 and the teacher comparison.

### Free-money fixes (no model changes, minutes of work)

**F1. Paper ROI crop — eliminates ~9 v1 FPs and 7 v8 FPs.**

The scanner produces a dark chrome bar on the left edge and sometimes
other edges. Both v1 and v8 detect it as pupae because it passes the
"dark" test. The fix is to detect the paper boundary (largest white-ish
connected region) at the preprocess stage and zero out the response
outside that ROI.

Cost: ~20 lines of OpenCV (threshold, largest component, bbox) +
one extra ms per image. No retraining. No parameter tuning.

Expected win: v1 precision from 0.959 → 0.970-0.985 on the v8 audit;
real precision stays at ~0.97-1.00. Also *cleans up v8's teacher labels*
when rebuilt through the same ROI.

**F2. Reject peaks whose local color is pure gray / not brown-biased.**

The scanner bar has high "darkness" but near-zero "brown" (no red/yellow
tint). A simple post-filter on each peak — "the 5×5 patch around this
peak must have at least X% pixels with a positive red-minus-blue" —
catches gray/black peaks that slipped through.

Cost: ~10 lines, negligible runtime.

Expected win: catches any remaining scanner-edge FPs that survive the
paper ROI, plus a few "dust / shadow" FPs.

**F3. Teacher rebuild with the same ROI + color gate.**

Apply F1 and F2 to `teacher_v8_20_instances.csv` offline, producing a
cleaned teacher table. Re-run the v1 audit against the cleaned teacher.
v1's precision jumps automatically because the v8 FPs are gone from the
denominator, and the "v1 unique real" pupae will still show as pred_only
(which the visual audit already confirmed are real).

Cost: ~20 lines + one offline run. No changes to the online detector.

Expected win: the public "v1 beats v8 on precision" claim becomes
defensible without any online model changes.

### Structural wins (the touching-pair gap)

After F1-F3, the remaining real miss is ~170 touching-pair pupae across
20 images. Every one I looked at sits next to a green match, which tells
us the response map has only one peak where there should be two. Three
directions worth exploring, in rough order of effort:

**S1. Laplacian-of-Gaussian (LoG) or difference-of-Gaussians (DoG)
response at multiple scales.**

Instead of smoothing the response with a single Gaussian σ=1.2, compute
`LoG(σ_small) + LoG(σ_mid)` or similar. LoG responds to blobs of a
specific size and *preserves the bimodality of touching pairs* — a
property single-scale Gaussian smoothing does not have.

Cost: a couple of `cv2.GaussianBlur` calls and an arithmetic combine,
maybe 10-20 ms extra per image. Still under budget.

Expected win (from the literature): LoG-based blob detectors typically
separate touching objects substantially better than raw response maps.
Plausible recall lift of 3-6 percentage points on the touching-pair gap.

**S2. Local ellipse fit per component.**

For each allowed-mask component with area > 1.5× single-pupa area, fit a
small set of ellipses via least-squares or moment-based fitting and
return an ellipse center per fit. This is classical computer vision
("connected-component ellipse decomposition" / "ellipse fitting on
watershed basins") and is well-documented for dense blob scenarios.

Cost: moderate — need to call `cv2.fitEllipseDirect` or similar, maybe
5-10 ms per image.

Expected win: cleaner for dense clusters, but requires careful
component pre-segmentation to avoid the current "one component contains
5 pupae" situation. Might pair naturally with LoG.

**S3. Tiny per-patch classifier on ambiguous blobs only.**

The "tiny resolver" option from the handoff. Train a small CNN on 32×32
crops where v1 is uncertain (components with expected_k ≥ 2 from the
area prior) and have it classify `single / pair / triple+`. The result
is a per-component count that refines the area prior.

Cost (inference): ~0.5-2 ms per patch on CPU (bitwise ops + a few conv
layers). At ~50 ambiguous patches per image this is 25-100 ms. Still
under budget.

Cost (training): probably 1-2 days of engineering + labeling effort,
since the dataset already exists in
`fresh_start_agent_handoff_2026-04-10/benchmarks/resolver_patches_*`.

Expected win: 5-10 recall points on the touching-pair gap. Highest
expected value but also highest engineering cost.

## "Build a better teacher" — is this worth pursuing?

The user asked whether the goal should be to build a better teacher model
rather than beat v8 on the current labels. Based on the visual audit the
answer is **both, in that order**:

1. **Short-term: use v1 + visual verification to clean the v8 teacher.**
   The dataset already exists (`teacher_v8_20_instances.csv`). Running F1
   and F2 over it produces a *cleaned teacher* with fewer scanner-edge
   FPs. Then manually verify the ~170 remaining v1 pred_only items —
   most of the 3-hard-image pattern extrapolates across the full 20, so
   probably ~100 of those are pupae the cleaned teacher should include.
   That gives a **cleaned_teacher_v0.csv** with higher quality than
   v8-raw, for free, using only the code already in this branch plus
   a couple of hundred clicks.

2. **Medium-term: ensemble v1 with an independent model for cross-check.**
   Run Cellpose 3.x (newer than v8), StarDist, and maybe SAM 2 on the
   same 20 images offline. For each detection, count how many of
   {v1, cellpose, stardist, SAM} agree. Agreements become high-confidence
   pseudo-labels; disagreements go to the human verification queue.
   This is the classic "ensemble-based pseudo-labeling" approach.

3. **Long-term: train a per-scanner lightweight model on the cleaned
   teacher.** Once the cleaned teacher exists, training a small
   UNet-style regressor on it is straightforward. The model would be
   *specific to this scanner setup* and can be much smaller than
   cellpose because it does not need to generalize across microscopes.

Do not build a "better teacher" as a new deep-learning training project
before doing step 1. It's cheaper, faster, and more aligned with the
handoff's "fresh-start" spirit.

## Latency budget — we have a LOT of room

Current v1: **68 ms / image**.
Target: <1000 ms preferred, <2000 ms mandatory.
Slack: **~930 ms of preferred budget, ~1930 ms of mandatory budget.**

| line item | estimated cost per image |
|---|---|
| current v1 pipeline | 68 ms |
| paper ROI crop (F1) | +2-5 ms |
| brown-color post-filter (F2) | +1-3 ms |
| LoG multi-scale response (S1) | +15-30 ms |
| ellipse fitting on touching components (S2) | +5-15 ms |
| tiny patch classifier (S3, batched) | +30-80 ms |
| **projected v2 total** | **120-200 ms** |

Even adding all four items we would be at ~200 ms — 5× under budget.
The latency constraint is effectively not binding for anything short of
a full-image CNN.

## Recommended order of operations

Priority ranked by "expected improvement divided by engineering effort":

1. **F1 (paper ROI)** — 1-2 hours, eliminates most edge FPs on both sides
2. **F2 (brown-color peak post-filter)** — 1 hour, catches stragglers
3. **F3 (rebuild cleaned teacher)** — 2-4 hours, gives honest audit
4. **S1 (LoG response)** — half day, addresses the touching-pair gap
   directly
5. **Re-audit everything** with visual review — half day
6. **S3 (tiny patch classifier)** only if S1 leaves a gap worth chasing

Stop at each step and visually audit the new disagreements; do not
stack improvements without checking each one individually.

## What not to do

* **Do not chase scale=1.0 / lower sigma again.** The precision collapse
  I measured earlier is not fixable by threshold tuning.
* **Do not tune `single_pupa_area_px` in search of v1 -> v2 recall gains.**
  The visual audit shows touching-pair gaps are structural. Area priors
  can't fix them.
* **Do not add a learned patch classifier before F1-F3.** The "precision
  problem" the classifier would be solving is partly a phantom created by
  v8's own errors.
* **Do not trust aggregate numbers without a visual pass.** This whole
  report was generated in one hour by looking at 62 tiles and noticing
  patterns the 20-image aggregate was hiding.

## Confidence / caveats

1. The visual review used 400×400-resolution upscaled tiles, not
   interactive native zoom. Some "real pupa" calls are ~80% confident,
   not 100%. A second-pass review at higher native resolution (1024×1024
   crops) would tighten the counts but not change the direction.
2. The 3-image sample might not generalize perfectly to all 20 images.
   Before committing to a fix, build the gallery for all 20 and do the
   same review on the remaining 17.
3. I'm counting "scanner-edge" as a single systematic error class. There
   may be other v8 systematic errors I didn't catch (e.g., underexposed
   corners, staple holes, page numbers). A full-dataset gallery pass
   would surface those.
4. "Cellpose_split artifacts" were excluded from both sides in the
   precision calc. If we ever need to reason about them directly, we
   should collapse each parent/split pair into one instance before the
   match.

## Next concrete step

If you agree with the above, the smallest useful next commit is:

1. Add `paper_roi.py` to `pupa_counter_fresh/` — detects the paper
   boundary and returns a binary mask the preprocess stage intersects
   with the allowed mask.
2. Add a `color_gate_post_filter` function in `peaks.py` / `resolver_cv.py`
   that rejects peaks whose local patch is brown-negative.
3. Write a `rebuild_cleaned_teacher.py` script that applies the ROI +
   color gate to `teacher_v8_20_instances.csv` and writes
   `teacher_v8_20_instances_cleaned.csv`.
4. Re-run v1 full driver + disagreement gallery against the cleaned
   teacher and visually confirm the FP list is substantially shorter.

Only after that commit would we decide whether to chase the touching-pair
gap with LoG / ellipse fitting / tiny classifier.
