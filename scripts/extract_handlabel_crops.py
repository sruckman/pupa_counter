"""Extract per-failure crops from hand-labeled corrections.

For each row in ``hand_labels/corrections.csv`` with recorded errors we
produce one or more crops centered on the failure region. The goal is to
build a *new* hard-case dataset where every crop is traceable back to a
specific hand-labeled error (FP, miss, or oversplit) so that we can
A/B test remediation methods on focused regions instead of full scans.

Inputs
------
* ``hand_labels/corrections.csv`` — hand labels (FP indices + miss counts +
  free-text notes that reference pupae by "#N" / "N-N" / "N/N/N" style).
* An ``instances.csv`` from the iter5 run (no resolver_v2) so the 1-indexed
  row ordering matches the numbered overlays the user labeled against.
* The original scans on disk (``source_path`` column).

Outputs (under ``--out-dir``)
-----------------------------
* ``crops/<name>.png`` — clean crops of the scan region, cropped from the
  native image.
* ``overlays/<name>_overlay.png`` — the same crop with iter5 detections
  annotated (yellow circles + small numbered labels) for context.
* ``manifest.csv`` — one row per crop with:
    - ``crop_name`` (filename stem)
    - ``image_id``, ``scan_native_path``
    - ``anchor_kind`` (fp / miss_ref / oversplit / cluster)
    - ``anchor_numbers`` (comma-sep 1-indexed detection numbers this crop is about)
    - ``x0, y0, x1, y1`` in native coordinates
    - ``failure_category`` — short tag (shadow_fp, dirt_fp, oversplit, dense_miss, edge_miss, centroid_drift, imprint_fp, scanner_strip_fp)
    - ``n_true_hint`` — free-text hint from notes (may be None)
    - ``notes`` — copy of the original row note for context
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pupa_counter_fresh.preprocess import load_image_rgb  # noqa: E402


# ---------------------------------------------------------------------------
# Failure plan  —  the important hand-labeled cases, mapped to concrete crops.
# ---------------------------------------------------------------------------
#
# Each entry describes **one crop** we want to extract. Anchor numbers refer
# to 1-indexed detection rows in the iter5 instances.csv for that image. The
# crop bounding box is built by taking the centroid of those anchors, then
# expanding to include any additional numbered references in the same
# cluster, then padding by CROP_PADDING_PX.
#
# Categories:
#   shadow_fp        — scan_3 under-pupa shadows wrongly detected
#   oversplit        — one real pupa detected as 2+
#   dense_miss       — touching cluster with fewer detections than pupae
#   edge_miss        — miss in edge region (still countable per >=50% rule)
#   scanner_strip_fp — FPs along scanner bars
#   dirt_fp          — isolated dirt / scanner dust FPs
#   imprint_fp       — imprint / residue looks like a pupa
#   centroid_drift   — count correct but circle off-center
#

CROP_PADDING_PX = 110       # pad around anchor cluster in native pixels


CROP_PLANS: list[dict] = [
    # ======================= scan_3 =======================
    {
        "image_id": "scan_20260313_3",
        "anchor_kind": "fp",
        "anchors": [5],
        "crop_name": "scan3_fp5_shadow",
        "failure_category": "shadow_fp",
        "n_true_hint": "0 (shadow, not a pupa)",
        "note_tag": "shadow FP",
    },
    {
        "image_id": "scan_20260313_3",
        "anchor_kind": "fp",
        "anchors": [8],
        "crop_name": "scan3_fp8_shadow",
        "failure_category": "shadow_fp",
        "n_true_hint": "0 (shadow, not a pupa)",
        "note_tag": "shadow FP",
    },
    {
        "image_id": "scan_20260313_3",
        "anchor_kind": "oversplit",
        "anchors": [36, 37],
        "crop_name": "scan3_oversplit_36_37_yellow_residue",
        "failure_category": "oversplit",
        "n_true_hint": "1 (squished pupa + yellow residue)",
        "note_tag": "oversplit of residue pupa",
    },
    {
        "image_id": "scan_20260313_3",
        "anchor_kind": "miss_ref",
        "anchors": [14],
        "crop_name": "scan3_miss_near_14",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "missing pupa near #14",
    },
    {
        "image_id": "scan_20260313_3",
        "anchor_kind": "miss_ref",
        "anchors": [49],
        "crop_name": "scan3_miss_near_49",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "missing pupa near #49",
    },
    {
        "image_id": "scan_20260313_3",
        "anchor_kind": "miss_ref",
        "anchors": [52],
        "crop_name": "scan3_miss_near_52",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "missing pupa near #52",
    },
    {
        "image_id": "scan_20260313_3",
        "anchor_kind": "miss_ref",
        "anchors": [63, 64],
        "crop_name": "scan3_miss_near_63_64",
        "failure_category": "dense_miss",
        "n_true_hint": ">=3",
        "note_tag": "missing pupa near #63/#64",
    },
    {
        "image_id": "scan_20260313_3",
        "anchor_kind": "centroid_drift",
        "anchors": [15, 18, 19],
        "crop_name": "scan3_centroid_drift_15_18_19",
        "failure_category": "centroid_drift",
        "n_true_hint": "3 (count correct, circles off-center)",
        "note_tag": "circles off-center",
    },
    # ======================= scan_7 =======================
    {
        "image_id": "scan_20260313_7",
        "anchor_kind": "fp",
        "anchors": [4],
        "crop_name": "scan7_fp4_blank_area",
        "failure_category": "dirt_fp",
        "n_true_hint": "0 (blank area)",
        "note_tag": "blank area FP",
    },
    {
        "image_id": "scan_20260313_7",
        "anchor_kind": "fp",
        "anchors": [100],
        "crop_name": "scan7_fp100_dirt",
        "failure_category": "dirt_fp",
        "n_true_hint": "0 (dirt)",
        "note_tag": "dirt FP",
    },
    {
        "image_id": "scan_20260313_7",
        "anchor_kind": "miss_ref",
        "anchors": [5],
        "crop_name": "scan7_miss_V_pair_near_5",
        "failure_category": "dense_miss",
        "n_true_hint": "V-shape pair + missing",
        "note_tag": "V-shape pair miss",
    },
    {
        "image_id": "scan_20260313_7",
        "anchor_kind": "miss_ref",
        "anchors": [87],
        "crop_name": "scan7_miss_edge_bottom_left_87",
        "failure_category": "edge_miss",
        "n_true_hint": ">=2",
        "note_tag": "bottom-left edge miss",
    },
    {
        "image_id": "scan_20260313_7",
        "anchor_kind": "miss_ref",
        "anchors": [96],
        "crop_name": "scan7_miss_pair_near_96",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    # ======================= scan_8 =======================
    {
        "image_id": "scan_20260313_8",
        "anchor_kind": "miss_ref",
        "anchors": [66, 67],
        "crop_name": "scan8_miss_cluster_66_67",
        "failure_category": "dense_miss",
        "n_true_hint": "3 (triplet, 2 det + 1 miss)",
        "note_tag": "cluster of 3 with half-at-edge excluded",
    },
    {
        "image_id": "scan_20260313_8",
        "anchor_kind": "miss_ref",
        "anchors": [104],
        "crop_name": "scan8_miss_pair_near_104",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    # ======================= scan_10 =======================
    {
        "image_id": "scan_20260313_10",
        "anchor_kind": "oversplit",
        "anchors": [81, 82],
        "crop_name": "scan10_oversplit_81_82",
        "failure_category": "oversplit",
        "n_true_hint": "1 (single pupa oversplit)",
        "note_tag": "oversplit",
    },
    {
        "image_id": "scan_20260313_10",
        "anchor_kind": "oversplit",
        "anchors": [86, 87],
        "crop_name": "scan10_oversplit_86_87",
        "failure_category": "oversplit",
        "n_true_hint": "1 (single pupa oversplit)",
        "note_tag": "oversplit",
    },
    {
        "image_id": "scan_20260313_10",
        "anchor_kind": "miss_ref",
        "anchors": [50],
        "crop_name": "scan10_miss_pair_50",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_10",
        "anchor_kind": "miss_ref",
        "anchors": [51],
        "crop_name": "scan10_miss_pair_51",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_10",
        "anchor_kind": "miss_ref",
        "anchors": [73],
        "crop_name": "scan10_miss_pair_73",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_10",
        "anchor_kind": "miss_ref",
        "anchors": [88, 89, 90],
        "crop_name": "scan10_dense_triplet_88_89_90",
        "failure_category": "dense_miss",
        "n_true_hint": "6 (triplet w/ 3 miss, 0 det)",
        "note_tag": "dense triplet with 3 missing pupae, ZERO detections",
    },
    # ======================= scan_15 =======================
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "fp",
        "anchors": [3, 4, 5],
        "crop_name": "scan15_scanner_strip_fps_3_4_5",
        "failure_category": "scanner_strip_fp",
        "n_true_hint": "0 (scanner edge)",
        "note_tag": "scanner-edge FPs",
    },
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "fp",
        "anchors": [6],
        "crop_name": "scan15_fp6_blank",
        "failure_category": "dirt_fp",
        "n_true_hint": "0 (blank)",
        "note_tag": "blank FP",
    },
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "fp",
        "anchors": [10],
        "crop_name": "scan15_fp10_dirt",
        "failure_category": "dirt_fp",
        "n_true_hint": "0 (dirt)",
        "note_tag": "dirt FP",
    },
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "fp",
        "anchors": [106],
        "crop_name": "scan15_fp106_dirt",
        "failure_category": "dirt_fp",
        "n_true_hint": "0 (dirt)",
        "note_tag": "dirt FP",
    },
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "miss_ref",
        "anchors": [26],
        "crop_name": "scan15_miss_pair_26",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "dense pair miss",
    },
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "miss_ref",
        "anchors": [35],
        "crop_name": "scan15_miss_pair_35",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "dense pair miss",
    },
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "miss_ref",
        "anchors": [55],
        "crop_name": "scan15_miss_pair_55",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "dense pair miss",
    },
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "miss_ref",
        "anchors": [102, 103],
        "crop_name": "scan15_miss_near_102_103",
        "failure_category": "edge_miss",
        "n_true_hint": "3 (3+half, half excluded)",
        "note_tag": "edge cluster miss",
    },
    {
        "image_id": "scan_20260313_15",
        "anchor_kind": "centroid_drift",
        "anchors": [75],
        "crop_name": "scan15_centroid_drift_75",
        "failure_category": "centroid_drift",
        "n_true_hint": "1",
        "note_tag": "centroid drift",
    },
    # ======================= scan_20 =======================
    {
        "image_id": "scan_20260313_20",
        "anchor_kind": "fp",
        "anchors": [1, 3, 4],
        "crop_name": "scan20_scanner_edge_fps_1_3_4",
        "failure_category": "scanner_strip_fp",
        "n_true_hint": "0 (scanner edge)",
        "note_tag": "scanner-edge FPs (don't exist)",
    },
    {
        "image_id": "scan_20260313_20",
        "anchor_kind": "miss_ref",
        "anchors": [6],
        "crop_name": "scan20_miss_above_6",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss above #6",
    },
    {
        "image_id": "scan_20260313_20",
        "anchor_kind": "miss_ref",
        "anchors": [12],
        "crop_name": "scan20_miss_near_12",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss near #12",
    },
    {
        "image_id": "scan_20260313_20",
        "anchor_kind": "miss_ref",
        "anchors": [17, 18],
        "crop_name": "scan20_miss_triplet_17_18",
        "failure_category": "dense_miss",
        "n_true_hint": "3 (triplet, 1 missed)",
        "note_tag": "triplet with 1 miss",
    },
    {
        "image_id": "scan_20260313_20",
        "anchor_kind": "miss_ref",
        "anchors": [99, 108],
        "crop_name": "scan20_edge_miss_99_108",
        "failure_category": "edge_miss",
        "n_true_hint": ">=2",
        "note_tag": "edge proximity miss",
    },
    # ======================= scan_22 =======================
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "fp",
        "anchors": [1],
        "crop_name": "scan22_fp1_phantom",
        "failure_category": "dirt_fp",
        "n_true_hint": "0 (doesn't exist)",
        "note_tag": "phantom FP",
    },
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "cluster",
        "anchors": [86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 101],
        "crop_name": "scan22_mega_cluster_11pupae",
        "failure_category": "oversplit",
        "n_true_hint": "10 (11 det, 1 oversplit)",
        "note_tag": "mega cluster: 11 det, 10 real, 1 oversplit",
    },
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "miss_ref",
        "anchors": [23],
        "crop_name": "scan22_miss_pair_23",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "miss_ref",
        "anchors": [30],
        "crop_name": "scan22_miss_pair_30",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "miss_ref",
        "anchors": [34],
        "crop_name": "scan22_miss_pair_34",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "miss_ref",
        "anchors": [47],
        "crop_name": "scan22_miss_pair_47",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "miss_ref",
        "anchors": [52],
        "crop_name": "scan22_miss_pair_52",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "miss_ref",
        "anchors": [96],
        "crop_name": "scan22_miss_pair_96",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "left-touching pair miss",
    },
    {
        "image_id": "scan_20260313_22",
        "anchor_kind": "centroid_drift",
        "anchors": [130, 131, 132],
        "crop_name": "scan22_centroid_drift_130_131_132",
        "failure_category": "centroid_drift",
        "n_true_hint": "3 (count correct, circles off-center)",
        "note_tag": "centroid drift",
    },
    # ======================= scan_25 =======================
    {
        "image_id": "scan_20260313_25",
        "anchor_kind": "oversplit",
        "anchors": [1, 2],
        "crop_name": "scan25_oversplit_1_2",
        "failure_category": "oversplit",
        "n_true_hint": "1 (single pupa, 2 det)",
        "note_tag": "oversplit",
    },
    {
        "image_id": "scan_20260313_25",
        "anchor_kind": "miss_ref",
        "anchors": [59],
        "crop_name": "scan25_miss_pair_59",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_25",
        "anchor_kind": "miss_ref",
        "anchors": [107, 108, 109],
        "crop_name": "scan25_miss_quad_107_108_109",
        "failure_category": "dense_miss",
        "n_true_hint": "4 (4 real, 3 det)",
        "note_tag": "quad with 1 miss",
    },
    # ======================= scan_30 =======================
    {
        "image_id": "scan_20260313_30",
        "anchor_kind": "oversplit",
        "anchors": [52, 53, 54, 55],
        "crop_name": "scan30_oversplit_52_55",
        "failure_category": "oversplit",
        "n_true_hint": "3 (4 det, 1 oversplit)",
        "note_tag": "52-55 oversplit",
    },
    {
        "image_id": "scan_20260313_30",
        "anchor_kind": "fp",
        "anchors": [57, 58, 59],
        "crop_name": "scan30_imprint_57_58_59",
        "failure_category": "imprint_fp",
        "n_true_hint": "2 (1 FP = imprint)",
        "note_tag": "imprint half-pupa FP (new subtype)",
    },
    {
        "image_id": "scan_20260313_30",
        "anchor_kind": "oversplit",
        "anchors": [75, 76, 77],
        "crop_name": "scan30_oversplit_75_77_ambiguous",
        "failure_category": "oversplit",
        "n_true_hint": "2 (AMBIGUOUS)",
        "note_tag": "oversplit per parallelism rule",
    },
    {
        "image_id": "scan_20260313_30",
        "anchor_kind": "miss_ref",
        "anchors": [39, 40, 41, 42],
        "crop_name": "scan30_miss_39_42_cluster",
        "failure_category": "dense_miss",
        "n_true_hint": "5 (4 det, 1 miss)",
        "note_tag": "5-pupa cluster, 1 missed",
    },
    {
        "image_id": "scan_20260313_30",
        "anchor_kind": "miss_ref",
        "anchors": [26, 27],
        "crop_name": "scan30_edge_miss_26_27",
        "failure_category": "edge_miss",
        "n_true_hint": "4 (2 fully-visible edge misses)",
        "note_tag": "left-edge fully-visible misses",
    },
    {
        "image_id": "scan_20260313_30",
        "anchor_kind": "miss_ref",
        "anchors": [91],
        "crop_name": "scan30_miss_pair_91",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "touching pair miss",
    },
    {
        "image_id": "scan_20260313_30",
        "anchor_kind": "miss_ref",
        "anchors": [47, 48, 49],
        "crop_name": "scan30_miss_quad_47_49",
        "failure_category": "dense_miss",
        "n_true_hint": "4 (3 det, 1 miss)",
        "note_tag": "quad with 1 miss",
    },
    # ======================= scan_35 =======================
    {
        "image_id": "scan_20260313_35",
        "anchor_kind": "fp",
        "anchors": [1, 2],
        "crop_name": "scan35_dirt_fp_1_2",
        "failure_category": "dirt_fp",
        "n_true_hint": "1 (1 real + 1 dirt)",
        "note_tag": "dirt FP #2",
    },
    {
        "image_id": "scan_20260313_35",
        "anchor_kind": "fp",
        "anchors": [22, 23],
        "crop_name": "scan35_dirt_fp_22_23",
        "failure_category": "dirt_fp",
        "n_true_hint": "1 (1 real + 1 dirt)",
        "note_tag": "dirt FP #23",
    },
    {
        "image_id": "scan_20260313_35",
        "anchor_kind": "fp",
        "anchors": [49, 50],
        "crop_name": "scan35_dirt_fp_49_50",
        "failure_category": "dirt_fp",
        "n_true_hint": "1 (1 real + 1 dirt)",
        "note_tag": "dirt FP #50",
    },
    {
        "image_id": "scan_20260313_35",
        "anchor_kind": "fp",
        "anchors": [63, 66, 67, 69],
        "crop_name": "scan35_fp69_cluster",
        "failure_category": "oversplit",
        "n_true_hint": "3 (1 oversplit)",
        "note_tag": "unspecified 1 extra in this cluster",
    },
    {
        "image_id": "scan_20260313_35",
        "anchor_kind": "miss_ref",
        "anchors": [15, 16, 17],
        "crop_name": "scan35_miss_quad_15_16_17",
        "failure_category": "dense_miss",
        "n_true_hint": "4 (3 det + 1 miss)",
        "note_tag": "quad with 1 miss",
    },
    {
        "image_id": "scan_20260313_35",
        "anchor_kind": "miss_ref",
        "anchors": [56, 58],
        "crop_name": "scan35_miss_pair_56_58",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "pair miss",
    },
    {
        "image_id": "scan_20260313_35",
        "anchor_kind": "miss_ref",
        "anchors": [82],
        "crop_name": "scan35_miss_pair_82",
        "failure_category": "dense_miss",
        "n_true_hint": ">=2",
        "note_tag": "pair miss",
    },
]


def _load_iter5_instances(path: Path) -> dict[str, pd.DataFrame]:
    """Return a dict of per-image iter5 instances keyed by image_id.

    The dataframe row index + 1 is the 1-indexed detection number the user
    saw on the numbered overlay, so anchors can be looked up by integer.
    """
    df = pd.read_csv(path)
    out: dict[str, pd.DataFrame] = {}
    for image_id, group in df.groupby("image_id"):
        group = group.reset_index(drop=True)
        out[image_id] = group
    return out


def _anchor_xy(
    instances: pd.DataFrame, anchors: Iterable[int]
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for anchor in anchors:
        if anchor < 1 or anchor > len(instances):
            continue
        row = instances.iloc[anchor - 1]
        out.append((float(row["centroid_x"]), float(row["centroid_y"])))
    return out


def _crop_bbox(
    anchors_xy: list[tuple[float, float]], img_h: int, img_w: int, pad: int
) -> tuple[int, int, int, int]:
    xs = [x for x, _ in anchors_xy]
    ys = [y for _, y in anchors_xy]
    x0 = int(np.floor(min(xs))) - pad
    y0 = int(np.floor(min(ys))) - pad
    x1 = int(np.ceil(max(xs))) + pad
    y1 = int(np.ceil(max(ys))) + pad
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    return x0, y0, x1, y1


def _draw_overlay_for_crop(
    crop_rgb: np.ndarray,
    instances: pd.DataFrame,
    x0: int,
    y0: int,
    anchor_numbers: set[int],
) -> np.ndarray:
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, row in instances.iterrows():
        cx = float(row["centroid_x"]) - x0
        cy = float(row["centroid_y"]) - y0
        if cx < -10 or cy < -10 or cx > bgr.shape[1] + 10 or cy > bgr.shape[0] + 10:
            continue
        number = idx + 1
        is_anchor = number in anchor_numbers
        color = (0, 0, 255) if is_anchor else (0, 180, 255)  # red anchors, yellow others
        cv2.circle(bgr, (int(round(cx)), int(round(cy))), 10, color, 2, cv2.LINE_AA)
        cv2.putText(
            bgr,
            str(number),
            (int(round(cx)) + 12, int(round(cy)) - 6),
            font,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            str(number),
            (int(round(cx)) + 12, int(round(cy)) - 6),
            font,
            0.5,
            (0, 255, 255) if not is_anchor else (0, 50, 255),
            1,
            cv2.LINE_AA,
        )
    return bgr


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract hand-labeled failure crops.")
    parser.add_argument(
        "--iter5-instances",
        type=Path,
        default=REPO_ROOT / "tmp" / "iter5_for_crops" / "iter5" / "instances.csv",
        help="Path to iter5 instances.csv (no resolver_v2, matches numbered overlays).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "hand_labels_hard_cases",
    )
    parser.add_argument("--pad", type=int, default=CROP_PADDING_PX)
    args = parser.parse_args()

    iter5_by_image = _load_iter5_instances(args.iter5_instances)
    crops_dir = args.out_dir / "crops"
    overlays_dir = args.out_dir / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    scan_cache: dict[str, np.ndarray] = {}

    for plan in CROP_PLANS:
        image_id = plan["image_id"]
        if image_id not in iter5_by_image:
            print(f"  ! no iter5 rows for {image_id}, skipping {plan['crop_name']}")
            continue
        instances = iter5_by_image[image_id]
        source_path = str(instances.iloc[0]["source_path"])
        if image_id not in scan_cache:
            scan_cache[image_id] = load_image_rgb(source_path)
        rgb = scan_cache[image_id]
        h, w = rgb.shape[:2]

        anchors_xy = _anchor_xy(instances, plan["anchors"])
        if not anchors_xy:
            print(
                f"  ! no anchors resolved for {plan['crop_name']} "
                f"(requested {plan['anchors']}, iter5 has {len(instances)})"
            )
            continue
        x0, y0, x1, y1 = _crop_bbox(anchors_xy, h, w, args.pad)
        crop_rgb = rgb[y0:y1, x0:x1]
        if crop_rgb.size == 0:
            print(f"  ! empty crop for {plan['crop_name']}")
            continue
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        out_png = crops_dir / f"{plan['crop_name']}.png"
        cv2.imwrite(str(out_png), crop_bgr)

        overlay = _draw_overlay_for_crop(
            crop_rgb,
            instances,
            x0,
            y0,
            set(plan["anchors"]),
        )
        overlay_png = overlays_dir / f"{plan['crop_name']}_overlay.png"
        cv2.imwrite(str(overlay_png), overlay)

        manifest.append(
            {
                "crop_name": plan["crop_name"],
                "image_id": image_id,
                "scan_native_path": source_path,
                "anchor_kind": plan["anchor_kind"],
                "anchor_numbers": ",".join(str(a) for a in plan["anchors"]),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "failure_category": plan["failure_category"],
                "n_true_hint": plan["n_true_hint"],
                "note_tag": plan["note_tag"],
            }
        )
        print(f"  . {plan['crop_name']:<42} {plan['failure_category']:<18} "
              f"({x1-x0}x{y1-y0} @ {x0},{y0})")

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_name",
                "image_id",
                "scan_native_path",
                "anchor_kind",
                "anchor_numbers",
                "x0",
                "y0",
                "x1",
                "y1",
                "failure_category",
                "n_true_hint",
                "note_tag",
            ],
        )
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)
    print()
    print(f"wrote {len(manifest)} crops to {crops_dir}")
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
