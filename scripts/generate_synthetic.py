"""Generate synthetic training data for the keypoint head.

1. Extract single-pupa prototypes from EXACT scans
2. Extract paper background patches
3. Synthesize pairs/triplets by compositing singles onto paper
4. Generate center heatmap labels (Gaussian at each center)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pupa_counter_fresh.preprocess import load_image_rgb, downscale, build_blue_mask
from pupa_counter_fresh.response import compute_response_map, build_allowed_mask

SCALE = 0.67
CROP_SIZE = 96
HEATMAP_SIZE = 48
SIGMA = 3.0  # Gaussian sigma for center heatmap


def _gaussian_peak(h, w, cy, cx, sigma):
    """Generate a 2D Gaussian peak centered at (cy, cx)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def extract_singles(scan_nums, min_area=160, max_area=280, min_sol=0.93):
    """Extract clean single-pupa crops at native resolution."""
    singles = []
    for sn in scan_nums:
        src = f"/Users/stephenyu/Documents/New project/data/probe_inputs/all_20/Scan_20260313 ({sn}).png"
        rgb = load_image_rgb(src)
        rw, _ = downscale(rgb, SCALE)
        hsv = cv2.cvtColor(rw, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rw, cv2.COLOR_RGB2LAB)
        bl = build_blue_mask(rw, hsv=hsv, lab=lab)
        rsp = compute_response_map(rw, bl, response_mode='adaptive',
            adaptive_small_sigma=0.6, adaptive_large_sigma=0.6,
            adaptive_area_threshold_px=500, hsv=hsv, lab=lab)
        msk = build_allowed_mask(rsp, abs_threshold=0.12, min_percentile=0.0)
        labels, n = ndi.label(msk > 0)
        sizes = np.bincount(labels.ravel())
        slices = ndi.find_objects(labels)
        inv = 1.0 / SCALE

        for c in range(1, n + 1):
            area = int(sizes[c])
            if area < min_area or area > max_area:
                continue
            sl = slices[c - 1]
            if sl is None:
                continue
            # Skip border-touching
            if sl[0].start == 0 or sl[1].start == 0:
                continue
            if sl[0].stop == rsp.shape[0] or sl[1].stop == rsp.shape[1]:
                continue
            cm = labels[sl] == c
            rp = regionprops(cm.astype(np.uint8))
            if not rp or float(rp[0].solidity or 0) < min_sol:
                continue
            # Native crop
            y0n = max(0, int(sl[0].start * inv))
            y1n = min(rgb.shape[0], int(sl[0].stop * inv))
            x0n = max(0, int(sl[1].start * inv))
            x1n = min(rgb.shape[1], int(sl[1].stop * inv))
            crop = rgb[y0n:y1n, x0n:x1n]
            # Native mask
            mask_up = cv2.resize(cm.astype(np.uint8) * 255,
                                  (crop.shape[1], crop.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            if crop.shape[0] < 15 or crop.shape[1] < 8:
                continue
            singles.append({
                'crop': crop,
                'mask': mask_up,
                'scan': sn,
                'center': (crop.shape[0] // 2, crop.shape[1] // 2),
            })
    return singles


def extract_paper_patches(scan_nums, patch_size=120, n_patches=200):
    """Extract clean paper background patches (no pupae)."""
    patches = []
    rng = np.random.default_rng(42)
    for sn in scan_nums:
        src = f"/Users/stephenyu/Documents/New project/data/probe_inputs/all_20/Scan_20260313 ({sn}).png"
        rgb = load_image_rgb(src)
        rw, _ = downscale(rgb, SCALE)
        hsv = cv2.cvtColor(rw, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rw, cv2.COLOR_RGB2LAB)
        bl = build_blue_mask(rw, hsv=hsv, lab=lab)
        rsp = compute_response_map(rw, bl, response_mode='adaptive',
            adaptive_small_sigma=0.6, adaptive_large_sigma=0.6,
            adaptive_area_threshold_px=500, hsv=hsv, lab=lab)
        msk = build_allowed_mask(rsp, abs_threshold=0.12, min_percentile=0.0)
        gate = msk > 0
        blue_gate = bl > 0
        # Find areas with no pupae and no blue ink
        forbidden = cv2.dilate((gate | blue_gate).astype(np.uint8), np.ones((30, 30), np.uint8))
        # Map to native
        inv = 1.0 / SCALE
        ps_work = int(patch_size * SCALE)
        h, w = forbidden.shape
        for _ in range(n_patches // len(scan_nums)):
            for attempt in range(50):
                y = rng.integers(20, h - ps_work - 20)
                x = rng.integers(20, w - ps_work - 20)
                if forbidden[y:y + ps_work, x:x + ps_work].sum() == 0:
                    y0n = int(y * inv)
                    x0n = int(x * inv)
                    patch = rgb[y0n:y0n + patch_size, x0n:x0n + patch_size]
                    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                        patches.append(patch)
                        break
    return patches


def synthesize_sample(singles, paper_patches, k, rng, overlap_frac=0.2):
    """Synthesize one training sample with k pupae.

    Returns (crop_96x96, heatmap_48x48).
    """
    paper = rng.choice(paper_patches).copy()
    ph, pw = paper.shape[:2]

    centers = []
    for i in range(k):
        single = rng.choice(singles)
        pupa_crop = single['crop'].copy()
        pupa_mask = single['mask'].copy()
        sh, sw = pupa_crop.shape[:2]

        # Random rotation
        angle = rng.uniform(0, 360)
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(sh * sin + sw * cos), int(sh * cos + sw * sin)
        M[0, 2] += (nw - sw) / 2
        M[1, 2] += (nh - sh) / 2
        rotated = cv2.warpAffine(pupa_crop, M, (nw, nh), borderValue=(255, 255, 255))
        rot_mask = cv2.warpAffine(pupa_mask, M, (nw, nh), borderValue=0)

        # Random position (centered in paper, with overlap control)
        margin = 10
        max_y = ph - nh - margin
        max_x = pw - nw - margin
        if max_y < margin or max_x < margin:
            continue
        py = rng.integers(margin, max(margin + 1, max_y))
        px = rng.integers(margin, max(margin + 1, max_x))

        # Composite
        mask_bool = rot_mask > 127
        region = paper[py:py + nh, px:px + nw]
        if region.shape[:2] != mask_bool.shape:
            continue
        region[mask_bool] = rotated[mask_bool]
        centers.append((py + nh / 2, px + nw / 2))

    if len(centers) < k:
        return None, None

    # Resize to CROP_SIZE
    crop = cv2.resize(paper[:CROP_SIZE, :CROP_SIZE] if ph >= CROP_SIZE and pw >= CROP_SIZE
                       else cv2.resize(paper, (CROP_SIZE, CROP_SIZE)),
                       (CROP_SIZE, CROP_SIZE))

    # Generate heatmap
    scale_y = HEATMAP_SIZE / ph
    scale_x = HEATMAP_SIZE / pw
    heatmap = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    for cy, cx in centers:
        hm_cy = cy * scale_y
        hm_cx = cx * scale_x
        heatmap += _gaussian_peak(HEATMAP_SIZE, HEATMAP_SIZE, hm_cy, hm_cx, SIGMA)
    heatmap = np.clip(heatmap, 0, 1)

    return crop.astype(np.float32) / 255.0, heatmap


def main():
    out_dir = REPO_ROOT / "tmp" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    exact_scans = ['3', '7', '8', '15', '20', '22', '25']

    print("Extracting singles...")
    singles = extract_singles(exact_scans)
    print(f"  {len(singles)} single-pupa prototypes")

    print("Extracting paper patches...")
    papers = extract_paper_patches(exact_scans)
    print(f"  {len(papers)} paper background patches")

    if not singles or not papers:
        print("ERROR: no singles or papers extracted!")
        return 1

    rng = np.random.default_rng(42)

    # Generate synthetic data
    crops = []
    heatmaps = []
    ks = []

    targets = {1: 500, 2: 1000, 3: 500}
    for k, n_target in targets.items():
        count = 0
        for _ in range(n_target * 3):  # over-generate to handle failures
            crop, hm = synthesize_sample(singles, papers, k, rng)
            if crop is not None:
                crops.append(crop)
                heatmaps.append(hm)
                ks.append(k)
                count += 1
                if count >= n_target:
                    break
        print(f"  k={k}: generated {count}/{n_target}")

    # Save
    crops = np.array(crops)
    heatmaps = np.array(heatmaps)
    ks = np.array(ks)

    np.save(str(out_dir / "crops.npy"), crops)
    np.save(str(out_dir / "heatmaps.npy"), heatmaps)
    np.save(str(out_dir / "ks.npy"), ks)

    print(f"\nSaved {len(crops)} synthetic samples to {out_dir}")
    print(f"  crops: {crops.shape}")
    print(f"  heatmaps: {heatmaps.shape}")
    print(f"  k distribution: {dict(zip(*np.unique(ks, return_counts=True)))}")

    # Save a few samples for visual inspection
    vis_dir = out_dir / "preview"
    vis_dir.mkdir(exist_ok=True)
    for i in range(min(20, len(crops))):
        crop_vis = (crops[i] * 255).astype(np.uint8)
        hm_vis = (heatmaps[i] * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(cv2.resize(hm_vis, (CROP_SIZE, CROP_SIZE)), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR), 0.6, hm_color, 0.4, 0)
        cv2.imwrite(str(vis_dir / f"sample_{i:03d}_k{ks[i]}.png"), overlay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
