import cv2
import numpy as np
import math
import torch
import pandas as pd
import time
from typing import List, Tuple

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# ==============================
# CONFIGURATION
# ==============================

# Path to the MobileSAM checkpoint (downloaded mobile_sam.pt)
MOBILE_SAM_CHECKPOINT = r"C:\Users\sarah\OneDrive - UC Irvine\Long Lab\Pupa_Counter\mobile_sam.pt"

# Model type for MobileSAM (tiny ViT)
MOBILE_SAM_MODEL_TYPE = "vit_t"

# Camera index:
# 0 = usually built-in webcam
# 1 = often the first external USB camera
CAMERA_INDEX = 1   # try 1 for external; if it doesn't work, try 0 or 2

# ==============================
# ROI SETTINGS
# ==============================
# Tube is horizontal. Right side is "top"; left side is "bottom".
# IMPORTANT: ROI_Y + ROI_H must be <= camera frame height.
# With 1280x720, ROI_Y=100 and ROI_H=600 gives a box from y=100 to y=700.
ROI_X = 20
ROI_Y = 100
ROI_W = 600   # width (horizontal, "length" of tube)
ROI_H = 600   # height (vertical)

# Minimum mask area (in pixels) to be considered a pupa
MIN_PUPA_AREA = 40

# Parameters for the SAM mask generator
MASK_GENERATOR_PARAMS = {
    "points_per_side": 8,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.80,
    "min_mask_region_area": MIN_PUPA_AREA,
}

# Where to save the Excel file with logged frames
OUTPUT_EXCEL_PATH = r"pupae_counts.xlsx"


# ==============================
# MODEL LOADING
# ==============================

def load_mobile_sam(checkpoint_path: str, model_type: str = "vit_t") -> SamAutomaticMaskGenerator:
    """
    Load MobileSAM and return an automatic mask generator.
    """
    if not torch.cuda.is_available():
        print("WARNING: CUDA (GPU) not available. Running on CPU will be very slow.")
        device = "cpu"
    else:
        device = "cuda"

    print(f"Loading MobileSAM model '{model_type}' from checkpoint: {checkpoint_path}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=MASK_GENERATOR_PARAMS["points_per_side"],
        pred_iou_thresh=MASK_GENERATOR_PARAMS["pred_iou_thresh"],
        stability_score_thresh=MASK_GENERATOR_PARAMS["stability_score_thresh"],
        min_mask_region_area=MASK_GENERATOR_PARAMS["min_mask_region_area"],
    )

    print(f"MobileSAM loaded on device: {device}")
    return mask_generator


# ==============================
# SEGMENTATION USING MOBILE-SAM
# ==============================

def segment_pupae_with_sam(
    roi_bgr: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator
) -> List[Tuple[int, int, float]]:
    """
    Segment pupae in the ROI using MobileSAM.
    Returns a list of (cx, cy, area_pixels) in ROI coordinates.
    """

    # --- Optional sharpening to help with blur ---
    # You can comment this block out if you don't want sharpening.
    blurred = cv2.GaussianBlur(roi_bgr, (0, 0), 1.0)
    roi_bgr = cv2.addWeighted(roi_bgr, 1.5, blurred, -0.5, 0)
    # --------------------------------------------

    # Convert BGR (OpenCV) to RGB (for SAM)
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    # Generate masks (list of dicts)
    masks = mask_generator.generate(roi_rgb)

    centroids = []
    for m in masks:
        area = m.get("area", 0)
        if area < MIN_PUPA_AREA:
            continue

        seg = m.get("segmentation", None)  # boolean mask HxW
        if seg is None:
            continue

        # Compute centroid of the mask
        ys, xs = np.where(seg)  # rows (y), cols (x)
        if len(xs) == 0:
            continue

        cx = int(xs.mean())
        cy = int(ys.mean())
        centroids.append((cx, cy, float(area)))

    return centroids


# ==============================
# DRAW LEGEND / KEY
# ==============================

def draw_color_legend(img: np.ndarray, start_x: int, start_y: int):
    """
    Draws a legend (color key) on the given image.
    Colors must match the ones used in the main visualization.
    """
    # Legend entries: (label, color in BGR)
    legend = [
        ("Bottom (blue)",   (255, 0, 0)),    # blue
        ("Middle (yellow)", (0, 255, 255)),  # yellow
        ("Top (red)",       (0, 0, 255)),    # red
    ]

    x = start_x
    y = start_y
    box_size = 12
    spacing = 20

    for label, color in legend:
        # Colored square
        cv2.rectangle(img, (x, y), (x + box_size, y + box_size), color, -1)
        # Text next to it (black text)
        cv2.putText(
            img,
            label,
            (x + box_size + 5, y + box_size - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # black text
            1,
            cv2.LINE_AA,
        )
        y += spacing


# ==============================
# MAIN LOOP
# ==============================

def main():
    # Load MobileSAM
    mask_generator = load_mobile_sam(MOBILE_SAM_CHECKPOINT, MOBILE_SAM_MODEL_TYPE)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {CAMERA_INDEX}")
        return

    # IMPORTANT: Use a resolution that can fit ROI_H = 600
    # 1280 x 720 works: ROI_Y(100) + ROI_H(600) = 700 <= 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Pupa Counter (MobileSAM)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Press SPACE to log current frame to Excel.")
    print("Press 'q' to quit.")

    # For logging results to Excel when SPACE is pressed
    results = []      # list of dicts; one per logged frame
    frame_idx = 0     # counts processed frames
    logged_idx = 0    # counts logged frames
    session_top5_total = 0  # running total across logged frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Draw ROI rectangle for visualization
        cv2.rectangle(
            frame,
            (ROI_X, ROI_Y),
            (ROI_X + ROI_W, ROI_Y + ROI_H),
            (0, 255, 0),
            2
        )

        # Label above the rectangle
        cv2.putText(
            frame,
            "Pupa Sheet",
            (ROI_X, ROI_Y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),  # black
            2,
            cv2.LINE_AA,
        )

        # Draw legend above the box (shifted up from ROI_Y)
        legend_start_y = max(20, ROI_Y - 80)  # keep it on-screen
        draw_color_legend(frame, start_x=ROI_X, start_y=legend_start_y)

        # Extract ROI
        roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
        if roi.size == 0:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # Segment pupae with MobileSAM
        pupae = segment_pupae_with_sam(roi, mask_generator)
        # pupae: list of (cx, cy, area)

        # Horizontal tube:
        # width W along x-axis; left = bottom, right = top.
        H, W = roi.shape[:2]
        left_boundary = int(0.25 * W)   # boundary between bottom 25% and middle
        right_boundary = int(0.75 * W)  # boundary between middle and top 25%

        # Counters
        count_top = 0      # right side
        count_middle = 0
        count_bottom = 0   # left side

        # For computing top/bottom 5% (by distance from left = "bottom")
        pupa_info = []  # (cx, cy, distance_from_left)

        for (cx, cy, area) in pupae:
            # Region classification along width
            if cx < left_boundary:
                count_bottom += 1   # leftmost = bottom region
            elif cx < right_boundary:
                count_middle += 1
            else:
                count_top += 1      # rightmost = top region

            # distance_from_left (bottom) = cx
            distance_from_left = cx
            pupa_info.append((cx, cy, distance_from_left))

        total_pupae = len(pupa_info)

        # Determine top 5% (rightmost) and bottom 5% (leftmost) pupae
        top5_indices = set()
        bottom5_indices = set()

        if total_pupae > 0:
            k = max(1, math.ceil(0.05 * total_pupae))  # at least 1

            # Top 5%: largest distance_from_left (closest to right = top)
            sorted_desc = sorted(
                range(total_pupae),
                key=lambda i: pupa_info[i][2],
                reverse=True
            )
            top5_indices = set(sorted_desc[:k])

            # Bottom 5%: smallest distance_from_left (closest to left = bottom)
            sorted_asc = sorted(
                range(total_pupae),
                key=lambda i: pupa_info[i][2]
            )
            bottom5_indices = set(sorted_asc[:k])

        # Visualization on ROI
        vis_roi = roi.copy()

        # Draw region boundaries as vertical lines (in black)
        cv2.line(vis_roi, (left_boundary, 0), (left_boundary, ROI_H), (0, 0, 0), 2)
        cv2.line(vis_roi, (right_boundary, 0), (right_boundary, ROI_H), (0, 0, 0), 2)

        # Draw each pupa centroid with region colors (no special 5% highlighting)
        for idx, (cx, cy, _) in enumerate(pupa_info):
            if cx < left_boundary:
                color = (255, 0, 0)       # Blue for bottom (left) region [BGR]
            elif cx < right_boundary:
                color = (0, 255, 255)     # Yellow for middle region
            else:
                color = (0, 0, 255)       # Red for top (right) region

            radius = 5
            thickness = -1
            cv2.circle(vis_roi, (cx, cy), radius, color, thickness)

        # Overlay counts on ROI (black text)
        text_top = f"Top (right 25%): {count_top}"
        text_mid = f"Middle (50%): {count_middle}"
        text_bot = f"Bottom (left 25%): {count_bottom}"
        text_total = (
            f"Total: {total_pupae} | "
            f"Top 5%: {len(top5_indices)} | Bottom 5%: {len(bottom5_indices)}"
        )

        cv2.putText(vis_roi, text_bot, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis_roi, text_mid, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis_roi, text_top, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis_roi, text_total, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Put processed ROI back into frame
        frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W] = vis_roi

        # Show the final frame in full-screen window
        cv2.imshow(window_name, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit
            break

        if key == ord(' '):
            # SPACE pressed: log this frame to Excel
            logged_idx += 1
            session_top5_total += len(top5_indices)

            log_row = {
                "logged_index": logged_idx,     # which logged frame this is
                "frame_index": frame_idx,       # raw frame index
                "timestamp": time.time(),
                "total_pupae": total_pupae,
                "top_25_count": count_top,
                "middle_50_count": count_middle,
                "bottom_25_count": count_bottom,
                "top_5pct_count": len(top5_indices),
                "bottom_5pct_count": len(bottom5_indices),
                "session_top5_total": session_top5_total,
            }
            results.append(log_row)
            print(
                f"Logged frame {frame_idx} as entry {logged_idx}. "
                f"Top 5% in this frame: {len(top5_indices)} | "
                f"Session total Top 5%: {session_top5_total}"
            )

        # Increment frame index (for all processed frames)
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save results to Excel (only logged frames)
    if results:
        df = pd.DataFrame(results)
        df.to_excel(OUTPUT_EXCEL_PATH, index=False)
        print(f"Saved {len(results)} logged frames to: {OUTPUT_EXCEL_PATH}")
    else:
        print("No frames logged; nothing to save.")


if __name__ == "__main__":
    main()