# Webcam Roadmap 2026-04-09

This note captures the practical path from the current scan/photo pipeline to a
usable webcam workflow.

## Short Version

The current high-accuracy pipeline is **not** true real-time video inference.
On the stable `v8`-class annotated probe, the current end-to-end latency is
roughly `56-74 seconds per image`, with a mean of about `60 seconds`.

That means the right near-term target is **capture-and-process**, not
frame-by-frame live counting.

## What "live" should mean for this project

There are three different goals that sound like "live":

1. **Static batch processing**  
   Existing workflow. A saved image is processed offline.

2. **Live preview + triggered recount**  
   A webcam shows a live preview. The user presses a key, clicks a button, or
   waits for the view to stabilize. The system captures one frame, processes
   it, and returns counts a short time later.

3. **Continuous live video inference**  
   The system updates counts continuously while the scene changes.

Right now, goal 2 is realistic. Goal 3 is not yet realistic with the current
pipeline.

## Why the current pipeline is too slow for true live video

The current code path is optimized for accuracy on marked scans, not for webcam
latency:

- it builds several image views (`original`, `reference`, `normalized`, `clean`)
- it runs `Cellpose` on a relatively large page-sized image
- it may invoke dense-patch rescue and annotated rescue logic
- it writes many debug artifacts for QA

All of that is useful offline, but too heavy for true video-rate inference.

## Recommended webcam v1 architecture

The recommended first webcam version is:

1. Open webcam preview with OpenCV.
2. Detect or constrain the paper ROI in the preview.
3. Capture a frame on demand or after the view is stable for a short window.
4. Run a **lighter local pipeline** on that single frame.
5. Render counts and overlay.
6. Repeat after the user removes pupae or repositions the sheet.

This gives a workflow that is much closer to real lab use without requiring
full frame-by-frame inference.

## What should be different in webcam mode

Webcam mode should **not** reuse the full marked-scan pipeline unchanged.

It should simplify aggressively:

- assume a fixed camera / fixed sheet position if possible
- assume clean images with no blue annotation marks
- skip blue-mask-heavy rescue logic when unnecessary
- reduce image size before inference
- fix `Cellpose` diameter once camera scale is calibrated
- disable the heaviest debug exports by default

## Where cluster access still helps

Cluster access is unlikely to help the webcam UI itself, because webcam
capture, display, and interaction all need to happen locally.

Cluster access **does** help with:

- batch image processing
- parameter sweeps
- replaying historical data repeatedly
- training or fine-tuning future models
- generating review artifacts faster while developing locally

## Practical milestone plan

### Milestone 1: webcam capture shell

- live preview
- keypress to capture
- save the captured frame
- run existing pipeline on that frame

### Milestone 2: webcam-specific lightweight path

- fixed ROI
- lower max-side resolution
- fixed Cellpose diameter
- no blue-mark cleanup
- lighter postprocess only

### Milestone 3: quasi-live updates

- automatic recount only when the scene is stable
- pause updates when a hand enters the frame or the sheet moves

### Milestone 4: true continuous live

Only consider this after the simpler webcam versions work. This would likely
require:

- local change detection
- tracking between frames
- partial re-inference instead of whole-page reruns
- or a dedicated lighter detector

## Bottom line

The current pipeline is good enough to support a **webcam-assisted counting
workflow**, but not yet good enough for true low-latency live video inference.

The right next step is a capture-and-process webcam prototype, not full
real-time video counting.
