"""Fresh peak-first pupa detector.

This package is a ground-up rewrite that intentionally does *not* import from
``pupa_counter``. Its goal is to validate the peak-first hypothesis from the
2026-04-10 handoff (``brown/dark response map + NMS peak proposals`` beats the
old ``threshold + connected components`` line by a wide margin on hard images)
without inheriting any of the long heuristic chain.

Rules (see docs/IMPLEMENTATION_PLAN_FRESH_START.md):

* Peak-first, not component-first.
* v8 is an offline teacher only — never called inside ``run_detector``.
* Instance-level audit is the primary success metric.
* Geometry is deferred until instance detection is stable.
"""

from .detector import DetectorConfig, DetectorOutput, run_detector

__all__ = ["DetectorConfig", "DetectorOutput", "run_detector"]
