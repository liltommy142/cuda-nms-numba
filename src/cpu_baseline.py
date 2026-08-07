"""Compatibility façade for the CPU baseline public API and CLI.

New code should import the focused modules in :mod:`baseline` and :mod:`common`.
This file remains so existing scripts, notebooks, benchmarks, and V3 keep their
documented imports and command path.
"""

from baseline.cli import main
from baseline.core import benchmark, iou_one_to_many, run_cpu, verify
from baseline.yolov5_adapter import (
    RawCandidateSelection,
    load_raw_yolo_candidate_selection,
    load_raw_yolo_candidates,
    load_raw_yolov5_model,
    raw_yolo_predictions_to_selection,
    raw_yolo_predictions_to_candidates,
)
from common.candidates import load_synthetic_candidates


def load_data(n, seed=0):
    """Return legacy one-class synthetic arrays required by untouched V3."""
    boxes, scores, _ = load_synthetic_candidates(n, seed=seed)
    return boxes, scores


__all__ = [
    "benchmark",
    "iou_one_to_many",
    "load_data",
    "RawCandidateSelection",
    "load_raw_yolo_candidate_selection",
    "load_raw_yolo_candidates",
    "load_raw_yolov5_model",
    "raw_yolo_predictions_to_selection",
    "raw_yolo_predictions_to_candidates",
    "run_cpu",
    "verify",
]


if __name__ == "__main__":
    main()
