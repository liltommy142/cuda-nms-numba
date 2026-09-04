"""Class-aware orchestration and host greedy resolver for GPU V1."""

from __future__ import annotations

import numpy as np

from common.candidates import (
    stable_class_partitions,
    stable_score_order,
    validate_candidates,
    validate_iou_threshold,
)
from v1.kernel import compute_iou_matrix_gpu


def _run_single_class(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Run V1's full-matrix strategy for one score-sorted class partition."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    iou_matrix = compute_iou_matrix_gpu(np.ascontiguousarray(boxes, dtype=np.float32))
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep_ranks = []
    for rank in range(len(boxes)):
        if suppressed[rank]:
            continue
        keep_ranks.append(rank)
        suppressed[rank + 1 :] |= iou_matrix[rank, rank + 1 :] > iou_threshold
    return np.asarray(keep_ranks, dtype=np.int64)


def run_gpu_v1(boxes, scores, class_ids=None, iou_threshold: float = 0.5) -> np.ndarray:
    """Run class-aware hard NMS with V1's one-thread-per-pair IoU matrix."""
    if class_ids is None:
        class_ids = np.zeros(len(scores), dtype=np.int32)
    boxes, scores, class_ids = validate_candidates(boxes, scores, class_ids)
    threshold = validate_iou_threshold(iou_threshold)
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    kept = [
        indices[_run_single_class(boxes[indices], threshold)]
        for indices in stable_class_partitions(scores, class_ids)
    ]
    return stable_score_order(np.concatenate(kept), scores)
