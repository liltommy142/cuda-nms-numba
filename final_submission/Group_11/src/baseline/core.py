"""Serial CPU reference implementation for class-aware greedy hard NMS."""

from __future__ import annotations

import time

import numpy as np

from common.candidates import (
    load_synthetic_candidates,
    stable_class_partitions,
    stable_score_order,
    validate_candidates,
    validate_iou_threshold,
)
from common.oracle import torchvision_class_aware_nms


def iou_one_to_many(box, boxes):
    """Return IoU between one ``xyxy`` box and an array of ``xyxy`` boxes."""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1e-9)


def _run_single_class(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Run greedy hard NMS for one score-sorted class partition."""
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep_ranks: list[int] = []
    for rank in range(len(boxes)):
        if suppressed[rank]:
            continue
        keep_ranks.append(rank)
        remaining = np.flatnonzero(~suppressed[rank + 1 :]) + rank + 1
        if len(remaining):
            ious = iou_one_to_many(boxes[rank], boxes[remaining])
            suppressed[remaining[ious > iou_threshold]] = True
    return np.asarray(keep_ranks, dtype=np.int64)


def run_cpu(boxes, scores, class_ids=None, iou_threshold: float = 0.5) -> np.ndarray:
    """Run class-aware serial greedy hard NMS on one image's candidates."""
    if class_ids is None:
        class_ids = np.zeros(len(scores), dtype=np.int32)
    boxes, scores, class_ids = validate_candidates(boxes, scores, class_ids)
    threshold = validate_iou_threshold(iou_threshold)
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    kept = []
    for indices in stable_class_partitions(scores, class_ids):
        kept.append(indices[_run_single_class(boxes[indices], threshold)])
    return stable_score_order(np.concatenate(kept), scores)


def verify(boxes, scores, iou_threshold, keep, class_ids=None):
    """Compare CPU hard NMS with the torchvision per-class oracle."""
    if class_ids is None:
        class_ids = np.zeros(len(scores), dtype=np.int32)
    try:
        reference_keep = torchvision_class_aware_nms(
            boxes, scores, class_ids, iou_threshold
        )
    except ImportError:
        print("torchvision not installed -- skipping verification against reference NMS")
        return None

    matches = np.array_equal(keep, reference_keep)
    print(f"Reference (torchvision) kept {len(reference_keep)} boxes, ours kept {len(keep)}")
    print(f"Exact ordered match: {matches}")
    if not matches:
        print(f"  ours:   {keep.tolist()}")
        print(f"  oracle: {reference_keep.tolist()}")
    return matches


def benchmark(
    ns: tuple[int, ...] = (100, 1_000, 10_000),
    iou_threshold: float = 0.5,
    seed: int = 0,
    verify_result: bool = False,
) -> dict[int, float]:
    """Time NMS only on deterministic synthetic, class-labelled candidates."""
    print("benchmark_scope: nms_only_synthetic")
    print("candidate_source: deterministic_synthetic")
    print("timing_scope: candidate NMS only; excludes model inference")
    print(f"{'N':>8} | {'time (s)':>10}")
    print("-" * 21)

    results = {}
    for n in ns:
        boxes, scores, class_ids = load_synthetic_candidates(n, seed=seed)
        start = time.perf_counter()
        keep = run_cpu(boxes, scores, class_ids, iou_threshold)
        elapsed = time.perf_counter() - start
        results[n] = elapsed
        print(f"{n:>8} | {elapsed:>10.4f}")
        if verify_result and not verify(boxes, scores, iou_threshold, keep, class_ids):
            raise RuntimeError(f"torchvision verification failed for N={n}")
    return results
