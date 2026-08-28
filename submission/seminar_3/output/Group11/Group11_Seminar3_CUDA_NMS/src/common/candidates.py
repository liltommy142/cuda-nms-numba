"""Canonical data contract and deterministic fixtures for hard NMS."""

from __future__ import annotations

import numpy as np


def validate_candidates(boxes, scores, class_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize and validate one image's candidate detections.

    Boxes use float32 ``xyxy`` coordinates with positive area; scores and class
    IDs are one-dimensional arrays aligned with the boxes. Empty input is
    accepted as an empty sequence.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0 and boxes.ndim == 1:
        boxes = boxes.reshape(0, 4)
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    class_ids = np.ascontiguousarray(class_ids, dtype=np.int32)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    expected_shape = (len(boxes),)
    if scores.shape != expected_shape or class_ids.shape != expected_shape:
        raise ValueError("scores and class_ids must have shape (N,)")
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
        raise ValueError("candidates must be finite")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("x2 must be greater than x1 and y2 must be greater than y1")
    return boxes, scores, class_ids


def validate_iou_threshold(iou_threshold: float) -> float:
    """Return a finite hard-NMS threshold in the closed interval ``[0, 1]``."""
    threshold = float(iou_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in [0, 1]")
    return threshold


def stable_score_order(indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Order original indices by descending score, then by input index."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return indices
    return indices[np.lexsort((indices, -scores[indices]))]


def stable_class_partitions(scores: np.ndarray, class_ids: np.ndarray) -> list[np.ndarray]:
    """Return each class partition in deterministic descending-score order."""
    indices = np.arange(len(scores), dtype=np.int64)
    return [
        stable_score_order(indices[class_ids == class_id], scores)
        for class_id in np.unique(class_ids)
    ]


def load_synthetic_candidates(
    n: int,
    seed: int = 0,
    num_classes: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic multi-class ``xyxy`` candidates for NMS tests."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 900, size=n)
    y1 = rng.uniform(0, 900, size=n)
    width = rng.uniform(10, 100, size=n)
    height = rng.uniform(10, 100, size=n)
    boxes = np.stack([x1, y1, x1 + width, y1 + height], axis=1).astype(np.float32)
    scores = rng.uniform(0, 1, size=n).astype(np.float32)
    class_ids = (np.arange(n, dtype=np.int32) % num_classes).copy()
    rng.shuffle(class_ids)
    return validate_candidates(boxes, scores, class_ids)
