"""Class-aware host orchestration for V2's GPU-built suppression masks."""

from __future__ import annotations

import numpy as np

from common.candidates import (
    stable_class_partitions,
    stable_score_order,
    validate_candidates,
    validate_iou_threshold,
)
from v2.kernels import (
    BOXES_PER_MASK_WORD,
    _nms_bitmask_kernel,
    _upload_batched_soa,
    cuda,
)


def _resolve_greedy_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """Resolve one packed pairwise mask in score order on the host."""
    suppressed = np.zeros(mask.shape[0], dtype=np.uint64)
    kept = []
    for anchor in range(n):
        word, bit = divmod(anchor, BOXES_PER_MASK_WORD)
        if suppressed[word] & (np.uint64(1) << np.uint64(bit)):
            continue
        kept.append(anchor)
        suppressed[word:] |= mask[word:, anchor]
    return np.asarray(kept, dtype=np.int64)


def _run_batched_single_class(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[np.ndarray]:
    """Execute one V2 mask launch for a fixed-size, one-class image batch."""
    batch_size, n, _ = boxes.shape
    if batch_size == 0:
        return []
    if n == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(batch_size)]
    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_boxes = np.take_along_axis(boxes, order[:, :, None], axis=1).astype(np.float32, copy=False)
    coordinates = _upload_batched_soa(sorted_boxes)
    words = (n + BOXES_PER_MASK_WORD - 1) // BOXES_PER_MASK_WORD
    device_masks = cuda.device_array((batch_size, words, n), dtype=np.uint64)
    _nms_bitmask_kernel[(words, words, batch_size), BOXES_PER_MASK_WORD](
        *coordinates, device_masks, n, np.float32(threshold)
    )
    cuda.synchronize()
    masks = device_masks.copy_to_host()
    return [order[index, _resolve_greedy_mask(masks[index], n)] for index in range(batch_size)]


def _run_single_class(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Execute V2 for one score partition of one image."""
    return _run_batched_single_class(boxes[None, :, :], scores[None, :], threshold)[0]


def _run_single_image(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    kept = [
        indices[_run_single_class(boxes[indices], scores[indices], threshold)]
        for indices in stable_class_partitions(scores, class_ids)
    ]
    return stable_score_order(np.concatenate(kept), scores)


def run_gpu_v2_batched(boxes, scores, class_ids=None, iou_threshold: float = 0.5) -> list[np.ndarray]:
    """Run class-aware V2 NMS for a fixed-size batch of candidate images."""
    boxes = np.asarray(boxes); scores = np.asarray(scores)
    if boxes.ndim != 3 or boxes.shape[2] != 4:
        raise ValueError("boxes must have shape (batch_size, n, 4)")
    if scores.ndim != 2 or scores.shape != boxes.shape[:2]:
        raise ValueError("scores must have shape (batch_size, n)")
    if class_ids is None:
        class_ids = np.zeros(scores.shape, dtype=np.int32)
    class_ids = np.asarray(class_ids)
    if class_ids.ndim != 2 or class_ids.shape != scores.shape:
        raise ValueError("class_ids must have shape (batch_size, n)")
    threshold = validate_iou_threshold(iou_threshold)
    normalized = [validate_candidates(boxes[i], scores[i], class_ids[i]) for i in range(len(boxes))]
    if not normalized:
        return []
    normalized_boxes = np.stack([item[0] for item in normalized])
    normalized_scores = np.stack([item[1] for item in normalized])
    normalized_classes = np.stack([item[2] for item in normalized])
    if normalized_boxes.shape[1] == 0:
        return [np.empty(0, dtype=np.int64) for _ in normalized]
    if np.all(normalized_classes == normalized_classes.flat[0]):
        return _run_batched_single_class(normalized_boxes, normalized_scores, threshold)
    return [
        _run_single_image(image_boxes, image_scores, image_classes, threshold)
        for image_boxes, image_scores, image_classes in zip(normalized_boxes, normalized_scores, normalized_classes)
    ]


def run_gpu_v2(boxes, scores, class_ids=None, iou_threshold: float = 0.5):
    """Run class-aware V2 on either one image or a fixed-size batch."""
    boxes = np.asarray(boxes); scores = np.asarray(scores)
    if boxes.ndim == 2:
        if class_ids is None:
            class_ids = np.zeros(len(scores), dtype=np.int32)
        boxes, scores, class_ids = validate_candidates(boxes, scores, class_ids)
        return _run_single_image(boxes, scores, class_ids, validate_iou_threshold(iou_threshold))
    if boxes.ndim == 3:
        return run_gpu_v2_batched(boxes, scores, class_ids, iou_threshold)
    raise ValueError("boxes must have shape (n, 4) or (batch_size, n, 4)")
