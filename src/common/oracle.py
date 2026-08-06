"""Reference hard-NMS implementation used only for verification."""

from __future__ import annotations

import numpy as np

from common.candidates import (
    stable_class_partitions,
    stable_score_order,
    validate_candidates,
    validate_iou_threshold,
)


def torchvision_class_aware_nms(
    boxes,
    scores,
    class_ids,
    iou_threshold: float,
) -> np.ndarray:
    """Apply torchvision greedy NMS independently per class and re-sort."""
    boxes, scores, class_ids = validate_candidates(boxes, scores, class_ids)
    threshold = validate_iou_threshold(iou_threshold)
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    import torch
    from torchvision.ops import nms

    kept: list[np.ndarray] = []
    for indices in stable_class_partitions(scores, class_ids):
        local_keep = nms(
            torch.from_numpy(boxes[indices]),
            torch.from_numpy(scores[indices]),
            threshold,
        ).cpu().numpy()
        kept.append(indices[local_keep])
    return stable_score_order(np.concatenate(kept), scores)
