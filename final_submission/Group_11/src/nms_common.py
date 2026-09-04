"""Compatibility exports for the canonical NMS candidate contract."""

from common.candidates import (
    load_synthetic_candidates,
    stable_class_partitions,
    stable_score_order,
    validate_candidates,
    validate_iou_threshold,
)
from common.oracle import torchvision_class_aware_nms

__all__ = [
    "load_synthetic_candidates",
    "stable_class_partitions",
    "stable_score_order",
    "torchvision_class_aware_nms",
    "validate_candidates",
    "validate_iou_threshold",
]
