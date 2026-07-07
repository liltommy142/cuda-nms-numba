"""Correctness tests for the CPU NMS baseline (CSC14116, topic A4).

Run with: pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))

from cpu_baseline import iou_one_to_many, load_data, run_cpu  # noqa: E402


def test_iou_identical_box_is_one():
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert np.isclose(iou_one_to_many(box, boxes)[0], 1.0)


def test_iou_non_overlapping_is_zero():
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    boxes = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert np.isclose(iou_one_to_many(box, boxes)[0], 0.0)


def test_iou_known_partial_overlap():
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    boxes = np.array([[5, 0, 15, 10]], dtype=np.float32)
    # intersection = 5x10 = 50, union = 100 + 100 - 50 = 150
    assert np.isclose(iou_one_to_many(box, boxes)[0], 50.0 / 150.0, atol=1e-4)


def test_run_cpu_suppresses_lower_score_duplicate():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    keep = run_cpu(boxes, scores, iou_threshold=0.5)
    assert list(keep) == [0]


def test_run_cpu_keeps_both_when_far_apart():
    boxes = np.array([[0, 0, 10, 10], [1000, 1000, 1010, 1010]], dtype=np.float32)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    keep = run_cpu(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {0, 1}


@pytest.mark.parametrize("n", [50, 200, 1000])
def test_matches_torchvision_reference(n):
    """CPU baseline must match torchvision.ops.nms exactly (required by proposal)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.ops import nms as torch_nms

    boxes, scores = load_data(n, seed=42)
    iou_threshold = 0.5

    ours = set(run_cpu(boxes, scores, iou_threshold).tolist())
    theirs = set(
        torch_nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold)
        .numpy()
        .tolist()
    )
    assert ours == theirs
