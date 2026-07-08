"""Correctness tests for CPU baseline + GPU V1 NMS (CSC14116, topic A4).

Run with:
    pytest tests/                          # all tests
    pytest tests/ -k "not gpu"             # CPU tests only (no GPU needed)
    pytest tests/ -k "gpu"                 # GPU tests only
"""

import os
import sys

import numpy as np
import pytest

# ── make src/ importable ──────────────────────────────────────────────────────
_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, _SRC)

from cpu_baseline import iou_one_to_many, load_data, run_cpu  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_available() -> bool:
    try:
        from numba import cuda
        return cuda.is_available()
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="No CUDA GPU available (or numba not installed) — skipping GPU tests",
)

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False

requires_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="torch / torchvision not installed — skipping reference-match tests",
)


# ─────────────────────────────────────────────────────────────────────────────
# CPU baseline — IoU unit tests
# ─────────────────────────────────────────────────────────────────────────────

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
    # intersection = 5×10 = 50, union = 100 + 100 − 50 = 150
    assert np.isclose(iou_one_to_many(box, boxes)[0], 50.0 / 150.0, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# CPU baseline — greedy NMS unit tests
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# CPU baseline — reference match (requires torch + torchvision)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [50, 200, 1000])
def test_cpu_matches_torchvision_reference(n):
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


# ─────────────────────────────────────────────────────────────────────────────
# GPU V1 — IoU matrix unit tests
# ─────────────────────────────────────────────────────────────────────────────

@requires_gpu
def test_gpu_v1_iou_matrix_diagonal_is_one():
    """IoU(box_i, box_i) must equal 1.0 for every box."""
    from gpu_v1 import compute_iou_matrix_gpu

    boxes, _ = load_data(20, seed=0)
    iou_mat = compute_iou_matrix_gpu(boxes)
    assert np.allclose(np.diag(iou_mat), 1.0, atol=1e-4), \
        "Diagonal of IoU matrix should be all 1s"


@requires_gpu
def test_gpu_v1_iou_matrix_is_symmetric():
    """IoU is symmetric: IoU(i, j) == IoU(j, i)."""
    from gpu_v1 import compute_iou_matrix_gpu

    boxes, _ = load_data(30, seed=1)
    iou_mat = compute_iou_matrix_gpu(boxes)
    assert np.allclose(iou_mat, iou_mat.T, atol=1e-5), \
        "IoU matrix should be symmetric"


@requires_gpu
def test_gpu_v1_iou_matrix_matches_cpu():
    """GPU IoU values must be within 1e-4 of the CPU iou_one_to_many reference."""
    from gpu_v1 import compute_iou_matrix_gpu

    boxes, _ = load_data(50, seed=2)
    iou_mat_gpu = compute_iou_matrix_gpu(boxes)

    # Build reference matrix using cpu iou_one_to_many
    n = len(boxes)
    iou_mat_cpu = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        iou_mat_cpu[i] = iou_one_to_many(boxes[i], boxes)

    assert np.allclose(iou_mat_gpu, iou_mat_cpu, atol=1e-4), \
        "GPU IoU matrix should match CPU reference within 1e-4"


# ─────────────────────────────────────────────────────────────────────────────
# GPU V1 — NMS correctness tests
# ─────────────────────────────────────────────────────────────────────────────

@requires_gpu
def test_gpu_v1_suppresses_duplicate():
    """GPU V1 must suppress the lower-score duplicate box."""
    from gpu_v1 import run_gpu_v1

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    keep = run_gpu_v1(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {0}


@requires_gpu
def test_gpu_v1_keeps_non_overlapping():
    """GPU V1 must keep both boxes when they don't overlap."""
    from gpu_v1 import run_gpu_v1

    boxes = np.array([[0, 0, 10, 10], [1000, 1000, 1010, 1010]], dtype=np.float32)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    keep = run_gpu_v1(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {0, 1}


@requires_gpu
@pytest.mark.parametrize("n", [50, 200, 1000])
def test_gpu_v1_matches_cpu_baseline(n):
    """GPU V1 kept-box set must match cpu_baseline.run_cpu exactly."""
    from gpu_v1 import run_gpu_v1

    boxes, scores = load_data(n, seed=42)
    iou_threshold = 0.5

    cpu_keep = set(run_cpu(boxes, scores, iou_threshold).tolist())
    gpu_keep = set(run_gpu_v1(boxes, scores, iou_threshold).tolist())

    assert cpu_keep == gpu_keep, (
        f"N={n}: GPU V1 kept {len(gpu_keep)} boxes, CPU kept {len(cpu_keep)} boxes\n"
        f"  only in CPU: {sorted(cpu_keep - gpu_keep)}\n"
        f"  only in GPU: {sorted(gpu_keep - cpu_keep)}"
    )
