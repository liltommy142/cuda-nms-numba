"""Correctness tests for CPU baseline + GPU V1 NMS (CSC14116, topic A4).

Run with:
    pytest tests/                          # all tests
    pytest tests/ -k "not gpu"             # CPU tests only (no GPU needed)
    pytest tests/ -k "gpu"                 # GPU tests only
"""

import os
import subprocess
import sys

import numpy as np
import pytest

# ── make src/ importable ──────────────────────────────────────────────────────
_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, _SRC)

from cpu_baseline import iou_one_to_many, load_data, run_cpu  # noqa: E402
from gpu_v3 import matrix_nms_reference  # noqa: E402
from nms_common import (  # noqa: E402
    load_synthetic_candidates,
    torchvision_class_aware_nms,
    validate_candidates,
)


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
# Canonical class-aware candidate contract
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_candidates_normalizes_dtypes_and_shapes():
    boxes, scores, class_ids = validate_candidates([[0, 0, 2, 2]], [0.9], [3])
    assert (boxes.dtype, scores.dtype, class_ids.dtype) == (
        np.float32,
        np.float32,
        np.int32,
    )
    assert boxes.shape == (1, 4)


def test_validate_candidates_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="x2 must be greater"):
        validate_candidates([[3, 0, 2, 1]], [0.9], [0])


def test_synthetic_candidates_are_deterministic_and_multiclass():
    first = load_synthetic_candidates(100, seed=8)
    second = load_synthetic_candidates(100, seed=8)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(np.unique(first[2])) > 1


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


def test_cpu_keeps_overlapping_boxes_from_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)
    assert run_cpu(boxes, scores, class_ids).tolist() == [0, 1]


@requires_torch
def test_cpu_matches_torchvision_per_class():
    boxes, scores, class_ids = load_synthetic_candidates(200, seed=12)
    actual = run_cpu(boxes, scores, class_ids)
    expected = torchvision_class_aware_nms(boxes, scores, class_ids, 0.5)
    assert np.array_equal(actual, expected)


def test_raw_yolo_predictions_convert_to_canonical_candidates():
    from cpu_baseline import raw_yolo_predictions_to_candidates

    raw_prediction = np.array([
        [20, 30, 10, 20, 0.8, 0.1, 0.9],
        [50, 60, 20, 10, 0.2, 0.8, 0.2],
    ], dtype=np.float32)
    boxes, scores, class_ids = raw_yolo_predictions_to_candidates(
        raw_prediction,
        conf_threshold=0.3,
    )
    assert np.array_equal(boxes, np.array([[15, 20, 25, 40]], dtype=np.float32))
    assert np.allclose(scores, np.array([0.72], dtype=np.float32))
    assert class_ids.tolist() == [1]


@requires_torch
def test_cpu_verify_uses_class_aware_oracle():
    from cpu_baseline import verify

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)
    keep = run_cpu(boxes, scores, class_ids)
    assert verify(boxes, scores, 0.5, keep, class_ids) is True


def test_cpu_cli_labels_synthetic_nms_scope():
    script = os.path.join(_SRC, "cpu_baseline.py")
    completed = subprocess.run(
        [sys.executable, script, "--source", "synthetic", "--benchmark", "--verify"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "benchmark_scope: nms_only_synthetic" in completed.stdout


def test_gpu_v1_wrapper_partitions_classes_without_cuda(monkeypatch):
    import gpu_v1

    def keep_every_rank(class_boxes, iou_threshold):
        return np.arange(len(class_boxes), dtype=np.int64)

    monkeypatch.setattr(gpu_v1, "_run_gpu_v1_single_class", keep_every_rank)
    boxes, scores, class_ids = load_synthetic_candidates(4, seed=5)
    scores = np.array([0.5, 0.9, 0.7, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1, 0, 1], dtype=np.int32)
    assert gpu_v1.run_gpu_v1(boxes, scores, class_ids).tolist() == [1, 3, 2, 0]


def test_gpu_v2_wrapper_partitions_batched_classes_without_cuda(monkeypatch):
    import gpu_v2

    def keep_every_rank(class_boxes, class_scores, iou_threshold):
        return np.arange(len(class_boxes), dtype=np.int64)

    monkeypatch.setattr(gpu_v2, "_run_gpu_v2_single_class", keep_every_rank)
    boxes, _, _ = load_synthetic_candidates(4, seed=6)
    scores = np.array([0.5, 0.9, 0.7, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1, 0, 1], dtype=np.int32)
    single = gpu_v2.run_gpu_v2(boxes, scores, class_ids)
    batched = gpu_v2.run_gpu_v2(
        np.stack([boxes, boxes]),
        np.stack([scores, scores]),
        np.stack([class_ids, class_ids]),
    )
    assert single.tolist() == [1, 3, 2, 0]
    assert [item.tolist() for item in batched] == [[1, 3, 2, 0], [1, 3, 2, 0]]


@requires_gpu
def test_gpu_v1_matches_cpu_and_torchvision_per_class():
    from gpu_v1 import run_gpu_v1

    boxes, scores, class_ids = load_synthetic_candidates(300, seed=22)
    assert np.array_equal(run_gpu_v1(boxes, scores, class_ids), run_cpu(boxes, scores, class_ids))


@requires_gpu
def test_gpu_v2_matches_cpu_per_class():
    from gpu_v2 import run_gpu_v2

    boxes, scores, class_ids = load_synthetic_candidates(300, seed=32)
    assert np.array_equal(run_gpu_v2(boxes, scores, class_ids), run_cpu(boxes, scores, class_ids))


@requires_gpu
def test_gpu_v2_batched_multiclass_matches_cpu():
    from gpu_v2 import run_gpu_v2

    samples = [load_synthetic_candidates(50, seed=i) for i in range(3)]
    actual = run_gpu_v2(
        np.stack([sample[0] for sample in samples]),
        np.stack([sample[1] for sample in samples]),
        np.stack([sample[2] for sample in samples]),
    )
    assert all(
        np.array_equal(result, run_cpu(*sample))
        for result, sample in zip(actual, samples)
    )


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

    ours = set(run_cpu(boxes, scores, iou_threshold=iou_threshold).tolist())
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
def test_gpu_v1_handles_empty_input():
    """N=0 must not crash (0-block kernel launch); previously unguarded/untested."""
    from gpu_v1 import run_gpu_v1

    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    keep = run_gpu_v1(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 0


@requires_gpu
@pytest.mark.parametrize("n", [50, 200, 1000, 10_000])
def test_gpu_v1_matches_cpu_baseline(n):
    """GPU V1 kept-box set must match cpu_baseline.run_cpu exactly.

    N=10_000 is included because that is the scale the 9.7x speedup claim is
    made at (see README.md) -- correctness was previously only verified up to
    N=1000, i.e. never actually checked at the scale the speedup is reported.
    """
    from gpu_v1 import run_gpu_v1

    boxes, scores = load_data(n, seed=42)
    iou_threshold = 0.5

    cpu_keep = set(run_cpu(boxes, scores, iou_threshold=iou_threshold).tolist())
    gpu_keep = set(run_gpu_v1(boxes, scores, iou_threshold=iou_threshold).tolist())

    assert cpu_keep == gpu_keep, (
        f"N={n}: GPU V1 kept {len(gpu_keep)} boxes, CPU kept {len(cpu_keep)} boxes\n"
        f"  only in CPU: {sorted(cpu_keep - gpu_keep)}\n"
        f"  only in GPU: {sorted(gpu_keep - cpu_keep)}"
    )


# -----------------------------------------------------------------------------
# GPU V2 -- IoU matrix unit tests (coalesced SoA kernel)
# -----------------------------------------------------------------------------

@requires_gpu
def test_gpu_v2_iou_helper_returns_host_array():
    """compute_iou_matrix_gpu_v2 must return a host ndarray, same as the V1
    helper. Four tests previously called .copy_to_host() on its result and
    would have crashed on the first GPU machine that ran them."""
    from gpu_v2 import compute_iou_matrix_gpu_v2

    boxes, _ = load_data(8, seed=0)
    out = compute_iou_matrix_gpu_v2(boxes)
    assert isinstance(out, np.ndarray), f"expected host ndarray, got {type(out)}"
    assert not hasattr(out, "copy_to_host")


@requires_gpu
def test_gpu_v2_iou_matrix_diagonal_is_one():
    """IoU(box_i, box_i) must equal 1.0 for every box (coalesced SoA kernel)."""
    from gpu_v2 import compute_iou_matrix_gpu_v2

    boxes, _ = load_data(20, seed=0)
    iou_mat = compute_iou_matrix_gpu_v2(boxes)
    assert np.allclose(np.diag(iou_mat), 1.0, atol=1e-4), \
        "Diagonal of V2 IoU matrix should be all 1s"


@requires_gpu
def test_gpu_v2_iou_matrix_is_symmetric():
    """IoU is symmetric: IoU(i, j) == IoU(j, i) (coalesced SoA kernel)."""
    from gpu_v2 import compute_iou_matrix_gpu_v2

    boxes, _ = load_data(30, seed=1)
    iou_mat = compute_iou_matrix_gpu_v2(boxes)
    assert np.allclose(iou_mat, iou_mat.T, atol=1e-5), \
        "V2 IoU matrix should be symmetric"


@requires_gpu
def test_gpu_v2_iou_matrix_matches_cpu():
    """GPU V2 coalesced IoU values must be within 1e-4 of the CPU reference."""
    from gpu_v2 import compute_iou_matrix_gpu_v2

    boxes, _ = load_data(50, seed=2)
    iou_mat_gpu = compute_iou_matrix_gpu_v2(boxes)

    n = len(boxes)
    iou_mat_cpu = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        iou_mat_cpu[i] = iou_one_to_many(boxes[i], boxes)

    assert np.allclose(iou_mat_gpu, iou_mat_cpu, atol=1e-4), \
        "GPU V2 SoA IoU matrix should match CPU reference within 1e-4"


@requires_gpu
def test_gpu_v2_iou_matrix_matches_v1():
    """GPU V2 coalesced IoU values must match GPU V1 values within 1e-5."""
    from gpu_v1 import compute_iou_matrix_gpu
    from gpu_v2 import compute_iou_matrix_gpu_v2

    boxes, _ = load_data(50, seed=3)
    iou_v1 = compute_iou_matrix_gpu(boxes)          # host array from V1
    iou_v2 = compute_iou_matrix_gpu_v2(boxes)

    assert np.allclose(iou_v1, iou_v2, atol=1e-5), \
        "GPU V2 IoU matrix should match GPU V1 IoU matrix within 1e-5"


# -----------------------------------------------------------------------------
# GPU V2 -- NMS correctness tests
# -----------------------------------------------------------------------------

@requires_gpu
def test_gpu_v2_suppresses_duplicate():
    """GPU V2 must suppress the lower-score duplicate box."""
    from gpu_v2 import run_gpu_v2

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    keep = run_gpu_v2(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {0}, \
        f"GPU V2 should keep only box 0, got {sorted(keep.tolist())}"


@requires_gpu
def test_gpu_v2_keeps_non_overlapping():
    """GPU V2 must keep both boxes when they do not overlap."""
    from gpu_v2 import run_gpu_v2

    boxes = np.array([[0, 0, 10, 10], [1000, 1000, 1010, 1010]], dtype=np.float32)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    keep = run_gpu_v2(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {0, 1}, \
        f"GPU V2 should keep both non-overlapping boxes, got {sorted(keep.tolist())}"


@requires_gpu
@pytest.mark.parametrize("batch_size", [3, 32])
def test_gpu_v2_batched_matches_cpu_at_partial_64_block(batch_size):
    """A fused batch launch must keep the CPU result for every image.

    N=50 deliberately exercises the partially-filled final 64-thread block;
    B=32 covers the catalog's batch-size target semantically.
    """
    from gpu_v2 import run_gpu_v2

    batch_boxes = []
    batch_scores = []
    for seed in range(11, 11 + batch_size):
        boxes, scores = load_data(50, seed=seed)
        batch_boxes.append(boxes)
        batch_scores.append(scores)
    boxes = np.stack(batch_boxes)
    scores = np.stack(batch_scores)

    gpu_keeps = run_gpu_v2(boxes, scores, iou_threshold=0.5)
    assert isinstance(gpu_keeps, list)
    assert len(gpu_keeps) == len(boxes)
    for image_idx, gpu_keep in enumerate(gpu_keeps):
        cpu_keep = run_cpu(boxes[image_idx], scores[image_idx], iou_threshold=0.5)
        assert set(gpu_keep.tolist()) == set(cpu_keep.tolist())


@requires_gpu
def test_gpu_v2_keeps_all_when_threshold_one():
    """With iou_threshold=1.0 only exact duplicates are suppressed."""
    from gpu_v2 import run_gpu_v2

    boxes, scores = load_data(20, seed=10)
    keep = run_gpu_v2(boxes, scores, iou_threshold=1.0)
    assert len(keep) == len(boxes), \
        (f"With threshold=1.0 all {len(boxes)} unique boxes should be kept, "
         f"got {len(keep)}")


@requires_gpu
@pytest.mark.parametrize("n", [50, 200, 1000, 10_000])
def test_gpu_v2_matches_cpu_baseline(n):
    """GPU V2 kept-box set must match cpu_baseline.run_cpu exactly.

    N=10_000 is included because that is the scale the >=15x speedup target
    is defined at (see README.md) -- correctness was previously only verified
    up to N=1000.
    """
    from gpu_v2 import run_gpu_v2

    boxes, scores = load_data(n, seed=42)
    iou_threshold = 0.5

    cpu_keep = set(run_cpu(boxes, scores, iou_threshold=iou_threshold).tolist())
    gpu_keep = set(run_gpu_v2(boxes, scores, iou_threshold=iou_threshold).tolist())

    assert cpu_keep == gpu_keep, (
        f"N={n}: GPU V2 kept {len(gpu_keep)} boxes, CPU kept {len(cpu_keep)} boxes\n"
        f"  only in CPU: {sorted(cpu_keep - gpu_keep)}\n"
        f"  only in GPU: {sorted(gpu_keep - cpu_keep)}"
    )


@requires_gpu
@pytest.mark.parametrize("n", [50, 200, 1000, 10_000])
def test_gpu_v2_matches_gpu_v1(n):
    """GPU V2 kept-box set must match GPU V1 exactly (cross-check)."""
    from gpu_v1 import run_gpu_v1
    from gpu_v2 import run_gpu_v2

    boxes, scores = load_data(n, seed=99)
    iou_threshold = 0.5

    v1_keep = set(run_gpu_v1(boxes, scores, iou_threshold=iou_threshold).tolist())
    v2_keep = set(run_gpu_v2(boxes, scores, iou_threshold=iou_threshold).tolist())

    assert v1_keep == v2_keep, (
        f"N={n}: GPU V2 kept {len(v2_keep)} boxes, GPU V1 kept {len(v1_keep)} boxes\n"
        f"  only in V1: {sorted(v1_keep - v2_keep)}\n"
        f"  only in V2: {sorted(v2_keep - v1_keep)}"
    )


@requires_gpu
@pytest.mark.parametrize("iou_threshold", [0.3, 0.5, 0.7])
def test_gpu_v2_various_thresholds(iou_threshold):
    """GPU V2 must match CPU baseline across different IoU thresholds."""
    from gpu_v2 import run_gpu_v2

    boxes, scores = load_data(200, seed=7)

    cpu_keep = set(run_cpu(boxes, scores, iou_threshold=iou_threshold).tolist())
    gpu_keep = set(run_gpu_v2(boxes, scores, iou_threshold=iou_threshold).tolist())

    assert cpu_keep == gpu_keep, (
        f"iou_threshold={iou_threshold}: GPU V2 kept {len(gpu_keep)}, "
        f"CPU kept {len(cpu_keep)}"
    )


@requires_gpu
def test_gpu_v2_matches_cpu_with_score_ties():
    """Many boxes sharing the exact same score -- exercises the stable-sort
    tie-break path end to end (previously untested; see QA_PREP.md 'Vi sao
    can stable sort?'). Both CPU and GPU sort with the same
    np.argsort(..., kind="stable") call before processing, so this mainly
    guards against off-by-one/rank-boundary bugs when many ranks are tied,
    not CPU/GPU divergence per se.
    """
    from gpu_v2 import run_gpu_v2

    boxes, _ = load_data(300, seed=5)
    # Collapse to 5 distinct score values so most boxes tie with several others.
    rng = np.random.default_rng(5)
    scores = (rng.integers(0, 5, size=300).astype(np.float32)) / 4.0

    iou_threshold = 0.5
    cpu_keep = set(run_cpu(boxes, scores, iou_threshold=iou_threshold).tolist())
    gpu_keep = set(run_gpu_v2(boxes, scores, iou_threshold=iou_threshold).tolist())

    assert cpu_keep == gpu_keep, (
        f"Tie-heavy scores: GPU V2 kept {len(gpu_keep)}, CPU kept {len(cpu_keep)}\n"
        f"  only in CPU: {sorted(cpu_keep - gpu_keep)[:10]}\n"
        f"  only in GPU: {sorted(gpu_keep - cpu_keep)[:10]}"
    )

# -----------------------------------------------------------------------------
# GPU V3 -- Matrix NMS unit tests
# -----------------------------------------------------------------------------

def test_matrix_nms_reference_non_overlapping_preserves_scores():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep, final_scores = matrix_nms_reference(boxes, scores, score_threshold=0.05)
    assert list(keep) == [0, 1]
    assert np.allclose(final_scores, scores)


def test_matrix_nms_reference_gaussian_duplicate_has_known_decay():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep, final_scores = matrix_nms_reference(
        boxes, scores, score_threshold=0.5, method="gaussian", sigma=2.0
    )
    assert list(keep) == [0]
    assert np.isclose(final_scores[1], 0.8 * np.exp(-0.5), atol=1e-6)


@pytest.mark.parametrize("method", ["linear", "gaussian"])
def test_matrix_nms_reference_validates_method_and_sigma(method):
    boxes, scores = load_data(2)
    with pytest.raises(ValueError):
        matrix_nms_reference(boxes, scores, method=method, sigma=0)
    with pytest.raises(ValueError):
        matrix_nms_reference(boxes, scores, method="invalid")

@requires_gpu
def test_gpu_v3_suppresses_duplicate():
    """V3 Matrix NMS must suppress exact duplicate boxes."""
    from gpu_v3 import run_gpu_v3_matrix_nms

    boxes = np.array([
        [10, 10, 50, 50],
        [10, 10, 50, 50],  # Exact duplicate
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    # Box 1 has score 0.8. IoU = 1.0. iou_max[1] = 1.0.
    # Linear: decay = (1-1) / (1-1) -> fallback to 0.0 or very small.
    # Gaussian: exp((1^2 - 1^2)/2) = 1.0. Wait!
    # If IoU(1, 1) = 1.0 and iou_max[1] = 1.0.
    # decay = exp(0) = 1.0.
    # Actually, in Matrix NMS, for box 1, we check min_{i<1} (f(IoU(0,1)) / f(iou_max[0])).
    # iou_max[0] = 0.0. IoU(0, 1) = 1.0.
    # Gaussian: decay_1 = exp((0 - 1)/2) = exp(-0.5) = 0.606.
    # Score becomes 0.8 * 0.606 = 0.485.
    # If we set score_threshold = 0.5, it will be suppressed.
    
    keep = run_gpu_v3_matrix_nms(boxes, scores, score_threshold=0.5, method="gaussian")
    assert len(keep) == 1, f"Expected 1 kept box, got {len(keep)}"
    assert keep[0] == 0, "Should keep the box with higher score"


@requires_gpu
def test_gpu_v3_keeps_non_overlapping():
    """V3 Matrix NMS must keep completely non-overlapping boxes."""
    from gpu_v3 import run_gpu_v3_matrix_nms

    boxes = np.array([
        [0, 0, 10, 10],
        [20, 20, 30, 30],
        [40, 40, 50, 50],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    # All IoUs are 0. Decay factors are 1.0.
    # Scores remain the same. All are above 0.05.
    keep = run_gpu_v3_matrix_nms(boxes, scores, score_threshold=0.05)
    
    assert len(keep) == 3, f"Expected 3 kept boxes, got {len(keep)}"


@requires_gpu
@pytest.mark.parametrize("n", [50, 200])
def test_gpu_v3_sanity(n):
    """V3 Matrix NMS runs without crashing and returns valid indices."""
    from gpu_v3 import run_gpu_v3_matrix_nms

    boxes, scores = load_data(n, seed=42)
    keep = run_gpu_v3_matrix_nms(boxes, scores, score_threshold=0.05, method="linear")

    assert len(keep) > 0, "Should keep at least some boxes"
    assert len(keep) <= n, "Kept boxes cannot exceed total boxes"
    assert len(set(keep)) == len(keep), "Kept indices must be unique"
    assert all(0 <= idx < n for idx in keep), "Indices must be in [0, N-1]"


@requires_gpu
@pytest.mark.parametrize("method", ["linear", "gaussian"])
def test_gpu_v3_matches_matrix_nms_reference(method):
    """GPU V3 must reproduce the tested CPU Matrix-NMS oracle on a small input."""
    from gpu_v3 import run_gpu_v3_matrix_nms

    boxes, scores = load_data(30, seed=31)
    expected_keep, _ = matrix_nms_reference(
        boxes, scores, score_threshold=0.05, method=method, sigma=2.0
    )
    actual_keep = run_gpu_v3_matrix_nms(
        boxes, scores, score_threshold=0.05, method=method, sigma=2.0
    )
    assert np.array_equal(actual_keep, expected_keep)
