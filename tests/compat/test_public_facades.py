"""Regression checks for the historic root-level module imports."""

import numpy as np


def test_cpu_package_core_matches_legacy_facade():
    import cpu_baseline
    from baseline.core import run_cpu

    assert cpu_baseline.run_cpu is run_cpu


def test_v1_public_iou_helper_returns_empty_matrix_without_cuda_launch():
    from gpu_v1 import compute_iou_matrix_gpu

    result = compute_iou_matrix_gpu(np.empty((0, 4), dtype=np.float32))
    assert result.shape == (0, 0)
    assert result.dtype == np.float32


def test_v2_public_iou_helper_returns_empty_matrix_without_cuda_launch():
    from gpu_v2 import compute_iou_matrix_gpu_v2

    result = compute_iou_matrix_gpu_v2(np.empty((0, 4), dtype=np.float32))
    assert result.shape == (0, 0)
    assert result.dtype == np.float32
