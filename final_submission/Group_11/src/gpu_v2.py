"""Compatibility façade for the organized GPU V2 implementation."""

from v2.cli import _synthetic_batch, benchmark, benchmark_batched, main
from v2.core import _run_single_class as _run_gpu_v2_single_class
from v2.core import run_gpu_v2, run_gpu_v2_batched
from v2.kernels import NUMBA_AVAILABLE as _NUMBA_AVAILABLE
from v2.kernels import compute_iou_matrix_gpu_v2, cuda

__all__ = ["benchmark", "benchmark_batched", "compute_iou_matrix_gpu_v2", "run_gpu_v2", "run_gpu_v2_batched"]


if __name__ == "__main__":
    main()
