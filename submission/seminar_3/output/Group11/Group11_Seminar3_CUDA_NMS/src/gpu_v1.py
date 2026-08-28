"""Compatibility façade for the organized GPU V1 implementation."""

from v1.cli import benchmark, main
from v1.core import _run_single_class as _run_gpu_v1_single_class
from v1.core import run_gpu_v1
from v1.kernel import NUMBA_AVAILABLE as _NUMBA_AVAILABLE
from v1.kernel import compute_iou_matrix_gpu, cuda

__all__ = ["benchmark", "compute_iou_matrix_gpu", "run_gpu_v1"]


if __name__ == "__main__":
    main()
