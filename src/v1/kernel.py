"""CUDA kernel for V1's deliberately naive full IoU matrix."""

from __future__ import annotations

import numpy as np

try:
    from numba import cuda

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    class _CudaDummy:
        def jit(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda function: function

    cuda = _CudaDummy()


THREADS_PER_BLOCK = (16, 16)


@cuda.jit
def _iou_matrix_kernel(boxes, iou_out):
    """Write IoU for every pair of input boxes into ``iou_out``."""
    row, column = cuda.grid(2)
    n = boxes.shape[0]
    if row >= n or column >= n:
        return

    x1 = max(boxes[row, 0], boxes[column, 0])
    y1 = max(boxes[row, 1], boxes[column, 1])
    x2 = min(boxes[row, 2], boxes[column, 2])
    y2 = min(boxes[row, 3], boxes[column, 3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    row_area = (boxes[row, 2] - boxes[row, 0]) * (boxes[row, 3] - boxes[row, 1])
    column_area = (boxes[column, 2] - boxes[column, 0]) * (boxes[column, 3] - boxes[column, 1])
    union = row_area + column_area - intersection
    iou_out[row, column] = intersection / union if union > 1e-9 else 0.0


def compute_iou_matrix_gpu(boxes: np.ndarray) -> np.ndarray:
    """Upload boxes, compute V1's full ``N×N`` IoU matrix, and copy it back."""
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    if boxes.shape == (0, 4):
        return np.empty((0, 0), dtype=np.float32)
    n = len(boxes)
    device_boxes = cuda.to_device(boxes)
    device_iou = cuda.device_array((n, n), dtype=np.float32)
    blocks = (
        (n + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0],
        (n + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1],
    )
    _iou_matrix_kernel[blocks, THREADS_PER_BLOCK](device_boxes, device_iou)
    cuda.synchronize()
    return device_iou.copy_to_host()
