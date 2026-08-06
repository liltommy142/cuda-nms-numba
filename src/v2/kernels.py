"""CUDA kernels and device-layout helpers for packed-mask V2 NMS."""

from __future__ import annotations

import numpy as np

try:
    from numba import cuda
    from numba import float32 as nb_float32
    from numba import uint64 as nb_uint64

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    nb_float32 = nb_uint64 = None

    class _CudaDummy:
        def jit(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda function: function

    cuda = _CudaDummy()


IOU_THREADS = (16, 16)
BOXES_PER_MASK_WORD = 64


@cuda.jit
def _iou_matrix_coalesced_kernel(x1, y1, x2, y2, output):
    row, column = cuda.grid(2)
    n = x1.shape[0]
    if row >= n or column >= n:
        return
    ix1 = max(x1[row], x1[column]); iy1 = max(y1[row], y1[column])
    ix2 = min(x2[row], x2[column]); iy2 = min(y2[row], y2[column])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_row = (x2[row] - x1[row]) * (y2[row] - y1[row])
    area_column = (x2[column] - x1[column]) * (y2[column] - y1[column])
    union = area_row + area_column - intersection
    output[row, column] = intersection / union if union > 1e-9 else 0.0


@cuda.jit
def _nms_bitmask_kernel(x1, y1, x2, y2, masks, n, threshold):
    anchor_word = cuda.blockIdx.x; target_word = cuda.blockIdx.y; batch = cuda.blockIdx.z
    lane = cuda.threadIdx.x; anchor = anchor_word * BOXES_PER_MASK_WORD + lane
    if target_word < anchor_word:
        return
    active = anchor < n
    sx1 = cuda.shared.array(shape=(BOXES_PER_MASK_WORD,), dtype=nb_float32)
    sy1 = cuda.shared.array(shape=(BOXES_PER_MASK_WORD,), dtype=nb_float32)
    sx2 = cuda.shared.array(shape=(BOXES_PER_MASK_WORD,), dtype=nb_float32)
    sy2 = cuda.shared.array(shape=(BOXES_PER_MASK_WORD,), dtype=nb_float32)
    target = target_word * BOXES_PER_MASK_WORD + lane
    if target < n:
        sx1[lane] = x1[batch, target]; sy1[lane] = y1[batch, target]
        sx2[lane] = x2[batch, target]; sy2[lane] = y2[batch, target]
    cuda.syncthreads()
    if not active:
        return
    ax1 = x1[batch, anchor]; ay1 = y1[batch, anchor]
    ax2 = x2[batch, anchor]; ay2 = y2[batch, anchor]
    area_anchor = (ax2 - ax1) * (ay2 - ay1); value = nb_uint64(0)
    for bit in range(BOXES_PER_MASK_WORD):
        candidate = target_word * BOXES_PER_MASK_WORD + bit
        if candidate >= n:
            break
        if candidate > anchor:
            ix1 = max(ax1, sx1[bit]); iy1 = max(ay1, sy1[bit])
            ix2 = min(ax2, sx2[bit]); iy2 = min(ay2, sy2[bit])
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if intersection > 0.0:
                area_candidate = (sx2[bit] - sx1[bit]) * (sy2[bit] - sy1[bit])
                if intersection / (area_anchor + area_candidate - intersection) > threshold:
                    value |= nb_uint64(1) << nb_uint64(bit)
    masks[batch, target_word, anchor] = value


def _upload_batched_soa(boxes: np.ndarray):
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    return tuple(cuda.to_device(np.ascontiguousarray(boxes[:, :, axis])) for axis in range(4))


def compute_iou_matrix_gpu_v2(boxes: np.ndarray) -> np.ndarray:
    """Return the coalesced SoA kernel's full IoU matrix on the host."""
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    coordinates = [cuda.to_device(np.ascontiguousarray(boxes[:, axis])) for axis in range(4)]
    output = cuda.device_array((len(boxes), len(boxes)), dtype=np.float32)
    blocks = tuple((len(boxes) + size - 1) // size for size in IOU_THREADS)
    _iou_matrix_coalesced_kernel[blocks, IOU_THREADS](*coordinates, output)
    cuda.synchronize()
    return output.copy_to_host()
