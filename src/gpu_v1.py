"""GPU V1 — Naive parallel IoU matrix kernel for Non-Maximum Suppression.

Strategy
--------
* GPU side  : compute the full N×N IoU matrix in one kernel launch.
              One CUDA thread handles one (i, j) pair → N² threads run
              simultaneously.  All pairs are independent, so there is no
              synchronisation overhead (embarrassingly parallel).
* CPU side  : greedy suppression loop that reads the precomputed IoU matrix
              with O(1) lookups instead of recomputing IoU each time.

This gives a significant speedup over the CPU baseline at large N because the
IoU-computation step — which dominates the CPU runtime — is fully offloaded to
the GPU.  The remaining CPU suppression loop is O(n²) in the number of *kept*
box pairs, which is much smaller than n in practice.

Usage
-----
    python src/gpu_v1.py                    # single run, N=1 000
    python src/gpu_v1.py --n 10000          # single run, N=10 000
    python src/gpu_v1.py --n 1000 --verify  # check against cpu_baseline
    python src/gpu_v1.py --benchmark        # CPU vs GPU V1 sweep
"""

import argparse
import os
import sys
import time

import numpy as np

# ── make cpu_baseline importable when this file is run directly ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_baseline import load_data, run_cpu  # noqa: E402

try:
    from numba import cuda
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# ── tunable constants ─────────────────────────────────────────────────────────
# 16×16 = 256 threads/block — a common sweet spot for 2-D grid kernels.
# Each thread block covers a 16×16 tile of the N×N IoU matrix.
_TPB = (16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA kernel
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit
def _iou_matrix_kernel(boxes, iou_out):
    """Compute iou_out[i, j] = IoU(boxes[i], boxes[j]) for every (i, j) pair.

    boxes   : (N, 4) float32 device array  [x1, y1, x2, y2]
    iou_out : (N, N) float32 device array  (pre-allocated, written by kernel)

    Grid shape  : (ceil(N/16), ceil(N/16))  blocks
    Block shape : (16, 16)                  threads
    """
    i, j = cuda.grid(2)
    n = boxes.shape[0]

    if i >= n or j >= n:
        return  # out-of-bounds guard for non-square grids

    # intersection top-left / bottom-right
    x1 = max(boxes[i, 0], boxes[j, 0])
    y1 = max(boxes[i, 1], boxes[j, 1])
    x2 = min(boxes[i, 2], boxes[j, 2])
    y2 = min(boxes[i, 3], boxes[j, 3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
    area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
    union = area_i + area_j - inter

    iou_out[i, j] = inter / union if union > 1e-9 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Host helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_matrix_gpu(boxes: np.ndarray) -> np.ndarray:
    """Upload *boxes* to the GPU, run the IoU kernel, return the (N, N) matrix
    on the host.

    Parameters
    ----------
    boxes : (N, 4) float32 ndarray  [x1, y1, x2, y2]

    Returns
    -------
    iou_matrix : (N, N) float32 ndarray  (host memory)
    """
    n = boxes.shape[0]
    # contiguous C-layout is required by Numba's CUDA backend
    d_boxes = cuda.to_device(np.ascontiguousarray(boxes, dtype=np.float32))
    d_iou = cuda.device_array((n, n), dtype=np.float32)

    bpg = (
        (n + _TPB[0] - 1) // _TPB[0],
        (n + _TPB[1] - 1) // _TPB[1],
    )
    _iou_matrix_kernel[bpg, _TPB](d_boxes, d_iou)
    cuda.synchronize()

    return d_iou.copy_to_host()


def run_gpu_v1(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """GPU V1 Non-Maximum Suppression.

    1. Sort boxes by score (stable, on CPU).
    2. Upload score-sorted boxes → GPU; compute N×N IoU matrix with the kernel.
    3. Download IoU matrix → CPU.
    4. Greedy suppression using *vectorized* NumPy row-slices  (no inner Python
       loop — this was the critical fix vs. the naïve nested-loop version).

    Parameters
    ----------
    boxes         : (N, 4) float32  [x1, y1, x2, y2]
    scores        : (N,)   float32  confidence scores
    iou_threshold : float           boxes with IoU > threshold are suppressed

    Returns
    -------
    keep : (K,) int64  indices of kept boxes in original array, descending score
    """
    n = len(boxes)
    order = np.argsort(-scores, kind="stable")   # stable: deterministic on score ties

    # Sort boxes/scores into score-descending order before uploading.
    # After this, row i of the IoU matrix corresponds to the i-th highest-score box.
    boxes_sorted = np.ascontiguousarray(boxes[order], dtype=np.float32)

    # ── GPU: compute full N×N IoU matrix (embarrassingly parallel) ────────────
    iou_matrix = compute_iou_matrix_gpu(boxes_sorted)  # shape (N, N)

    # ── CPU: vectorized greedy suppression ─────────────────────────────────
    # suppressed[i] = True means box at rank i has been suppressed.
    suppressed = np.zeros(n, dtype=bool)
    keep_ranks = []   # ranks (in sorted order) of kept boxes

    for i in range(n):
        if suppressed[i]:
            continue
        keep_ranks.append(i)
        if i + 1 < n:
            # KEY FIX: one NumPy broadcast instead of an inner Python loop.
            # iou_matrix[i, i+1:] gives all IoU values between box i and every
            # lower-score box in a single C-speed array operation.
            suppressed[i + 1 :] |= iou_matrix[i, i + 1 :] > iou_threshold

    # Map sorted ranks back to original box indices
    return order[np.array(keep_ranks, dtype=np.int64)]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(
    ns: tuple = (100, 1_000, 10_000),
    iou_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    """Print a side-by-side timing table: CPU baseline vs GPU V1.

    The first call to run_gpu_v1 triggers Numba JIT compilation; we warm up on
    a tiny batch so that compilation time is *not* counted in the benchmark.
    """
    # warm up: JIT compile the kernel on a small problem
    _boxes, _scores = load_data(10, seed=seed)
    _ = run_gpu_v1(_boxes, _scores, iou_threshold)

    header = f"{'N':>8} | {'CPU (s)':>10} | {'GPU V1 (s)':>12} | {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    results = {}
    for n in ns:
        boxes, scores = load_data(n, seed=seed)

        t0 = time.perf_counter()
        run_cpu(boxes, scores, iou_threshold)
        cpu_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_gpu_v1(boxes, scores, iou_threshold)
        gpu_t = time.perf_counter() - t0

        speedup = cpu_t / gpu_t
        results[n] = {"cpu": cpu_t, "gpu_v1": gpu_t, "speedup": speedup}
        print(f"{n:>8} | {cpu_t:>10.4f} | {gpu_t:>12.4f} | {speedup:>7.1f}x")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU V1 NMS — naive parallel IoU kernel (topic A4)")
    parser.add_argument("--n", type=int, default=1_000, help="number of boxes for a single run")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true", help="compare kept-set with cpu_baseline.run_cpu")
    parser.add_argument("--benchmark", action="store_true", help="sweep N in {100, 1000, 10000}")
    args = parser.parse_args()

    if args.benchmark:
        benchmark(iou_threshold=args.iou_threshold, seed=args.seed)
        return

    boxes, scores = load_data(args.n, seed=args.seed)

    print(f"Generated {len(boxes)} synthetic boxes.")
    print("Warming up GPU (JIT compile)…")
    _ = run_gpu_v1(boxes[:16], scores[:16], args.iou_threshold)

    t0 = time.perf_counter()
    keep = run_gpu_v1(boxes, scores, args.iou_threshold)
    elapsed = time.perf_counter() - t0
    print(f"GPU V1 NMS: kept {len(keep)}/{len(boxes)} boxes in {elapsed:.4f}s")

    if args.verify:
        cpu_keep = set(run_cpu(boxes, scores, args.iou_threshold).tolist())
        gpu_keep = set(keep.tolist())
        match = cpu_keep == gpu_keep
        print(f"Exact match with cpu_baseline: {match}")
        if not match:
            print(f"  only in CPU result : {sorted(cpu_keep - gpu_keep)}")
            print(f"  only in GPU result : {sorted(gpu_keep - cpu_keep)}")


if __name__ == "__main__":
    if not _NUMBA_AVAILABLE:
        print("ERROR: numba is not installed.  Run: pip install numba")
        sys.exit(1)
    if not cuda.is_available():
        print("ERROR: No CUDA-capable GPU detected.")
        sys.exit(1)
    main()
