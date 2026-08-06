"""GPU V2: batched greedy NMS with a coalesced, GPU-built bitmask.

Input boxes are sorted by score on the host.  A 3-D CUDA grid then builds the
pairwise suppression relation for every image in the batch:

* ``grid.z`` selects the image;
* each CUDA block owns 64 score-sorted anchor boxes;
* each thread produces one uint64 mask word, with one bit per target box.

The compressed mask is copied back and resolved with the serial greedy rule.
That final dependency is intentional: V2 accelerates greedy NMS's expensive
pairwise work; V3 changes the algorithm to fully parallel Matrix NMS.

Run a batch-size-32 benchmark on a CUDA machine with:
    python src/gpu_v2.py --n 10000 --batch-size 32 --benchmark
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_baseline import load_data, run_cpu  # noqa: E402
from gpu_v1 import run_gpu_v1              # noqa: E402
from nms_common import (  # noqa: E402
    stable_class_partitions,
    stable_score_order,
    validate_candidates,
    validate_iou_threshold,
)

try:
    from numba import cuda
    from numba import float32 as nb_float32
    from numba import uint64 as nb_uint64

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    nb_float32 = None
    nb_uint64 = None

    # Without this, `@cuda.jit` below would raise NameError at import time
    # (cuda is never bound) instead of reaching the friendly "ERROR: numba is
    # not installed" message in main() -- this dummy lets the module import
    # cleanly on a machine without numba; actually calling a kernel still
    # fails, but only once _NUMBA_AVAILABLE has already been checked.
    class _CudaDummy:
        def jit(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]      # bare `@cuda.jit` usage
            return lambda f: f      # parametrized `@cuda.jit(...)` usage

    cuda = _CudaDummy()

# The standalone IoU helper is retained for numerical tests.
_IOU_THREADS = (16, 16)
# One warp-friendly bitmask word represents 64 target boxes.
_BOXES_PER_MASK_WORD = 64


# ---------------------------------------------------------------------------
# CUDA kernel 1 -- Coalesced IoU matrix (SoA layout)
# ---------------------------------------------------------------------------

@cuda.jit
def _iou_matrix_coalesced_kernel(x1, y1, x2, y2, iou_out):
    """Compute iou_out[i, j] = IoU(box_i, box_j) with coalesced SoA reads.

    SoA layout: x1[N], y1[N], x2[N], y2[N] (four 1-D arrays).
    Consecutive threads in a warp read consecutive elements:
        thread i -> x1[i], thread i+1 -> x1[i+1]
    => single 128-byte L2 transaction per warp (COALESCED).
    Vs V1 AoS where thread stride=16 bytes => multiple transactions (NON-COALESCED).

    Parameters
    ----------
    x1, y1, x2, y2 : (N,) float32 device arrays -- SoA box coordinates
    iou_out         : (N, N) float32 device array -- pre-allocated output

    Grid  : (ceil(N/16), ceil(N/16)) blocks
    Block : (16, 16) threads
    """
    i, j = cuda.grid(2)
    n = x1.shape[0]

    if i >= n or j >= n:
        return  # bounds guard

    xi1 = x1[i];  yi1 = y1[i];  xi2 = x2[i];  yi2 = y2[i]
    xj1 = x1[j];  yj1 = y1[j];  xj2 = x2[j];  yj2 = y2[j]

    ix1 = max(xi1, xj1)
    iy1 = max(yi1, yj1)
    ix2 = min(xi2, xj2)
    iy2 = min(yi2, yj2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_i = (xi2 - xi1) * (yi2 - yi1)
    area_j = (xj2 - xj1) * (yj2 - yj1)
    union  = area_i + area_j - inter

    iou_out[i, j] = inter / union if union > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# CUDA kernel 2 -- batched suppression-mask construction
# ---------------------------------------------------------------------------

@cuda.jit
def _nms_bitmask_kernel(x1, y1, x2, y2, mask_out, n, iou_threshold):
    """Write one packed suppression word for each anchor box.

    ``mask_out[batch, target_word, anchor]`` bit ``k`` is one exactly when
    ``anchor`` suppresses ``target_word * 64 + k``.  Inputs arrive score
    sorted, so only targets with a larger index can be suppressed.
    """
    anchor_word = cuda.blockIdx.x
    target_word = cuda.blockIdx.y
    batch_idx = cuda.blockIdx.z
    lane = cuda.threadIdx.x

    anchor = anchor_word * _BOXES_PER_MASK_WORD + lane

    # This branch is uniform across the entire block, so returning before the
    # barrier is safe.  Earlier target words only contain higher-score boxes.
    if target_word < anchor_word:
        return

    # A final word may be partially full.  Inactive lanes must still reach the
    # barrier; they return only after the target tile is ready.
    is_active_anchor = anchor < n

    # One coalesced load per coordinate populates a target tile reused by all
    # 64 anchors in this CUDA block.
    sx1 = cuda.shared.array(shape=(_BOXES_PER_MASK_WORD,), dtype=nb_float32)
    sy1 = cuda.shared.array(shape=(_BOXES_PER_MASK_WORD,), dtype=nb_float32)
    sx2 = cuda.shared.array(shape=(_BOXES_PER_MASK_WORD,), dtype=nb_float32)
    sy2 = cuda.shared.array(shape=(_BOXES_PER_MASK_WORD,), dtype=nb_float32)

    target_to_load = target_word * _BOXES_PER_MASK_WORD + lane
    if target_to_load < n:
        sx1[lane] = x1[batch_idx, target_to_load]
        sy1[lane] = y1[batch_idx, target_to_load]
        sx2[lane] = x2[batch_idx, target_to_load]
        sy2[lane] = y2[batch_idx, target_to_load]
    cuda.syncthreads()

    if not is_active_anchor:
        return

    # Read the anchor only after the bounds check above.
    xi1 = x1[batch_idx, anchor]
    yi1 = y1[batch_idx, anchor]
    xi2 = x2[batch_idx, anchor]
    yi2 = y2[batch_idx, anchor]
    area_i = (xi2 - xi1) * (yi2 - yi1)

    mask_val = nb_uint64(0)

    for bit in range(_BOXES_PER_MASK_WORD):
        target = target_word * _BOXES_PER_MASK_WORD + bit
        if target >= n:
            break
        if target > anchor:
            xj1 = sx1[bit]
            yj1 = sy1[bit]
            xj2 = sx2[bit]
            yj2 = sy2[bit]

            ix1 = max(xi1, xj1)
            iy1 = max(yi1, yj1)
            ix2 = min(xi2, xj2)
            iy2 = min(yi2, yj2)

            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            inter = inter_w * inter_h

            if inter > 0.0:
                area_j = (xj2 - xj1) * (yj2 - yj1)
                union = area_i + area_j - inter
                if (inter / union) > iou_threshold:
                    mask_val |= nb_uint64(1) << nb_uint64(bit)

    mask_out[batch_idx, target_word, anchor] = mask_val



# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

def _boxes_to_soa_device(boxes: np.ndarray):
    """Transpose (N, 4) AoS ndarray to four (N,) SoA device arrays.

    SoA layout enables coalesced global-memory reads in the IoU kernel.

    Returns
    -------
    d_x1, d_y1, d_x2, d_y2 : Numba CUDA device arrays, each (N,) float32
    """
    b = np.ascontiguousarray(boxes, dtype=np.float32)
    return (
        cuda.to_device(np.ascontiguousarray(b[:, 0])),
        cuda.to_device(np.ascontiguousarray(b[:, 1])),
        cuda.to_device(np.ascontiguousarray(b[:, 2])),
        cuda.to_device(np.ascontiguousarray(b[:, 3])),
    )


def _batched_boxes_to_soa_device(boxes: np.ndarray):
    """Upload ``(B, N, 4)`` boxes as four contiguous ``(B, N)`` arrays."""
    b = np.ascontiguousarray(boxes, dtype=np.float32)
    return (
        cuda.to_device(np.ascontiguousarray(b[:, :, 0])),
        cuda.to_device(np.ascontiguousarray(b[:, :, 1])),
        cuda.to_device(np.ascontiguousarray(b[:, :, 2])),
        cuda.to_device(np.ascontiguousarray(b[:, :, 3])),
    )


def compute_iou_matrix_gpu_v2(boxes: np.ndarray) -> np.ndarray:
    """Run the coalesced SoA IoU kernel alone and return the (N, N) matrix on host.

    Exists separately from run_gpu_v2 so tests can check the coalesced kernel's
    numerical output (diagonal, symmetry, match vs CPU/V1) in isolation from the
    bitmask suppression pipeline.
    """
    n = boxes.shape[0]
    d_x1, d_y1, d_x2, d_y2 = _boxes_to_soa_device(boxes)
    d_iou = cuda.device_array((n, n), dtype=np.float32)

    blocks = (
        (n + _IOU_THREADS[0] - 1) // _IOU_THREADS[0],
        (n + _IOU_THREADS[1] - 1) // _IOU_THREADS[1],
    )
    _iou_matrix_coalesced_kernel[blocks, _IOU_THREADS](
        d_x1, d_y1, d_x2, d_y2, d_iou
    )
    cuda.synchronize()

    return d_iou.copy_to_host()


def _resolve_greedy_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """Apply greedy NMS to one image's packed GPU suppression mask.

    The mask contains pairwise relations, not final keep decisions.  We must
    visit score ranks in order because a box can suppress later boxes only if
    it was itself kept.  ``suppressed`` stores the running union of mask words.
    """
    num_words = mask.shape[0]
    suppressed = np.zeros(num_words, dtype=np.uint64)
    keep_ranks = []

    for anchor in range(n):
        word = anchor // _BOXES_PER_MASK_WORD
        bit = anchor % _BOXES_PER_MASK_WORD
        is_suppressed = (
            suppressed[word] & (np.uint64(1) << np.uint64(bit))
        ) != 0
        if is_suppressed:
            continue

        keep_ranks.append(anchor)
        # Words before ``word`` are intentionally uninitialized on the device:
        # an anchor never suppresses a higher-score (earlier) target.
        suppressed[word:] |= mask[word:, anchor]

    return np.asarray(keep_ranks, dtype=np.int64)


def _run_gpu_v2_batched_single_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[np.ndarray]:
    """Run the original V2 batched mask kernel for one class per image.

    Parameters are ``boxes`` with shape ``(B, N, 4)`` and ``scores`` with
    shape ``(B, N)``.  Every item may keep a different number of boxes, so the
    result is a list of ``B`` index arrays in the corresponding original-image
    index space.  The CUDA work is batched; the unavoidable greedy bitwise-OR
    resolution remains one short CPU loop per image.
    """
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    if boxes.ndim != 3 or boxes.shape[2] != 4:
        raise ValueError("boxes must have shape (batch_size, n, 4)")
    if scores.ndim != 2 or scores.shape != boxes.shape[:2]:
        raise ValueError("scores must have shape (batch_size, n)")

    batch_size, n, _ = boxes.shape
    if batch_size == 0:
        return []
    if n == 0:
        return [np.array([], dtype=np.int64) for _ in range(batch_size)]

    order = np.argsort(-scores, axis=1, kind="stable")
    boxes_sorted = np.take_along_axis(
        boxes, order[:, :, np.newaxis], axis=1
    ).astype(np.float32, copy=False)

    # ``grid.z`` selects an image; x/y blocks select anchor/target mask words.
    d_x1, d_y1, d_x2, d_y2 = _batched_boxes_to_soa_device(boxes_sorted)
    num_words = (n + _BOXES_PER_MASK_WORD - 1) // _BOXES_PER_MASK_WORD

    # Only the upper triangle is written.  The resolver below never reads the
    # uninitialized lower triangle, so no host-to-device zero-fill is needed.
    d_mask = cuda.device_array((batch_size, num_words, n), dtype=np.uint64)
    _nms_bitmask_kernel[(num_words, num_words, batch_size), _BOXES_PER_MASK_WORD](
        d_x1, d_y1, d_x2, d_y2, d_mask, n, np.float32(iou_threshold)
    )
    cuda.synchronize()

    mask_host = d_mask.copy_to_host()
    return [
        order[batch_idx, _resolve_greedy_mask(mask_host[batch_idx], n)]
        for batch_idx in range(batch_size)
    ]


def _run_gpu_v2_single_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Run V2's SoA packed-mask kernel for a score-sorted class partition."""
    return _run_gpu_v2_batched_single_class(
        boxes[np.newaxis, :, :],
        scores[np.newaxis, :],
        iou_threshold,
    )[0]


def _run_gpu_v2_single_image(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Run class-aware V2 for one validated image."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    kept: list[np.ndarray] = []
    for indices in stable_class_partitions(scores, class_ids):
        local_keep = _run_gpu_v2_single_class(boxes[indices], scores[indices], iou_threshold)
        kept.append(indices[local_keep])
    return stable_score_order(np.concatenate(kept), scores)


def run_gpu_v2_batched(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids=None,
    iou_threshold: float = 0.5,
) -> list[np.ndarray]:
    """Class-aware V2 NMS for a fixed-size image batch.

    A one-class batch preserves the original single CUDA launch.  Multi-class
    detector candidates are partitioned per image/class, then each partition
    uses the same coalesced SoA packed-mask kernel; the serial greedy resolver
    remains on the host by design.
    """
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    if boxes.ndim != 3 or boxes.shape[2] != 4:
        raise ValueError("boxes must have shape (batch_size, n, 4)")
    if scores.ndim != 2 or scores.shape != boxes.shape[:2]:
        raise ValueError("scores must have shape (batch_size, n)")
    if class_ids is None:
        class_ids = np.zeros(scores.shape, dtype=np.int32)
    class_ids = np.asarray(class_ids)
    if class_ids.ndim != 2 or class_ids.shape != scores.shape:
        raise ValueError("class_ids must have shape (batch_size, n)")
    threshold = validate_iou_threshold(iou_threshold)

    batch_size, n, _ = boxes.shape
    if batch_size == 0:
        return []
    if n == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(batch_size)]
    normalized = [
        validate_candidates(boxes[index], scores[index], class_ids[index])
        for index in range(batch_size)
    ]
    normalized_boxes = np.stack([item[0] for item in normalized])
    normalized_scores = np.stack([item[1] for item in normalized])
    normalized_classes = np.stack([item[2] for item in normalized])
    if np.all(normalized_classes == normalized_classes.flat[0]):
        return _run_gpu_v2_batched_single_class(
            normalized_boxes,
            normalized_scores,
            threshold,
        )
    return [
        _run_gpu_v2_single_image(image_boxes, image_scores, image_classes, threshold)
        for image_boxes, image_scores, image_classes in zip(
            normalized_boxes,
            normalized_scores,
            normalized_classes,
        )
    ]


def run_gpu_v2(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids=None,
    iou_threshold: float = 0.5,
) -> np.ndarray | list[np.ndarray]:
    """Class-aware GPU V2 NMS for single-image or batched candidates.

    ``(N, 4)`` / ``(N,)`` input returns one index array as before.  ``(B, N,
    4)`` / ``(B, N)`` input runs one batched GPU mask launch and returns a list
    of index arrays, one per image.
    """
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    if boxes.ndim == 2:
        if class_ids is None:
            class_ids = np.zeros(len(scores), dtype=np.int32)
        normalized_boxes, normalized_scores, normalized_classes = validate_candidates(
            boxes, scores, class_ids
        )
        return _run_gpu_v2_single_image(
            normalized_boxes,
            normalized_scores,
            normalized_classes,
            validate_iou_threshold(iou_threshold),
        )
    if boxes.ndim == 3:
        return run_gpu_v2_batched(boxes, scores, class_ids, iou_threshold)
    raise ValueError("boxes must have shape (n, 4) or (batch_size, n, 4)")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _synthetic_batch(batch_size: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic, independent synthetic inputs for a batch."""
    samples = [load_data(n, seed + image_idx) for image_idx in range(batch_size)]
    return (
        np.stack([boxes for boxes, _ in samples]),
        np.stack([scores for _, scores in samples]),
    )

def benchmark(
    ns: tuple = (100, 1_000, 10_000),
    iou_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    """Time CPU, GPU V1, GPU V2 side by side. Warm-up excludes JIT time."""
    _boxes, _scores = load_data(10, seed=seed)
    _ = run_gpu_v1(_boxes, _scores, iou_threshold=iou_threshold)
    _ = run_gpu_v2(_boxes, _scores, iou_threshold=iou_threshold)

    cols = ["N", "CPU (s)", "GPU V1 (s)", "GPU V2 (s)", "V1 Speedup", "V2 Speedup"]
    header = (
        f"{cols[0]:>8} | {cols[1]:>10} | {cols[2]:>12} | "
        f"{cols[3]:>12} | {cols[4]:>12} | {cols[5]:>12}"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for n in ns:
        boxes, scores = load_data(n, seed=seed)

        t0 = time.perf_counter()
        run_cpu(boxes, scores, iou_threshold=iou_threshold)
        cpu_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_gpu_v1(boxes, scores, iou_threshold=iou_threshold)
        v1_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_gpu_v2(boxes, scores, iou_threshold=iou_threshold)
        v2_t = time.perf_counter() - t0

        v1_sp = cpu_t / v1_t
        v2_sp = cpu_t / v2_t
        results[n] = {
            "cpu": cpu_t, "gpu_v1": v1_t, "gpu_v2": v2_t,
            "v1_speedup": v1_sp, "v2_speedup": v2_sp,
        }
        print(
            f"{n:>8} | {cpu_t:>10.4f} | {v1_t:>12.4f} | "
            f"{v2_t:>12.4f} | {v1_sp:>11.1f}x | {v2_sp:>11.1f}x"
        )

    return results


def benchmark_batched(
    batch_size: int = 32,
    ns: tuple = (100, 1_000, 10_000),
    iou_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    """Compare sequential CPU greedy NMS with one fused V2 batch launch.

    This is the catalog A4 measurement shape: ``batch_size`` independent
    images, each with ``n`` candidate boxes.  V1 is deliberately excluded: it
    has no batched kernel and timing 32 separate V1 launches would not measure
    the V2 batch optimization.
    """
    warmup_boxes, warmup_scores = _synthetic_batch(batch_size, 10, seed)
    _ = run_gpu_v2(warmup_boxes, warmup_scores, iou_threshold=iou_threshold)

    header = (
        f"{'N/image':>10} | {'Batch':>5} | {'CPU greedy (s)':>14} | "
        f"{'GPU V2 batch (s)':>16} | {'Speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    results = {}
    for n in ns:
        boxes, scores = _synthetic_batch(batch_size, n, seed)

        t0 = time.perf_counter()
        for image_idx in range(batch_size):
            run_cpu(boxes[image_idx], scores[image_idx], iou_threshold=iou_threshold)
        cpu_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_gpu_v2(boxes, scores, iou_threshold=iou_threshold)
        gpu_t = time.perf_counter() - t0

        speedup = cpu_t / gpu_t
        results[n] = {
            "batch_size": batch_size,
            "cpu": cpu_t,
            "gpu_v2": gpu_t,
            "speedup": speedup,
        }
        print(
            f"{n:>10} | {batch_size:>5} | {cpu_t:>14.4f} | "
            f"{gpu_t:>16.4f} | {speedup:>7.1f}x"
        )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GPU V2 NMS -- batched GPU suppression masks with coalesced SoA reads"
        )
    )
    parser.add_argument("--n", type=int, default=1_000,
                        help="number of boxes for a single run")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="number of independent N-box images processed in one V2 CUDA launch",
    )
    parser.add_argument("--verify", action="store_true",
                        help="compare kept-set with cpu_baseline and gpu_v1")
    parser.add_argument("--benchmark", action="store_true",
                        help="sweep N in {100, 1000, 10000}")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.benchmark:
        if args.batch_size == 1:
            benchmark(iou_threshold=args.iou_threshold, seed=args.seed)
        else:
            benchmark_batched(
                batch_size=args.batch_size,
                iou_threshold=args.iou_threshold,
                seed=args.seed,
            )
        return

    if args.batch_size == 1:
        boxes, scores = load_data(args.n, seed=args.seed)
        image_count = 1
    else:
        boxes, scores = _synthetic_batch(args.batch_size, args.n, args.seed)
        image_count = args.batch_size
    print(f"Generated {image_count} image(s) × {args.n} synthetic boxes.")
    print("Warming up GPU (JIT compile)...")
    if args.batch_size == 1:
        _ = run_gpu_v2(boxes[:16], scores[:16], iou_threshold=args.iou_threshold)
    else:
        warmup_n = min(16, args.n)
        _ = run_gpu_v2(
            boxes[:, :warmup_n], scores[:, :warmup_n], iou_threshold=args.iou_threshold
        )

    t0 = time.perf_counter()
    keep = run_gpu_v2(boxes, scores, iou_threshold=args.iou_threshold)
    elapsed = time.perf_counter() - t0
    if args.batch_size == 1:
        print(f"GPU V2 NMS: kept {len(keep)}/{len(boxes)} boxes in {elapsed:.4f}s")
    else:
        kept_count = sum(len(item) for item in keep)
        print(
            f"GPU V2 batched NMS: kept {kept_count}/{args.batch_size * args.n} "
            f"boxes across {args.batch_size} images in {elapsed:.4f}s"
        )

    if args.verify:
        if args.batch_size == 1:
            boxes_to_check, scores_to_check, keeps_to_check = [boxes], [scores], [keep]
        else:
            boxes_to_check, scores_to_check, keeps_to_check = boxes, scores, keep
        all_cpu_match = True
        all_v1_match = True
        for image_idx, (image_boxes, image_scores, v2_keep) in enumerate(
            zip(boxes_to_check, scores_to_check, keeps_to_check)
        ):
            cpu_keep = set(
                run_cpu(image_boxes, image_scores, iou_threshold=args.iou_threshold).tolist()
            )
            v1_keep = set(
                run_gpu_v1(image_boxes, image_scores, iou_threshold=args.iou_threshold).tolist()
            )
            v2_keep = set(v2_keep.tolist())
            all_cpu_match &= cpu_keep == v2_keep
            all_v1_match &= v1_keep == v2_keep
            if cpu_keep != v2_keep or v1_keep != v2_keep:
                print(f"Mismatch in image {image_idx}")
        print(f"Exact match with cpu_baseline : {all_cpu_match}")
        print(f"Exact match with gpu_v1       : {all_v1_match}")


if __name__ == "__main__":
    if not _NUMBA_AVAILABLE:
        print("ERROR: numba is not installed.  Run: pip install numba")
        sys.exit(1)
    if not cuda.is_available():
        print("ERROR: No CUDA-capable GPU detected.")
        sys.exit(1)
    main()
