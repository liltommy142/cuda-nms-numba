"""Command-line interface and synthetic benchmark for GPU V1."""

from __future__ import annotations

import argparse
import time

import numpy as np

from baseline.core import run_cpu
from common.candidates import load_synthetic_candidates
from v1.core import run_gpu_v1
from v1.kernel import NUMBA_AVAILABLE, cuda


def benchmark(ns=(100, 1_000, 10_000), iou_threshold: float = 0.5, seed: int = 0) -> dict:
    """Time class-aware CPU and V1 NMS; the warm-up excludes JIT time."""
    boxes, scores, class_ids = load_synthetic_candidates(10, seed=seed)
    run_gpu_v1(boxes, scores, class_ids, iou_threshold)
    results = {}
    for n in ns:
        boxes, scores, class_ids = load_synthetic_candidates(n, seed=seed)
        start = time.perf_counter(); run_cpu(boxes, scores, class_ids, iou_threshold)
        cpu_seconds = time.perf_counter() - start
        start = time.perf_counter(); run_gpu_v1(boxes, scores, class_ids, iou_threshold)
        gpu_seconds = time.perf_counter() - start
        results[n] = {"cpu": cpu_seconds, "gpu_v1": gpu_seconds, "speedup": cpu_seconds / gpu_seconds}
        print(f"{n:>8} | {cpu_seconds:>10.4f} | {gpu_seconds:>12.4f} | {cpu_seconds / gpu_seconds:>7.1f}x")
    return results


def main() -> None:
    """Run the historic ``python src/gpu_v1.py`` command."""
    parser = argparse.ArgumentParser(description="GPU V1 NMS — naive parallel IoU kernel (topic A4)")
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if not NUMBA_AVAILABLE:
        parser.error("numba is not installed")
    if not cuda.is_available():
        parser.error("No CUDA-capable GPU detected")
    if args.benchmark:
        benchmark(iou_threshold=args.iou_threshold, seed=args.seed)
        return
    boxes, scores, class_ids = load_synthetic_candidates(args.n, seed=args.seed)
    run_gpu_v1(boxes[:16], scores[:16], class_ids[:16], args.iou_threshold)
    start = time.perf_counter()
    keep = run_gpu_v1(boxes, scores, class_ids, args.iou_threshold)
    print(f"GPU V1 NMS: kept {len(keep)}/{len(boxes)} boxes in {time.perf_counter() - start:.4f}s")
    if args.verify:
        print(f"Exact match with cpu_baseline: {np.array_equal(keep, run_cpu(boxes, scores, class_ids, args.iou_threshold))}")
