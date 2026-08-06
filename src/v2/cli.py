"""CLI and synthetic benchmark helpers for GPU V2."""

from __future__ import annotations

import argparse
import time

import numpy as np

from baseline.core import run_cpu
from common.candidates import load_synthetic_candidates
from v1.core import run_gpu_v1
from v2.core import run_gpu_v2
from v2.kernels import NUMBA_AVAILABLE, cuda


def _synthetic_batch(batch_size: int, n: int, seed: int):
    samples = [load_synthetic_candidates(n, seed + index) for index in range(batch_size)]
    return tuple(np.stack([sample[axis] for sample in samples]) for axis in range(3))


def benchmark(ns=(100, 1_000, 10_000), iou_threshold: float = 0.5, seed: int = 0) -> dict:
    """Compare class-aware CPU, V1, and V2 NMS for one image."""
    warmup = load_synthetic_candidates(10, seed)
    run_gpu_v1(*warmup, iou_threshold); run_gpu_v2(*warmup, iou_threshold)
    results = {}
    for n in ns:
        candidates = load_synthetic_candidates(n, seed)
        start = time.perf_counter(); run_cpu(*candidates, iou_threshold); cpu = time.perf_counter() - start
        start = time.perf_counter(); run_gpu_v1(*candidates, iou_threshold); v1 = time.perf_counter() - start
        start = time.perf_counter(); run_gpu_v2(*candidates, iou_threshold); v2 = time.perf_counter() - start
        results[n] = {"cpu": cpu, "gpu_v1": v1, "gpu_v2": v2, "v1_speedup": cpu / v1, "v2_speedup": cpu / v2}
        print(f"{n:>8} | {cpu:>10.4f} | {v1:>12.4f} | {v2:>12.4f}")
    return results


def benchmark_batched(batch_size: int = 32, ns=(100, 1_000, 10_000), iou_threshold: float = 0.5, seed: int = 0) -> dict:
    """Compare sequential CPU NMS with one V2 batch call."""
    boxes, scores, class_ids = _synthetic_batch(batch_size, 10, seed)
    run_gpu_v2(boxes, scores, class_ids, iou_threshold)
    results = {}
    for n in ns:
        boxes, scores, class_ids = _synthetic_batch(batch_size, n, seed)
        start = time.perf_counter()
        for index in range(batch_size):
            run_cpu(boxes[index], scores[index], class_ids[index], iou_threshold)
        cpu = time.perf_counter() - start
        start = time.perf_counter(); run_gpu_v2(boxes, scores, class_ids, iou_threshold); gpu = time.perf_counter() - start
        results[n] = {"batch_size": batch_size, "cpu": cpu, "gpu_v2": gpu, "speedup": cpu / gpu}
        print(f"{n:>10} | {batch_size:>5} | {cpu:>14.4f} | {gpu:>16.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU V2 NMS -- batched GPU suppression masks with coalesced SoA reads")
    parser.add_argument("--n", type=int, default=1_000); parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verify", action="store_true"); parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not NUMBA_AVAILABLE:
        parser.error("numba is not installed")
    if not cuda.is_available():
        parser.error("No CUDA-capable GPU detected")
    if args.benchmark:
        (benchmark if args.batch_size == 1 else benchmark_batched)(
            **({"iou_threshold": args.iou_threshold, "seed": args.seed} if args.batch_size == 1 else {"batch_size": args.batch_size, "iou_threshold": args.iou_threshold, "seed": args.seed})
        )
        return
    candidates = load_synthetic_candidates(args.n, args.seed) if args.batch_size == 1 else _synthetic_batch(args.batch_size, args.n, args.seed)
    run_gpu_v2(*(item[:16] if args.batch_size == 1 else item[:, :16] for item in candidates), args.iou_threshold)
    start = time.perf_counter(); keep = run_gpu_v2(*candidates, args.iou_threshold)
    print(f"GPU V2 NMS completed in {time.perf_counter() - start:.4f}s")
    if args.verify:
        expected = run_cpu(*candidates, args.iou_threshold) if args.batch_size == 1 else [run_cpu(*items, args.iou_threshold) for items in zip(*candidates)]
        print(f"Exact match with cpu_baseline: {np.array_equal(keep, expected) if args.batch_size == 1 else all(np.array_equal(a, b) for a, b in zip(keep, expected))}")
