"""Reproducible NMS benchmark runner.

Examples
--------
python benchmarks/run_all.py --versions cpu
python benchmarks/run_all.py --versions cpu v1 v2 v3 --repeats 7 --json benchmarks/results/t4.json

Each invocation measures a single image / one set of boxes.  V2 also has a
separate batch-size benchmark in ``benchmarks/run_v2_batch.py``; V1 and V3
remain single-image implementations.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpu_baseline import load_data, run_cpu  # noqa: E402


def environment() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import numba
        from numba import cuda

        info["numba"] = numba.__version__
        info["cuda_available"] = cuda.is_available()
        if cuda.is_available():
            dev = cuda.get_current_device()
            info["gpu"] = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
            info["compute_capability"] = list(dev.compute_capability)
    except ImportError:
        info["cuda_available"] = False
    return info


def select_runner(version: str):
    if version == "cpu":
        return run_cpu
    if version == "v1":
        from gpu_v1 import run_gpu_v1
        return run_gpu_v1
    if version == "v2":
        from gpu_v2 import run_gpu_v2
        return run_gpu_v2
    if version == "v3":
        from gpu_v3 import run_gpu_v3_matrix_nms

        return lambda boxes, scores: run_gpu_v3_matrix_nms(boxes, scores)
    raise ValueError(version)


def timed_run(runner, boxes: np.ndarray, scores: np.ndarray) -> float:
    start = time.perf_counter()
    runner(boxes, scores)
    return time.perf_counter() - start


def summarize(samples: list[float]) -> dict:
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "mean_seconds": statistics.mean(samples),
        "stddev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated CPU/GPU NMS benchmark")
    parser.add_argument("--n", type=int, nargs="+", default=[100, 1_000, 10_000])
    parser.add_argument("--versions", nargs="+", choices=["cpu", "v1", "v2", "v3"],
                        default=["cpu", "v1", "v2", "v3"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write raw samples and environment metadata")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be >= 1; warmup >= 0")

    info = environment()
    gpu_versions = {"v1", "v2", "v3"}
    unavailable = gpu_versions.intersection(args.versions) if not info["cuda_available"] else set()
    if unavailable:
        print("CUDA unavailable; skipping " + ", ".join(sorted(unavailable)))
    versions = [v for v in args.versions if v not in unavailable]
    if not versions:
        raise SystemExit("No requested runner is available")

    report = {
        "environment": info,
        "configuration": vars(args) | {"json": str(args.json) if args.json else None},
        "input_semantics": "one image / one box set",
        "results": {},
    }
    print(json.dumps(info, indent=2))
    for n in args.n:
        boxes, scores = load_data(n, seed=args.seed)
        report["results"][str(n)] = {}
        for version in versions:
            runner = select_runner(version)
            for _ in range(args.warmup):
                timed_run(runner, boxes, scores)
            samples = [timed_run(runner, boxes, scores) for _ in range(args.repeats)]
            result = summarize(samples)
            report["results"][str(n)][version] = result
            print(
                f"N={n:>6} {version:>3}: "
                f"median={result['median_seconds'] * 1e3:9.3f} ms  "
                f"std={result['stddev_seconds'] * 1e3:7.3f} ms"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
