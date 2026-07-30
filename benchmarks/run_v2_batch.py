"""Repeated end-to-end benchmark for GPU V2 batched greedy NMS.

Example:
    python benchmarks/run_v2_batch.py --batch-size 32 --n 10000 \
        --warmup 2 --repeats 7 \
        --json presentation/seminar_2/evidence/batch32_t4.json
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

from cpu_baseline import load_data  # noqa: E402
from gpu_v2 import run_gpu_v2  # noqa: E402


def environment() -> dict:
    """Return the metadata required to interpret a GPU timing result."""
    from numba import cuda
    import numba

    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable an NVIDIA GPU runtime first")

    device = cuda.get_current_device()
    name = device.name.decode() if isinstance(device.name, bytes) else device.name
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "gpu": name,
        "compute_capability": list(device.compute_capability),
    }


def make_batch(batch_size: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic but different synthetic boxes for every image."""
    samples = [load_data(n, seed + image_idx) for image_idx in range(batch_size)]
    return (
        np.stack([boxes for boxes, _ in samples]),
        np.stack([scores for _, scores in samples]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated V2 batched-NMS benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n", type=int, default=10_000, help="boxes per image")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.n < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("batch-size/n/repeats must be positive; warmup must be non-negative")

    info = environment()
    boxes, scores = make_batch(args.batch_size, args.n, args.seed)

    # The small launch compiles the kernel.  The following full-size calls also
    # warm allocation/caching effects before any sample is recorded.
    run_gpu_v2(boxes[:, : min(64, args.n)], scores[:, : min(64, args.n)])
    for _ in range(args.warmup):
        run_gpu_v2(boxes, scores)

    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run_gpu_v2(boxes, scores)
        samples.append(time.perf_counter() - start)

    median = statistics.median(samples)
    report = {
        "environment": info,
        "configuration": vars(args) | {"json": str(args.json)},
        "input_semantics": "batch_size independent images, n boxes per image",
        "timing_semantics": (
            "end-to-end V2 call: host sort, transfer, GPU mask kernel, "
            "mask download, and host greedy mask resolution"
        ),
        "samples_seconds": samples,
        "median_batch_seconds": median,
        "mean_batch_seconds": statistics.mean(samples),
        "stddev_batch_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median_per_image_seconds": median / args.batch_size,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
