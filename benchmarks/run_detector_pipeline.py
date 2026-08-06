"""Measure raw detector candidate extraction and class-aware NMS separately.

The catalog requires an end-to-end detector demonstration, but NMS-only stress
benchmarks must not be misrepresented as detector latency.  This runner records
the two phases independently and validates hard-NMS against torchvision per
class.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpu_baseline import (  # noqa: E402
    load_raw_yolo_candidates,
    load_raw_yolov5_model,
    run_cpu,
)
from nms_common import torchvision_class_aware_nms  # noqa: E402
from run_all import environment  # noqa: E402


def select_hard_nms_runner(name: str):
    if name == "cpu":
        return run_cpu
    if name == "v1":
        from gpu_v1 import run_gpu_v1

        return run_gpu_v1
    if name == "v2":
        from gpu_v2 import run_gpu_v2

        return run_gpu_v2
    raise ValueError(f"Unsupported hard-NMS runner: {name}")


def build_detector_report(
    image,
    *,
    runner=run_cpu,
    loader=load_raw_yolo_candidates,
    conf_threshold: float = 0.01,
    repeats: int = 3,
    warmup: int = 1,
) -> dict:
    """Build a real detector-plus-NMS report using raw pre-NMS candidates."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup must be >= 0")

    for _ in range(warmup):
        boxes, scores, class_ids = loader(image, conf_threshold=conf_threshold)
        runner(boxes, scores, class_ids)

    raw_candidate_seconds: list[float] = []
    nms_seconds: list[float] = []
    last_candidates = None
    last_keep = None
    for _ in range(repeats):
        start = time.perf_counter()
        boxes, scores, class_ids = loader(image, conf_threshold=conf_threshold)
        raw_candidate_seconds.append(time.perf_counter() - start)

        start = time.perf_counter()
        keep = runner(boxes, scores, class_ids)
        nms_seconds.append(time.perf_counter() - start)
        last_candidates = (boxes, scores, class_ids)
        last_keep = keep

    boxes, scores, class_ids = last_candidates
    oracle = torchvision_class_aware_nms(boxes, scores, class_ids, 0.5)
    return {
        "benchmark_scope": "detector_plus_nms_real",
        "candidate_source": "yolo_raw_pre_nms",
        "timing_scope": (
            "raw detector candidate extraction and class-aware NMS are measured "
            "separately; model initialization is excluded when the loader reuses a model"
        ),
        "environment": environment(),
        "configuration": {
            "conf_threshold": conf_threshold,
            "repeats": repeats,
            "warmup": warmup,
        },
        "candidate_count": len(boxes),
        "class_count": len(set(class_ids.tolist())),
        "kept_count": len(last_keep),
        "raw_candidate_seconds": raw_candidate_seconds,
        "nms_seconds": nms_seconds,
        "median_raw_candidate_seconds": statistics.median(raw_candidate_seconds),
        "median_nms_seconds": statistics.median(nms_seconds),
        "torchvision_parity": bool((last_keep == oracle).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw YOLO candidate + NMS benchmark")
    parser.add_argument("--image", required=True, help="local image path or URL")
    parser.add_argument("--runner", choices=("cpu", "v1", "v2"), default="cpu")
    parser.add_argument("--conf-threshold", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    model = load_raw_yolov5_model()

    def loader(image, conf_threshold=0.01):
        return load_raw_yolo_candidates(
            image,
            conf_threshold=conf_threshold,
            model=model,
        )

    report = build_detector_report(
        args.image,
        runner=select_hard_nms_runner(args.runner),
        loader=loader,
        conf_threshold=args.conf_threshold,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    report["configuration"]["runner"] = args.runner
    report["configuration"]["json"] = str(args.json)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
