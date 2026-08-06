"""Command-line interface for the CPU hard-NMS baseline."""

from __future__ import annotations

import argparse
import time

from baseline.core import benchmark, run_cpu, verify
from baseline.yolov5_adapter import load_raw_yolo_candidates
from common.candidates import load_synthetic_candidates


def main() -> None:
    """Run the documented CPU baseline command-line interface."""
    parser = argparse.ArgumentParser(description="CPU baseline for NMS (topic A4)")
    parser.add_argument("--n", type=int, default=1_000, help="number of boxes for a single run")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source", choices=("synthetic", "yolo-live"), default="synthetic",
        help="candidate source; yolo-live supplies raw pre-NMS model outputs",
    )
    parser.add_argument("--image", help="local path or URL for --source yolo-live")
    parser.add_argument("--conf-threshold", type=float, default=0.01)
    parser.add_argument("--verify", action="store_true", help="compare against torchvision.ops.nms")
    parser.add_argument("--benchmark", action="store_true", help="sweep N in {100, 1000, 10000}")
    args = parser.parse_args()

    if args.benchmark:
        if args.source != "synthetic":
            parser.error("--benchmark is only defined for --source synthetic")
        benchmark(iou_threshold=args.iou_threshold, seed=args.seed, verify_result=args.verify)
        return

    if args.source == "yolo-live":
        if not args.image:
            parser.error("--source yolo-live requires --image")
        boxes, scores, class_ids = load_raw_yolo_candidates(
            args.image, conf_threshold=args.conf_threshold
        )
        print(f"Loaded {len(boxes)} raw pre-NMS candidates from YOLOv5s")
    else:
        boxes, scores, class_ids = load_synthetic_candidates(args.n, seed=args.seed)
        print(f"Generated {len(boxes)} deterministic synthetic candidates")

    start = time.perf_counter()
    keep = run_cpu(boxes, scores, class_ids, args.iou_threshold)
    elapsed = time.perf_counter() - start
    print(f"NMS kept {len(keep)}/{len(boxes)} boxes in {elapsed:.4f}s")
    if args.verify:
        verify(boxes, scores, args.iou_threshold, keep, class_ids)
