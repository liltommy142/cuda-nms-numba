"""CPU baseline for Non-Maximum Suppression (NMS) -- CSC14116, topic A4.

Greedy NMS implemented in pure NumPy. This is the serial O(n^2) reference
that the GPU kernels (V1/V2/V3) in this project are benchmarked against, and
the thing cProfile should point at as the bottleneck.

Usage:
    python cpu_baseline.py                       # single run, N=1000 synthetic boxes
    python cpu_baseline.py --n 10000 --verify     # run + check against torchvision.ops.nms
    python cpu_baseline.py --benchmark            # sweep N in {100, 1000, 10000}
    python cpu_baseline.py --real-boxes --verify  # use YOLOv5s detections instead of synthetic boxes
"""

import argparse
import time

import numpy as np


def load_data(n, seed=0):
    """Generate synthetic (boxes, scores) for benchmarking.

    boxes: (n, 4) array of [x1, y1, x2, y2]
    scores: (n,) array of confidence scores in [0, 1)
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 900, size=n)
    y1 = rng.uniform(0, 900, size=n)
    w = rng.uniform(10, 100, size=n)
    h = rng.uniform(10, 100, size=n)
    boxes = np.stack([x1, y1, x1 + w, y1 + h], axis=1).astype(np.float32)
    scores = rng.uniform(0, 1, size=n).astype(np.float32)
    return boxes, scores


def load_real_boxes(image_paths=None, conf_threshold=0.25):
    """Run pretrained YOLOv5s on a handful of images to get real (boxes, scores)."""
    import torch

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, trust_repo=True)
    model.conf = conf_threshold
    # AutoShape runs its own NMS internally before returning results; disable it
    # (iou=1.0 keeps virtually all overlapping candidates) so load_real_boxes returns
    # raw pre-NMS boxes for this project's own run_cpu/run_gpu_v1 to suppress.
    model.iou = 1.0
    if not image_paths:
        image_paths = ["https://ultralytics.com/images/zidane.jpg"]

    results = model(image_paths)
    boxes_list = [pred[:, :4].cpu().numpy() for pred in results.xyxy]
    scores_list = [pred[:, 4].cpu().numpy() for pred in results.xyxy]
    boxes = np.concatenate(boxes_list, axis=0).astype(np.float32)
    scores = np.concatenate(scores_list, axis=0).astype(np.float32)
    return boxes, scores


def iou_one_to_many(box, boxes):
    """IoU between a single box (4,) and an array of boxes (M, 4)."""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1e-9)


def run_cpu(boxes, scores, iou_threshold=0.5):
    """Greedy NMS: sort by score, then sequentially keep/suppress.

    Kept intentionally serial (not vectorized across the suppression loop)
    since the sequential dependency here is exactly what GPU V3 (Matrix NMS)
    is meant to remove -- see The Challenge section in PROPOSAL_DRAFT.md.
    """
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep.append(idx)

        remaining = order[i + 1:]
        remaining = remaining[~suppressed[remaining]]
        if len(remaining) == 0:
            continue

        ious = iou_one_to_many(boxes[idx], boxes[remaining])
        suppressed[remaining[ious > iou_threshold]] = True

    return np.array(keep, dtype=np.int64)


def verify(boxes, scores, iou_threshold, keep):
    """Compare our NMS output against torchvision.ops.nms (ground truth)."""
    try:
        import torch
        from torchvision.ops import nms as torch_nms
    except ImportError:
        print("torchvision not installed -- skipping verification against reference NMS")
        return None

    ref_keep = torch_nms(
        torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold
    ).numpy()

    ours, theirs = set(keep.tolist()), set(ref_keep.tolist())
    matches = ours == theirs
    print(f"Reference (torchvision) kept {len(theirs)} boxes, ours kept {len(ours)}")
    print(f"Exact match: {matches}")
    if not matches:
        print(f"  only ours:   {sorted(ours - theirs)}")
        print(f"  only theirs: {sorted(theirs - ours)}")
    return matches


def benchmark(ns=(100, 1000, 10000), iou_threshold=0.5, seed=0):
    """Time run_cpu across increasing N to find where it becomes a bottleneck."""
    print(f"{'N':>8} | {'time (s)':>10}")
    print("-" * 21)
    results = {}
    for n in ns:
        boxes, scores = load_data(n, seed=seed)
        start = time.perf_counter()
        run_cpu(boxes, scores, iou_threshold)
        elapsed = time.perf_counter() - start
        results[n] = elapsed
        print(f"{n:>8} | {elapsed:>10.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CPU baseline for NMS (topic A4)")
    parser.add_argument("--n", type=int, default=1000, help="number of boxes for a single run")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-boxes", action="store_true", help="use YOLOv5s detections instead of synthetic boxes")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="YOLOv5s confidence threshold (--real-boxes only); lower = more raw boxes")
    parser.add_argument("--verify", action="store_true", help="compare against torchvision.ops.nms")
    parser.add_argument("--benchmark", action="store_true", help="sweep N in {100, 1000, 10000}")
    args = parser.parse_args()

    if args.benchmark:
        benchmark(iou_threshold=args.iou_threshold, seed=args.seed)
        return

    if args.real_boxes:
        boxes, scores = load_real_boxes(conf_threshold=args.conf_threshold)
        print(f"Loaded {len(boxes)} real boxes from YOLOv5s")
    else:
        boxes, scores = load_data(args.n, seed=args.seed)
        print(f"Generated {len(boxes)} synthetic boxes")

    start = time.perf_counter()
    keep = run_cpu(boxes, scores, args.iou_threshold)
    elapsed = time.perf_counter() - start
    print(f"NMS kept {len(keep)}/{len(boxes)} boxes in {elapsed:.4f}s")

    if args.verify:
        verify(boxes, scores, args.iou_threshold, keep)


if __name__ == "__main__":
    main()
