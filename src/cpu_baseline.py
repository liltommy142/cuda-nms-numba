"""CPU baseline for Non-Maximum Suppression (NMS) -- CSC14116, topic A4.

Greedy NMS implemented in pure NumPy. This is the serial O(n^2) reference
that the GPU kernels (V1/V2/V3) in this project are benchmarked against, and
the thing cProfile should point at as the bottleneck.

Usage:
    python cpu_baseline.py --source synthetic --n 10000 --verify
    python cpu_baseline.py --source synthetic --benchmark --verify
    python cpu_baseline.py --source yolo-live --image path/to/image.jpg --verify
"""

import argparse
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from nms_common import (
    load_synthetic_candidates,
    stable_class_partitions,
    stable_score_order,
    torchvision_class_aware_nms,
    validate_candidates,
    validate_iou_threshold,
)


def load_data(n, seed=0):
    """Legacy two-array synthetic source retained for the untouched V3 path."""
    boxes, scores, _ = load_synthetic_candidates(n, seed=seed)
    return boxes, scores


def raw_yolo_predictions_to_candidates(
    raw_prediction,
    conf_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one YOLO raw prediction tensor to canonical NMS candidates.

    ``raw_prediction`` is the pre-NMS model output ``(N, 5 + C)`` containing
    ``cx, cy, width, height, objectness, class_probabilities...``.  Scores are
    objectness multiplied by the best class probability, and class ids are
    retained so hard NMS runs independently for every class.
    """
    prediction = np.asarray(raw_prediction, dtype=np.float32)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError("raw prediction batch must contain exactly one image")
        prediction = prediction[0]
    if prediction.ndim != 2 or prediction.shape[1] < 6:
        raise ValueError("raw prediction must have shape (N, 5 + num_classes)")
    if not np.isfinite(prediction).all():
        raise ValueError("raw prediction must be finite")
    if not 0.0 <= float(conf_threshold) <= 1.0:
        raise ValueError("conf_threshold must be in [0, 1]")

    class_ids = np.argmax(prediction[:, 5:], axis=1).astype(np.int32)
    scores = prediction[:, 4] * prediction[np.arange(len(prediction)), 5 + class_ids]
    keep = scores >= float(conf_threshold)
    selected = prediction[keep, :4]
    selected_scores = scores[keep]
    selected_classes = class_ids[keep]
    if len(selected) == 0:
        return validate_candidates([], [], [])

    cx, cy, width, height = selected.T
    boxes = np.column_stack((
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    ))
    return validate_candidates(boxes, selected_scores, selected_classes)


def _open_rgb_image(image):
    """Load a local path, URL, or PIL image as RGB without detector postprocessing."""
    from PIL import Image

    if hasattr(image, "convert"):
        return image.convert("RGB")
    image_text = str(image)
    if image_text.startswith(("http://", "https://")):
        with urlopen(image_text) as response:
            return Image.open(response).convert("RGB")
    return Image.open(image_text).convert("RGB")


def load_raw_yolo_candidates(
    image,
    conf_threshold: float = 0.01,
    image_size: int = 640,
    weights: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run local YOLOv5 weights and return pre-NMS candidates for one image.

    This deliberately calls the underlying network on a normalized tensor, not
    Ultralytics' high-level result API (which includes its own NMS).
    """
    import torch
    from ultralytics import YOLO

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    image_rgb = _open_rgb_image(image)
    original_width, original_height = image_rgb.size
    resized = image_rgb.resize((image_size, image_size))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    default_weights = Path(__file__).resolve().parents[1] / "yolov5s.pt"
    model = YOLO(str(default_weights if weights is None else weights))
    model.model.eval()
    with torch.inference_mode():
        raw_output = model.model(tensor)
    if isinstance(raw_output, (tuple, list)):
        raw_output = raw_output[0]
    boxes, scores, class_ids = raw_yolo_predictions_to_candidates(
        raw_output.detach().cpu().numpy(),
        conf_threshold=conf_threshold,
    )
    if len(boxes):
        scale = np.array(
            [original_width / image_size, original_height / image_size] * 2,
            dtype=np.float32,
        )
        boxes = boxes * scale
    return validate_candidates(boxes, scores, class_ids)


def load_real_boxes(image_paths=None, conf_threshold=0.01):
    """Compatibility wrapper returning raw YOLO boxes/scores for one image."""
    if image_paths is None:
        image_paths = ["https://ultralytics.com/images/zidane.jpg"]
    if len(image_paths) != 1:
        raise ValueError("use load_raw_yolo_candidates once per image")
    boxes, scores, _ = load_raw_yolo_candidates(image_paths[0], conf_threshold)
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


def _run_cpu_single_class(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS for boxes already sorted from highest to lowest score."""
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep_ranks: list[int] = []
    for rank in range(len(boxes)):
        if suppressed[rank]:
            continue
        keep_ranks.append(rank)
        remaining = np.flatnonzero(~suppressed[rank + 1 :]) + rank + 1
        if len(remaining):
            ious = iou_one_to_many(boxes[rank], boxes[remaining])
            suppressed[remaining[ious > iou_threshold]] = True
    return np.asarray(keep_ranks, dtype=np.int64)


def run_cpu(boxes, scores, class_ids=None, iou_threshold=0.5):
    """Class-aware serial greedy NMS.

    ``class_ids=None`` preserves the old one-class behavior used by the
    untouched V3 implementation.  Returned indexes always refer to the
    original input and are sorted by descending score (then input index).
    """
    if class_ids is None:
        class_ids = np.zeros(len(scores), dtype=np.int32)
    boxes, scores, class_ids = validate_candidates(boxes, scores, class_ids)
    threshold = validate_iou_threshold(iou_threshold)
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    kept: list[np.ndarray] = []
    for indices in stable_class_partitions(scores, class_ids):
        local_keep = _run_cpu_single_class(boxes[indices], threshold)
        kept.append(indices[local_keep])
    return stable_score_order(np.concatenate(kept), scores)


def verify(boxes, scores, iou_threshold, keep, class_ids=None):
    """Compare class-aware CPU NMS against the torchvision per-class oracle."""
    try:
        ref_keep = torchvision_class_aware_nms(
            boxes,
            scores,
            np.zeros(len(scores), dtype=np.int32) if class_ids is None else class_ids,
            iou_threshold,
        )
    except ImportError:
        print("torchvision not installed -- skipping verification against reference NMS")
        return None

    matches = np.array_equal(keep, ref_keep)
    print(f"Reference (torchvision) kept {len(ref_keep)} boxes, ours kept {len(keep)}")
    print(f"Exact ordered match: {matches}")
    if not matches:
        print(f"  ours:   {keep.tolist()}")
        print(f"  oracle: {ref_keep.tolist()}")
    return matches


def benchmark(ns=(100, 1000, 10000), iou_threshold=0.5, seed=0, verify_result=False):
    """Time class-aware CPU NMS on deterministic synthetic candidates only."""
    print("benchmark_scope: nms_only_synthetic")
    print("candidate_source: deterministic_synthetic")
    print("timing_scope: candidate NMS only; excludes model inference")
    print(f"{'N':>8} | {'time (s)':>10}")
    print("-" * 21)
    results = {}
    for n in ns:
        boxes, scores, class_ids = load_synthetic_candidates(n, seed=seed)
        start = time.perf_counter()
        keep = run_cpu(boxes, scores, class_ids, iou_threshold)
        elapsed = time.perf_counter() - start
        results[n] = elapsed
        print(f"{n:>8} | {elapsed:>10.4f}")
        if verify_result and not verify(boxes, scores, iou_threshold, keep, class_ids):
            raise RuntimeError(f"torchvision verification failed for N={n}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CPU baseline for NMS (topic A4)")
    parser.add_argument("--n", type=int, default=1000, help="number of boxes for a single run")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=("synthetic", "yolo-live"),
        default="synthetic",
        help="candidate source; yolo-live supplies raw pre-NMS model outputs",
    )
    parser.add_argument("--image", help="local path or URL for --source yolo-live")
    parser.add_argument("--real-boxes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--conf-threshold", type=float, default=0.01, help="minimum objectness × class probability for yolo-live")
    parser.add_argument("--verify", action="store_true", help="compare against torchvision.ops.nms")
    parser.add_argument("--benchmark", action="store_true", help="sweep N in {100, 1000, 10000}")
    args = parser.parse_args()

    if args.benchmark:
        if args.source != "synthetic" or args.real_boxes:
            parser.error("--benchmark is only defined for --source synthetic")
        benchmark(
            iou_threshold=args.iou_threshold,
            seed=args.seed,
            verify_result=args.verify,
        )
        return

    if args.real_boxes:
        args.source = "yolo-live"
    if args.source == "yolo-live":
        if not args.image:
            parser.error("--source yolo-live requires --image")
        boxes, scores, class_ids = load_raw_yolo_candidates(
            args.image,
            conf_threshold=args.conf_threshold,
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


if __name__ == "__main__":
    main()
