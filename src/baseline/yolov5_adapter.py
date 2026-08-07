"""Convert raw YOLOv5 network output into the canonical NMS contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from common.candidates import validate_candidates


@dataclass(frozen=True)
class RawCandidateSelection:
    """Canonical raw-YOLO candidates plus auditable selection metadata."""

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    raw_proposal_count: int
    selected_count: int
    effective_conf_threshold: float | None
    max_candidates: int | None


def _validate_max_candidates(max_candidates: int | None) -> int | None:
    """Validate the optional adaptive raw-proposal budget."""
    if max_candidates is None:
        return None
    if isinstance(max_candidates, bool) or not isinstance(
        max_candidates, (int, np.integer)
    ):
        raise ValueError("max_candidates must be a positive integer or None")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer or None")
    return int(max_candidates)


def raw_yolo_predictions_to_selection(
    raw_prediction,
    conf_threshold: float = 0.01,
    max_candidates: int | None = None,
) -> RawCandidateSelection:
    """Select canonical candidates from one pre-NMS YOLO tensor ``(N, 5 + C)``.

    Without a budget, selection retains every proposal satisfying
    ``score >= conf_threshold``. With a budget, it takes the globally highest
    scores, resolving equal-score boundaries by original proposal index.
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
    budget = _validate_max_candidates(max_candidates)
    if budget is None:
        selected_indices = np.flatnonzero(scores >= float(conf_threshold))
        effective_conf_threshold = None
    else:
        raw_indices = np.arange(len(scores))
        selected_indices = np.lexsort((raw_indices, -scores))[:budget]
        effective_conf_threshold = (
            float(scores[selected_indices[-1]]) if len(selected_indices) else None
        )

    selected = prediction[selected_indices, :4]
    selected_scores = scores[selected_indices]
    selected_classes = class_ids[selected_indices]

    cx, cy, width, height = selected.T
    boxes = np.column_stack((
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    ))
    boxes, selected_scores, selected_classes = validate_candidates(
        boxes, selected_scores, selected_classes
    )
    return RawCandidateSelection(
        boxes=boxes,
        scores=selected_scores,
        class_ids=selected_classes,
        raw_proposal_count=len(prediction),
        selected_count=len(boxes),
        effective_conf_threshold=effective_conf_threshold,
        max_candidates=budget,
    )


def raw_yolo_predictions_to_candidates(
    raw_prediction,
    conf_threshold: float = 0.01,
    max_candidates: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw-YOLO candidates while preserving the legacy tuple contract."""
    selection = raw_yolo_predictions_to_selection(
        raw_prediction,
        conf_threshold=conf_threshold,
        max_candidates=max_candidates,
    )
    return selection.boxes, selection.scores, selection.class_ids


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


def load_raw_yolov5_model(weights: str | Path | None = None):
    """Load matching YOLOv5 hub code and weights with postprocessing disabled."""
    import torch

    default_weights = Path(__file__).resolve().parents[2] / "yolov5s.pt"
    checkpoint = Path(default_weights if weights is None else weights).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLOv5 weights not found: {checkpoint}")
    cached_repo = Path(torch.hub.get_dir()) / "ultralytics_yolov5_master"
    if cached_repo.is_dir():
        return torch.hub.load(
            str(cached_repo), "custom", path=str(checkpoint), source="local",
            autoshape=False, verbose=False,
        )
    return torch.hub.load(
        "ultralytics/yolov5", "custom", path=str(checkpoint), autoshape=False,
        trust_repo=True, verbose=False,
    )


def load_raw_yolo_candidate_selection(
    image,
    conf_threshold: float = 0.01,
    image_size: int = 640,
    weights: str | Path | None = None,
    model=None,
    max_candidates: int | None = None,
) -> RawCandidateSelection:
    """Run raw YOLOv5 inference and return selected pre-NMS candidates."""
    import torch

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    image_rgb = _open_rgb_image(image)
    original_width, original_height = image_rgb.size
    resized = image_rgb.resize((image_size, image_size))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    if model is None:
        model = load_raw_yolov5_model(weights)
    model.eval()
    with torch.inference_mode():
        raw_output = model(tensor)
    if isinstance(raw_output, (tuple, list)):
        raw_output = raw_output[0]
    selection = raw_yolo_predictions_to_selection(
        raw_output.detach().cpu().numpy(),
        conf_threshold=conf_threshold,
        max_candidates=max_candidates,
    )
    boxes = selection.boxes
    if len(boxes):
        boxes = boxes * np.array(
            [original_width / image_size, original_height / image_size] * 2,
            dtype=np.float32,
        )
    boxes, scores, class_ids = validate_candidates(
        boxes, selection.scores, selection.class_ids
    )
    return RawCandidateSelection(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        raw_proposal_count=selection.raw_proposal_count,
        selected_count=selection.selected_count,
        effective_conf_threshold=selection.effective_conf_threshold,
        max_candidates=selection.max_candidates,
    )


def load_raw_yolo_candidates(
    image,
    conf_threshold: float = 0.01,
    image_size: int = 640,
    weights: str | Path | None = None,
    model=None,
    max_candidates: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw-YOLO candidates while preserving the legacy tuple contract."""
    selection = load_raw_yolo_candidate_selection(
        image,
        conf_threshold=conf_threshold,
        image_size=image_size,
        weights=weights,
        model=model,
        max_candidates=max_candidates,
    )
    return selection.boxes, selection.scores, selection.class_ids
