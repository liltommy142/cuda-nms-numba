"""Convert raw YOLOv5 network output into the canonical NMS contract."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np

from common.candidates import validate_candidates


def raw_yolo_predictions_to_candidates(
    raw_prediction,
    conf_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one pre-NMS YOLO tensor ``(N, 5 + C)`` into NMS candidates."""
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
    selected = prediction[scores >= float(conf_threshold), :4]
    selected_scores = scores[scores >= float(conf_threshold)]
    selected_classes = class_ids[scores >= float(conf_threshold)]
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


def load_raw_yolo_candidates(
    image,
    conf_threshold: float = 0.01,
    image_size: int = 640,
    weights: str | Path | None = None,
    model=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run raw YOLOv5 inference and return pre-NMS candidates for one image."""
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
    boxes, scores, class_ids = raw_yolo_predictions_to_candidates(
        raw_output.detach().cpu().numpy(), conf_threshold=conf_threshold
    )
    if len(boxes):
        boxes = boxes * np.array(
            [original_width / image_size, original_height / image_size] * 2,
            dtype=np.float32,
        )
    return validate_candidates(boxes, scores, class_ids)
