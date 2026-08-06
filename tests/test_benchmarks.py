"""Metadata contracts for reproducible Seminar 2 benchmark reports."""

import os
import sys


_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

from run_all import build_synthetic_report  # noqa: E402
from run_detector_pipeline import build_detector_report  # noqa: E402
from run_v2_batch import make_batch  # noqa: E402


def test_synthetic_benchmark_report_has_explicit_scope():
    report = build_synthetic_report(n=100, repeats=1, warmup=0, seed=0, versions=["cpu"])
    assert report["benchmark_scope"] == "nms_only_synthetic"
    assert report["candidate_source"] == "deterministic_synthetic"
    assert report["timing_scope"] == "candidate NMS only; excludes model inference"
    assert set(report["results"]) == {"100"}


def test_detector_report_separates_raw_candidates_from_nms_time():
    import numpy as np

    def fake_raw_loader(image, conf_threshold=0.01):
        return (
            np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32),
            np.array([0.9, 0.8], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
        )

    report = build_detector_report(
        "fixture-image",
        loader=fake_raw_loader,
        repeats=1,
        warmup=0,
    )
    assert report["benchmark_scope"] == "detector_plus_nms_real"
    assert report["candidate_source"] == "yolo_raw_pre_nms"
    assert report["candidate_count"] == 2
    assert report["torchvision_parity"] is True
    assert report["raw_candidate_seconds"][0] >= 0
    assert report["nms_seconds"][0] >= 0


def test_v2_batch_input_contains_class_ids():
    boxes, scores, class_ids = make_batch(batch_size=3, n=10, seed=4)
    assert boxes.shape == (3, 10, 4)
    assert scores.shape == (3, 10)
    assert class_ids.shape == (3, 10)
