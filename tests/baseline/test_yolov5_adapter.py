"""Compatibility tests for the raw YOLOv5 candidate adapter."""


def test_cpu_facade_reexports_package_yolo_adapter():
    import cpu_baseline
    from baseline.yolov5_adapter import raw_yolo_predictions_to_candidates

    assert (
        cpu_baseline.raw_yolo_predictions_to_candidates
        is raw_yolo_predictions_to_candidates
    )
