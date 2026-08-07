"""Compatibility tests for the raw YOLOv5 candidate adapter."""

import numpy as np
import pytest


def _raw_predictions(scores: np.ndarray) -> np.ndarray:
    raw = np.zeros((len(scores), 6), dtype=np.float32)
    raw[:, 0] = np.arange(len(scores), dtype=np.float32)
    raw[:, 1] = 10.0
    raw[:, 2:4] = 1.0
    raw[:, 4] = scores
    raw[:, 5] = 1.0
    return raw


def test_cpu_facade_reexports_package_yolo_adapter():
    import cpu_baseline
    from baseline.yolov5_adapter import raw_yolo_predictions_to_candidates

    assert (
        cpu_baseline.raw_yolo_predictions_to_candidates
        is raw_yolo_predictions_to_candidates
    )


def test_adaptive_budget_keeps_exactly_top_k_raw_proposals():
    from baseline.yolov5_adapter import raw_yolo_predictions_to_selection

    selection = raw_yolo_predictions_to_selection(
        _raw_predictions(np.linspace(0.0, 1.0, 25_200, dtype=np.float32)),
        max_candidates=11_000,
    )

    assert selection.raw_proposal_count == 25_200
    assert selection.selected_count == 11_000
    assert selection.max_candidates == 11_000
    assert selection.boxes[0, 0] > selection.boxes[-1, 0]
    assert selection.effective_conf_threshold == pytest.approx(selection.scores[-1])


def test_adaptive_budget_expands_below_the_fixed_confidence_threshold():
    from baseline.yolov5_adapter import raw_yolo_predictions_to_selection

    selection = raw_yolo_predictions_to_selection(
        _raw_predictions(np.array([0.1, 0.3, 0.2], dtype=np.float32)),
        conf_threshold=0.99,
        max_candidates=11_000,
    )

    assert selection.selected_count == 3
    assert selection.scores.tolist() == pytest.approx([0.3, 0.2, 0.1])
    assert selection.effective_conf_threshold == pytest.approx(0.1)


def test_adaptive_budget_breaks_boundary_ties_by_raw_index():
    from baseline.yolov5_adapter import raw_yolo_predictions_to_selection

    selection = raw_yolo_predictions_to_selection(
        _raw_predictions(np.full(4, 0.5, dtype=np.float32)),
        max_candidates=2,
    )

    assert selection.selected_count == 2
    assert selection.boxes[:, 0].tolist() == pytest.approx([-0.5, 0.5])


@pytest.mark.parametrize("max_candidates", [0, -1, 1.5, True])
def test_adaptive_budget_rejects_invalid_max_candidates(max_candidates):
    from baseline.yolov5_adapter import raw_yolo_predictions_to_selection

    with pytest.raises(ValueError, match="max_candidates"):
        raw_yolo_predictions_to_selection(
            _raw_predictions(np.array([0.9], dtype=np.float32)),
            max_candidates=max_candidates,
        )


def test_threshold_only_selection_preserves_legacy_candidate_arrays():
    from baseline.yolov5_adapter import (
        raw_yolo_predictions_to_candidates,
        raw_yolo_predictions_to_selection,
    )

    raw = _raw_predictions(np.array([0.1, 0.8, 0.4], dtype=np.float32))
    expected = raw_yolo_predictions_to_candidates(raw, conf_threshold=0.3)
    selection = raw_yolo_predictions_to_selection(raw, conf_threshold=0.3)

    assert selection.max_candidates is None
    assert selection.effective_conf_threshold is None
    assert all(
        np.array_equal(actual, reference)
        for actual, reference in zip(
            (selection.boxes, selection.scores, selection.class_ids), expected
        )
    )
