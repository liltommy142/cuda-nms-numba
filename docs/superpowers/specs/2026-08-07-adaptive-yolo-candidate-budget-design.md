# Adaptive YOLO Candidate Budget Design

**Goal:** Let the raw YOLOv5 adapter supply as many pre-NMS candidates as
possible without exceeding a configurable budget, initially 11,000 boxes.

## Scope

- Extend the Baseline raw-YOLO adapter with an optional candidate budget.
- Preserve the current three-array public result when the budget is omitted.
- Record the effective cutoff and candidate counts in detector benchmark JSON.
- Add deterministic CPU tests for cap, expansion, tie handling and legacy mode.
- Do not modify V3.

## Selection Semantics

Raw YOLO scores remain:

```text
score[i] = objectness[i] * max_class_probability[i]
```

The public adapter gains `max_candidates: int | None = None`.

- `max_candidates=None`: retain the current behavior exactly: keep all proposals
  where `score >= conf_threshold`.
- `max_candidates=K`: adaptive mode ignores `conf_threshold` as a fixed floor,
  ranks every raw proposal by descending score and stable raw index, then keeps
  exactly `min(K, raw_proposal_count)` proposals.

The score of the lowest-ranked selected proposal is reported as
`effective_conf_threshold`. It describes the adaptive cutoff, but stable
top-K selection is authoritative: equal scores at the boundary never allow the
candidate count to exceed K.

This is a pre-NMS candidate budget for stress testing and controlled GPU input.
It is not a production detector-confidence policy and must not be described as
one in Seminar 2 results.

## API Shape

Introduce an immutable `RawCandidateSelection` with:

- `boxes: np.ndarray`
- `scores: np.ndarray`
- `class_ids: np.ndarray`
- `raw_proposal_count: int`
- `selected_count: int`
- `effective_conf_threshold: float | None`
- `max_candidates: int | None`

`raw_yolo_predictions_to_selection(...)` and
`load_raw_yolo_candidate_selection(...)` return this metadata-bearing value.

Existing `raw_yolo_predictions_to_candidates(...)` and
`load_raw_yolo_candidates(...)` keep their three-array return contract and
delegate internally to the selection helpers. Existing callers therefore need
no change unless they request adaptive metadata.

## Benchmark Integration

`benchmarks/run_detector_pipeline.py` gains `--max-candidates`. Its loader
uses the metadata-bearing adapter path and writes these fields into the JSON:

- `raw_proposal_count`
- `candidate_count`
- `effective_conf_threshold`
- `max_candidates`

The report still separates raw detector candidate extraction from NMS time and
still verifies the kept output with per-class `torchvision.ops.nms`.

## Acceptance Checks

- Legacy threshold filtering returns exactly the current arrays.
- With K=11,000 and 25,200 raw scores, output length is exactly 11,000.
- With fewer than K raw proposals, all proposals are retained.
- A tie at the budget boundary is resolved by original raw index and never
  exceeds K.
- Invalid budgets (`0`, negative, non-integer) raise `ValueError`.
- Detector report records the adaptive fields whenever a budget is supplied.
