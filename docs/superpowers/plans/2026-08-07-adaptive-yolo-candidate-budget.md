# Adaptive YOLO Candidate Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Bound raw YOLOv5 pre-NMS candidates by a stable score-ranked budget while recording the adaptive cutoff in the detector benchmark report.

**Architecture:** `baseline.yolov5_adapter` owns selection semantics and returns an immutable metadata holder. Its legacy tuple APIs delegate to the new helpers. `benchmarks/run_detector_pipeline.py` consumes the metadata holder and persists the selection facts without changing any NMS implementation, including V3.

**Tech Stack:** Python 3, NumPy, dataclasses, pytest, argparse, YOLOv5 raw network tensors.

## Global Constraints

- V3 source and notebooks are out of scope and must not change.
- `max_candidates=None` must preserve the existing `score >= conf_threshold` result exactly.
- `max_candidates=K` ranks all raw proposals by descending score and ascending raw index, then keeps exactly `min(K, raw_proposal_count)`.
- Adaptive mode records the lowest selected score as `effective_conf_threshold`; score ties at the boundary must never exceed K.
- The detector report must continue to measure candidate extraction and NMS separately and must retain its torchvision parity check.

---

### Task 1: Add a metadata-bearing, bounded raw-YOLO selection API

**Files:**

- Modify: `tests/baseline/test_yolov5_adapter.py`
- Modify: `tests/test_correctness.py`
- Modify: `src/baseline/yolov5_adapter.py`
- Modify: `src/cpu_baseline.py`

**Interfaces:**

- Produces: frozen `RawCandidateSelection` with `boxes`, `scores`, `class_ids`, `raw_proposal_count`, `selected_count`, `effective_conf_threshold`, and `max_candidates`.
- Produces: `raw_yolo_predictions_to_selection(raw_prediction, conf_threshold=0.01, max_candidates=None) -> RawCandidateSelection`.
- Produces: `load_raw_yolo_candidate_selection(image, conf_threshold=0.01, image_size=640, weights=None, model=None, max_candidates=None) -> RawCandidateSelection`.
- Preserves: `raw_yolo_predictions_to_candidates(...)` and `load_raw_yolo_candidates(...) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.

- [x] **Step 1: Write the failing selection tests**

Add tests with direct, synthetic raw tensors for the public selection helper:

```python
def test_adaptive_budget_keeps_exactly_top_k_raw_proposals():
    raw = np.zeros((25_200, 6), dtype=np.float32)
    raw[:, 0] = np.arange(25_200)
    raw[:, 2:4] = 1
    raw[:, 4] = np.linspace(0.0, 1.0, 25_200, dtype=np.float32)
    raw[:, 5] = 1.0

    selection = raw_yolo_predictions_to_selection(raw, max_candidates=11_000)

    assert selection.raw_proposal_count == 25_200
    assert selection.selected_count == 11_000
    assert selection.max_candidates == 11_000
    assert selection.boxes[0, 0] > selection.boxes[-1, 0]
    assert selection.effective_conf_threshold == pytest.approx(selection.scores[-1])
```

Also cover fewer-than-budget inputs, equal-score boundary retaining the lower raw indices, invalid budgets `0`, `-1`, `1.5`, and `True`, plus a legacy tuple comparison against the existing threshold-only expected arrays.

- [x] **Step 2: Run selection tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/baseline/test_yolov5_adapter.py tests/test_correctness.py -k 'adaptive or raw_yolo' -q`

Expected: FAIL because `raw_yolo_predictions_to_selection` and `RawCandidateSelection` do not exist.

- [x] **Step 3: Implement the smallest selection path**

In `src/baseline/yolov5_adapter.py`, validate the raw tensor once, compute class IDs and scores once, then branch on `max_candidates`:

```python
if max_candidates is None:
    selected_indices = np.flatnonzero(scores >= float(conf_threshold))
    effective_conf_threshold = None
else:
    _validate_max_candidates(max_candidates)
    selected_indices = np.lexsort((np.arange(len(scores)), -scores))[:max_candidates]
    effective_conf_threshold = float(scores[selected_indices[-1]]) if len(selected_indices) else None
```

Build canonical `xyxy` boxes from `selected_indices`, validate the three candidate arrays, and return the frozen selection object. The legacy helpers must return `selection.boxes`, `selection.scores`, and `selection.class_ids`. The live loader scales `selection.boxes` back to original image dimensions before returning a replacement selection with unchanged metadata. Re-export the new type and helpers through `src/cpu_baseline.py`.

- [x] **Step 4: Run focused tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/baseline/test_yolov5_adapter.py tests/test_correctness.py -k 'adaptive or raw_yolo' -q`

Expected: PASS, with optional live-model tests skipped only when weights are not available.

- [x] **Step 5: Commit the focused API change**

```bash
git add src/baseline/yolov5_adapter.py src/cpu_baseline.py tests/baseline/test_yolov5_adapter.py tests/test_correctness.py
git commit -m "feat: bound raw YOLO candidates by budget"
```

### Task 2: Persist adaptive selection facts in detector benchmark output

**Files:**

- Modify: `tests/test_benchmarks.py`
- Modify: `benchmarks/run_detector_pipeline.py`
- Modify: `src/baseline/explain.md`

**Interfaces:**

- Consumes: `RawCandidateSelection` from `load_raw_yolo_candidate_selection`.
- Produces: `build_detector_report(..., max_candidates=None)` with `raw_proposal_count`, `candidate_count`, `effective_conf_threshold`, and `max_candidates`.
- Produces: CLI flag `--max-candidates K` forwarded to the metadata loader.

- [x] **Step 1: Write the failing benchmark report test**

Replace the fake tuple loader with a fake `RawCandidateSelection`, then assert:

```python
assert report["raw_proposal_count"] == 25_200
assert report["candidate_count"] == 2
assert report["effective_conf_threshold"] == pytest.approx(0.042)
assert report["max_candidates"] == 11_000
assert report["configuration"]["max_candidates"] == 11_000
```

Use 2 valid candidate rows for the real NMS call while declaring the raw count and budget metadata, so the test remains fast and CPU-only. The 11,000-row cap is covered by Task 1's direct selection test.

- [x] **Step 2: Run the report test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_benchmarks.py::test_detector_report_separates_raw_candidates_from_nms_time -q`

Expected: FAIL because the report lacks adaptive selection fields or the builder cannot consume the metadata holder.

- [x] **Step 3: Implement benchmark and CLI integration**

Make the builder call a loader with `conf_threshold` and `max_candidates`, use the selection's three arrays for NMS, and write its metadata at report top level and under `configuration`. In `main`, load the YOLO model once and make the closure call `load_raw_yolo_candidate_selection(..., model=model, max_candidates=max_candidates)`. Add argparse:

```python
parser.add_argument(
    "--max-candidates", type=int, default=None,
    help="adaptive raw-YOLO pre-NMS candidate budget; e.g. 11000",
)
```

Document in `src/baseline/explain.md` that this is a test-input budget, not a production detection-confidence calibration.

- [x] **Step 4: Run the report test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_benchmarks.py::test_detector_report_separates_raw_candidates_from_nms_time -q`

Expected: PASS without loading YOLO weights or requiring CUDA.

- [x] **Step 5: Commit benchmark integration**

```bash
git add benchmarks/run_detector_pipeline.py tests/test_benchmarks.py src/baseline/explain.md
git commit -m "feat: report adaptive YOLO candidate budget"
```

### Task 3: Verify complete behavior and inspect a real raw output shape

**Files:**

- Modify: no production files.

**Interfaces:**

- Consumes: all existing tests and the real detector benchmark runner.
- Produces: a JSON artifact outside the repository with observed candidate selection metadata.

- [x] **Step 1: Run the complete local suite**

Run: `./.venv/bin/python -m pytest tests -q`

Expected: all CPU tests pass; CUDA-only tests may skip on a non-CUDA host.

- [x] **Step 2: Run static checks**

Run: `./.venv/bin/python -m compileall -q src benchmarks && git diff --check`

Expected: both commands exit 0.

- [x] **Step 3: Run the public-image integration check**

Run the existing `benchmarks/run_detector_pipeline.py` once with the public Zidane image, `--runner cpu`, `--conf-threshold 0.01`, and `--max-candidates 11000`; write JSON under `/private/tmp/`, never into the repository. If certificate configuration prevents downloading the image, use the project's existing `certifi` CA bundle environment variables and report that transport setup separately from timing results.

- [x] **Step 4: Inspect selection evidence**

Confirm the JSON identifies `yolo_raw_pre_nms`, records `raw_proposal_count >= candidate_count`, records `max_candidates=11000`, and does not claim GPU execution when CUDA is unavailable.
