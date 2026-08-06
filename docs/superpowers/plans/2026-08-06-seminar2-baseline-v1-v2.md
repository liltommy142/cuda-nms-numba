# Seminar 2 Baseline and V1/V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a catalog-aligned class-aware CPU NMS baseline with real detector integration and rebuilt V1/V2 hard-NMS paths, leaving Matrix NMS V3 untouched.

**Architecture:** `nms_common.py` owns the candidate contract, deterministic synthetic data and the per-class torchvision oracle. CPU, V1 and V2 consume `(boxes, scores, class_ids)`; synthetic data powers fixed-N stress benchmarks while raw YOLO predictions power a separate detector-to-NMS demonstration.

**Tech Stack:** Python, NumPy, PyTorch, torchvision, Ultralytics YOLO, Numba CUDA, pytest, JSON.

## Global Constraints

- Do not modify `src/gpu_v3.py`, `src/gpu_v3.ipynb`, V3-specific tests, or existing T4 evidence.
- CPU baseline contains no CUDA/Numba imports and runs without an NVIDIA GPU.
- Hard NMS is class-aware and matches a per-class `torchvision.ops.nms` oracle.
- Preserve stable descending-score ordering, including ties.
- Keep NMS-only synthetic benchmarks separate from real detector-plus-NMS measurements.
- Never claim a restructured performance result before it is generated on CUDA.

---

### Task 1: Canonical candidates and CPU contract

**Files:**
- Create: `src/nms_common.py`
- Modify: `tests/test_correctness.py`

**Interfaces:**
- Produces `validate_candidates(boxes, scores, class_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- Produces `load_synthetic_candidates(n: int, seed: int = 0, num_classes: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- Produces `stable_class_partitions(scores, class_ids) -> list[np.ndarray]`.
- Produces `torchvision_class_aware_nms(boxes, scores, class_ids, iou_threshold) -> np.ndarray`.

- [ ] **Step 1: Write failing tests**

```python
def test_validate_candidates_normalizes_dtypes_and_shapes():
    boxes, scores, class_ids = validate_candidates([[0, 0, 2, 2]], [0.9], [3])
    assert (boxes.dtype, scores.dtype, class_ids.dtype) == (np.float32, np.float32, np.int32)

def test_validate_candidates_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="x2 must be greater"):
        validate_candidates([[3, 0, 2, 1]], [0.9], [0])

def test_synthetic_candidates_are_deterministic_and_multiclass():
    first = load_synthetic_candidates(100, seed=8)
    second = load_synthetic_candidates(100, seed=8)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(np.unique(first[2])) > 1
```

- [ ] **Step 2: Confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_correctness.py -k 'validate_candidates or synthetic_candidates' -v`

Expected: FAIL because `nms_common` is absent.

- [ ] **Step 3: Implement contract utilities**

```python
def validate_candidates(boxes, scores, class_ids):
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    class_ids = np.ascontiguousarray(class_ids, dtype=np.int32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if scores.shape != (len(boxes),) or class_ids.shape != (len(boxes),):
        raise ValueError("scores and class_ids must have shape (N,)")
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
        raise ValueError("candidates must be finite")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("x2 must be greater than x1 and y2 must be greater than y1")
    return boxes, scores, class_ids
```

Implement clustered, deterministic xyxy synthetic boxes with float32 scores and int32 classes. The oracle applies torchvision NMS per class, then stable-sorts retained original indices by score.

- [ ] **Step 4: Verify CPU contract coverage**

Run: `.venv/bin/python -m pytest tests/test_correctness.py -k 'not gpu' -v`

Expected: CPU/reference tests PASS; CUDA tests skip locally.

- [ ] **Step 5: Commit**

```bash
git add src/nms_common.py tests/test_correctness.py
git commit -m "feat: add class-aware NMS candidate contract"
```

### Task 2: CPU baseline and raw detector source

**Files:**
- Modify: `src/cpu_baseline.py`
- Modify: `tests/test_correctness.py`
- Create: `tests/fixtures/raw_yolo_candidates.npz`

**Interfaces:**
- Consumes all Task 1 utilities.
- Produces `run_cpu(boxes, scores, class_ids=None, iou_threshold=0.5) -> np.ndarray`.
- Produces `load_raw_yolo_candidates(image, conf_threshold=0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing CPU tests**

```python
def test_cpu_keeps_overlapping_boxes_from_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    assert run_cpu(boxes, np.array([.9, .8], dtype=np.float32), np.array([0, 1])).tolist() == [0, 1]

def test_cpu_matches_torchvision_per_class():
    boxes, scores, class_ids = load_synthetic_candidates(200, seed=12)
    assert np.array_equal(run_cpu(boxes, scores, class_ids), torchvision_class_aware_nms(boxes, scores, class_ids, .5))

def test_saved_raw_yolo_fixture_uses_canonical_contract():
    fixture = np.load("tests/fixtures/raw_yolo_candidates.npz")
    validate_candidates(fixture["boxes"], fixture["scores"], fixture["class_ids"])
```

- [ ] **Step 2: Confirm legacy baseline fails**

Run: `.venv/bin/python -m pytest tests/test_correctness.py -k 'different_classes or torchvision_per_class or raw_yolo_fixture' -v`

Expected: FAIL because `run_cpu` has no class-id contract and the fixture is absent.

- [ ] **Step 3: Implement CPU NMS and raw model adapter**

```python
def run_cpu(boxes, scores, class_ids=None, iou_threshold=0.5):
    if class_ids is None:
        class_ids = np.zeros(len(scores), dtype=np.int32)
    boxes, scores, class_ids = validate_candidates(boxes, scores, class_ids)
    kept = []
    for indices in stable_class_partitions(scores, class_ids):
        kept.extend(indices[_run_cpu_single_class(boxes[indices], scores[indices], iou_threshold)])
    return stable_score_order(np.asarray(kept, dtype=np.int64), scores)
```

Read raw model prediction tensors, filter by objectness times the best class probability, convert xywh to xyxy and save one inspected result as the offline fixture. Do not use a post-NMS `Results.boxes`/AutoShape result. Add `--source synthetic|yolo-fixture|yolo-live`, `--benchmark`, `--profile` and `--verify` commands.

- [ ] **Step 4: Verify baseline**

Run: `.venv/bin/python -m pytest tests/test_correctness.py -k 'cpu or raw_yolo or validate_candidates or synthetic_candidates' -v`

Run: `.venv/bin/python src/cpu_baseline.py --source synthetic --benchmark --verify`

Expected: selected tests PASS and benchmark identifies `nms_only_synthetic`.

- [ ] **Step 5: Commit**

```bash
git add src/cpu_baseline.py tests/test_correctness.py tests/fixtures/raw_yolo_candidates.npz
git commit -m "feat: rebuild catalog-aligned CPU NMS baseline"
```

### Task 3: GPU V1 class-aware hard NMS

**Files:**
- Modify: `src/gpu_v1.py`
- Modify: `tests/test_correctness.py`
- Modify: `src/gpu_v1.ipynb`

**Interfaces:**
- Consumes Task 1 utilities and `run_cpu`.
- Produces `run_gpu_v1(boxes, scores, class_ids=None, iou_threshold=0.5) -> np.ndarray`.

- [ ] **Step 1: Write failing CUDA test**

```python
@requires_gpu
def test_gpu_v1_matches_cpu_and_torchvision_per_class():
    boxes, scores, class_ids = load_synthetic_candidates(300, seed=22)
    assert np.array_equal(run_gpu_v1(boxes, scores, class_ids), run_cpu(boxes, scores, class_ids))
```

- [ ] **Step 2: Confirm it fails on T4**

Run: `python -m pytest tests/test_correctness.py::test_gpu_v1_matches_cpu_and_torchvision_per_class -v`

Expected: FAIL because the third V1 argument is currently the IoU threshold.

- [ ] **Step 3: Adapt V1 without optimizing it**

```python
def run_gpu_v1(boxes, scores, class_ids=None, iou_threshold=0.5):
    if class_ids is None:
        class_ids = np.zeros(len(scores), dtype=np.int32)
    boxes, scores, class_ids = validate_candidates(boxes, scores, class_ids)
    kept = []
    for indices in stable_class_partitions(scores, class_ids):
        kept.extend(indices[_run_gpu_v1_single_class(boxes[indices], scores[indices], iou_threshold)])
    return stable_score_order(np.asarray(kept, dtype=np.int64), scores)
```

Move the legacy V1 body into `_run_gpu_v1_single_class`. Preserve one thread per pair, the full IoU matrix transfer and CPU greedy resolution. Update the notebook with class-aware parity and source labels.

- [ ] **Step 4: Verify**

Run locally: `.venv/bin/python -m pytest tests/test_correctness.py -k 'not gpu' -v`

Run on T4: `python -m pytest tests/test_correctness.py -k 'gpu_v1' -v`

Expected: CPU suite passes locally; selected V1 tests pass on T4.

- [ ] **Step 5: Commit**

```bash
git add src/gpu_v1.py src/gpu_v1.ipynb tests/test_correctness.py
git commit -m "feat: make GPU V1 class-aware"
```

### Task 4: GPU V2 class-aware hard NMS

**Files:**
- Modify: `src/gpu_v2.py`
- Modify: `tests/test_correctness.py`
- Modify: `benchmarks/run_v2_batch.py`
- Modify: `src/gpu_v2.ipynb`

**Interfaces:**
- Consumes Task 1 utilities and `run_cpu`.
- Produces `run_gpu_v2(boxes, scores, class_ids=None, iou_threshold=0.5) -> np.ndarray | list[np.ndarray]`.

- [ ] **Step 1: Write failing CUDA tests**

```python
@requires_gpu
def test_gpu_v2_matches_cpu_per_class():
    boxes, scores, class_ids = load_synthetic_candidates(300, seed=32)
    assert np.array_equal(run_gpu_v2(boxes, scores, class_ids), run_cpu(boxes, scores, class_ids))

@requires_gpu
def test_gpu_v2_batched_multiclass_matches_cpu():
    samples = [load_synthetic_candidates(50, seed=i) for i in range(3)]
    actual = run_gpu_v2(np.stack([x[0] for x in samples]), np.stack([x[1] for x in samples]), np.stack([x[2] for x in samples]))
    assert all(np.array_equal(result, run_cpu(*sample)) for result, sample in zip(actual, samples))
```

- [ ] **Step 2: Confirm they fail on T4**

Run: `python -m pytest tests/test_correctness.py -k 'gpu_v2 and multiclass' -v`

Expected: FAIL because V2 lacks class-id handling.

- [ ] **Step 3: Adapt V2 partition-by-class execution**

```python
def run_gpu_v2(boxes, scores, class_ids=None, iou_threshold=0.5):
    if class_ids is None:
        class_ids = np.zeros(np.asarray(scores).shape, dtype=np.int32)
    if np.asarray(boxes).ndim == 3:
        return [run_gpu_v2(b, s, c, iou_threshold) for b, s, c in zip(boxes, scores, class_ids)]
    boxes, scores, class_ids = validate_candidates(boxes, scores, class_ids)
    kept = []
    for indices in stable_class_partitions(scores, class_ids):
        kept.extend(indices[_run_gpu_v2_single_class(boxes[indices], scores[indices], iou_threshold)])
    return stable_score_order(np.asarray(kept, dtype=np.int64), scores)
```

Keep the existing SoA/shared-memory/uint64-mask CUDA kernel behind `_run_gpu_v2_single_class`. Preserve host greedy-mask resolution and state it as the V2 limitation. Add class ids and `benchmark_scope: "nms_only_synthetic"` to the batch report.

- [ ] **Step 4: Verify**

Run on T4: `python -m pytest tests/test_correctness.py -k 'gpu_v2' -v`

Run on T4: `python benchmarks/run_v2_batch.py --batch-size 32 --n 10000 --warmup 2 --repeats 7 --json benchmarks/results/v2_batch32_restructured.json`

Expected: tests pass; JSON labels the source/scope and reports the measured value without an unsupported target claim.

- [ ] **Step 5: Commit**

```bash
git add src/gpu_v2.py src/gpu_v2.ipynb benchmarks/run_v2_batch.py tests/test_correctness.py
git commit -m "feat: make GPU V2 class-aware"
```

### Task 5: Separate evidence and Seminar 2 narrative

**Files:**
- Modify: `benchmarks/run_all.py`
- Create: `benchmarks/run_detector_pipeline.py`
- Create: `tests/test_benchmarks.py`
- Modify: `README.md`, `docs/HOW_TO_RUN.md`
- Modify: `presentation/seminar_2/README.md`, `presentation/seminar_2/OUTLINE_AND_CONTENT.md`, `presentation/seminar_2/SCRIPT.md`, `presentation/seminar_2/QA_PREP.md`

**Interfaces:**
- Produces NMS report field `benchmark_scope: "nms_only_synthetic"`.
- Produces detector report field `benchmark_scope: "detector_plus_nms_real"`.

- [ ] **Step 1: Write failing report-metadata test**

```python
def test_synthetic_benchmark_report_has_explicit_scope():
    report = build_synthetic_report(n=100, repeats=1, warmup=0, seed=0, versions=["cpu"])
    assert report["benchmark_scope"] == "nms_only_synthetic"
    assert report["candidate_source"] == "deterministic_synthetic"
```

- [ ] **Step 2: Confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_benchmarks.py -v`

Expected: FAIL because the report builder does not exist.

- [ ] **Step 3: Implement report builders and copy changes**

```python
report = {
    "benchmark_scope": "nms_only_synthetic",
    "candidate_source": "deterministic_synthetic",
    "timing_scope": "candidate NMS only; excludes model inference",
    "input_semantics": "one image / one class-labelled candidate set",
    "results": results,
}
```

Create `run_detector_pipeline.py` with raw detector inference and NMS timings as separate fields plus candidate count, class count and torchvision parity. Update all Seminar documents to distinguish synthetic stress results, real integration, V1/V2 parity and untouched V3 Matrix NMS. Historical T4 values must be labelled pre-restructure until rerun.

- [ ] **Step 4: Verify reports and host suite**

Run: `.venv/bin/python -m pytest tests -q`

Run: `.venv/bin/python benchmarks/run_all.py --versions cpu --n 100 --repeats 1 --warmup 0 --json /tmp/nms-synthetic.json`

Run: `jq '.benchmark_scope, .candidate_source, .timing_scope' /tmp/nms-synthetic.json`

Expected: tests pass; JSON prints the three explicit scope values.

- [ ] **Step 5: Commit**

```bash
git add benchmarks tests README.md docs/HOW_TO_RUN.md presentation/seminar_2
git commit -m "docs: clarify Seminar 2 benchmark semantics"
```

### Task 6: Colab T4 verification and final artifacts

**Files:**
- Create: `presentation/seminar_2/evidence/pytest_restructured_t4.txt`
- Create: `presentation/seminar_2/evidence/benchmark_restructured_t4.json`
- Create: `presentation/seminar_2/evidence/detector_pipeline_restructured_t4.json`
- Modify: `presentation/seminar_2/Seminar_2_Final_T4.pptx`, `presentation/seminar_2/Seminar_2_Final_T4.pdf`

**Interfaces:**
- Consumes completed code on the exact branch/commit being presented.
- Produces evidence files and a deck whose every number comes from them.

- [ ] **Step 1: Verify V3 preservation**

Run: `git diff --exit-code c3df233 -- src/gpu_v3.py src/gpu_v3.ipynb`

Expected: exit code 0.

- [ ] **Step 2: Run correctness before timing**

Run in Colab: `python -m pytest tests -v | tee presentation/seminar_2/evidence/pytest_restructured_t4.txt`

Expected: CPU, V1, V2 and unchanged V3 tests pass.

- [ ] **Step 3: Generate separate T4 evidence**

```bash
python benchmarks/run_all.py --versions cpu v1 v2 --repeats 7 --warmup 2 \
  --json presentation/seminar_2/evidence/benchmark_restructured_t4.json
python benchmarks/run_detector_pipeline.py --runner cpu --repeats 3 --warmup 1 \
  --json presentation/seminar_2/evidence/detector_pipeline_restructured_t4.json
```

- [ ] **Step 4: Update and validate the deck**

Replace stale values with the new evidence or remove them. Add one visual distinction between synthetic stress benchmarks and real detector integration. Render all slides and run overflow checks.

- [ ] **Step 5: Commit final evidence and deck**

```bash
git add presentation/seminar_2
git commit -m "docs: publish restructured Seminar 2 evidence"
```
