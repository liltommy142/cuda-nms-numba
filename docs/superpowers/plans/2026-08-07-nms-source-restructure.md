# NMS Source Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Organize the baseline, V1, and V2 implementations into focused version packages without changing their hard-NMS results, supported CLI commands, or public imports; preserve V3 unchanged.

**Architecture:** `src/common` owns the candidate contract and reference oracle. `src/baseline`, `src/v1`, and `src/v2` own their core algorithm, device kernels where applicable, and CLI. The historic root modules remain small, documented façades so existing scripts, benchmarks, notebooks, and V3 continue to import their established names.

**Tech Stack:** Python 3, NumPy, Numba CUDA, PyTorch/Torchvision, pytest.

## Global Constraints

- Keep `src/gpu_v3.py` and `src/gpu_v3.ipynb` byte-for-byte unchanged.
- Preserve the candidate contract: finite `float32 (N, 4)` `xyxy` boxes, finite `float32 (N,)` scores, and `int32 (N,)` class IDs.
- Preserve hard-NMS output order: original indices in descending score, then input-index order; suppress only within a class.
- Preserve CLI entry paths: `python src/cpu_baseline.py`, `python src/gpu_v1.py`, and `python src/gpu_v2.py`.
- Preserve historical import paths through thin façade modules; `load_data` remains a one-class adapter used by V3 only.
- Do not create, delete, or alter benchmark evidence during this refactor.
- CUDA tests may skip without an NVIDIA CUDA device; all CPU, import, CLI-help, and report-contract tests must pass.

---

### Task 1: Create the package skeleton and prove the CPU compatibility bridge

**Files:**
- Create: `src/common/__init__.py`, `src/baseline/__init__.py`, `src/v1/__init__.py`, `src/v2/__init__.py`
- Create: `tests/compat/test_public_facades.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces importable packages rooted at `src` and a temporary CPU bridge that
  is replaced by the real core in Task 3.
- Produces a shared pytest import-path fixture for all reorganized tests.

- [ ] **Step 1: Write the failing façade test**

```python
def test_cpu_package_core_matches_legacy_facade():
    import cpu_baseline
    from baseline.core import run_cpu

    assert cpu_baseline.run_cpu is run_cpu
```

- [ ] **Step 2: Run it to verify it fails because `baseline.core` is absent**

Run: `pytest tests/compat/test_public_facades.py::test_cpu_facade_reexports_package_core -q`

Expected: import failure for `baseline` or `baseline.core`.

- [ ] **Step 3: Add package markers and a temporary, importable CPU bridge**

```python
# src/baseline/core.py
"""Temporary bridge while CPU code is extracted in Task 3."""
from cpu_baseline import iou_one_to_many, run_cpu, verify
```

Do not rewrite the root façade until Task 3; this bridge makes the package
boundary testable without changing runtime behaviour.

- [ ] **Step 4: Run the focused test and then the current CPU tests**

Run: `pytest tests/compat/test_public_facades.py tests/test_correctness.py -k 'cpu or facade' -q`

Expected: the compatibility-bridge test and existing CPU tests pass.

- [ ] **Step 5: Commit the skeleton and façade-test boundary**

```bash
git add src/common src/baseline src/v1 src/v2 tests/conftest.py tests/compat/test_public_facades.py
git commit -m "refactor: add versioned NMS package skeleton"
```

### Task 2: Extract the shared candidate contract and oracle

**Files:**
- Create: `src/common/candidates.py`, `src/common/oracle.py`
- Modify: `src/nms_common.py`
- Create: `tests/common/test_candidates.py`

**Interfaces:**
- `common.candidates` provides `validate_candidates`, `validate_iou_threshold`, `stable_score_order`, `stable_class_partitions`, and `load_synthetic_candidates`.
- `common.oracle` provides `torchvision_class_aware_nms`.
- `nms_common` re-exports these names for existing imports.

- [ ] **Step 1: Write a failing direct-package test**

```python
def test_common_contract_has_stable_class_partitions():
    from common.candidates import stable_class_partitions

    scores = np.array([0.5, 0.9, 0.9], dtype=np.float32)
    class_ids = np.array([1, 0, 0], dtype=np.int32)
    assert [part.tolist() for part in stable_class_partitions(scores, class_ids)] == [[1, 2], [0]]
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/common/test_candidates.py::test_common_contract_has_stable_class_partitions -q`

Expected: import failure because `common.candidates` does not yet exist.

- [ ] **Step 3: Move the shared functions without changing their bodies, then turn `nms_common.py` into a documented re-export façade**

```python
# src/nms_common.py
"""Compatibility exports for the canonical candidate contract."""
from common.candidates import load_synthetic_candidates, stable_class_partitions
from common.candidates import stable_score_order, validate_candidates, validate_iou_threshold
from common.oracle import torchvision_class_aware_nms
```

- [ ] **Step 4: Verify direct and compatibility imports**

Run: `pytest tests/common/test_candidates.py tests/test_correctness.py -k 'validate or synthetic or torchvision' -q`

Expected: shared-package and legacy `nms_common` paths return identical arrays and oracle results.

- [ ] **Step 5: Commit shared extraction**

```bash
git add src/common src/nms_common.py tests/common/test_candidates.py
git commit -m "refactor: isolate shared NMS candidate contract"
```

### Task 3: Separate the CPU baseline, raw YOLO adapter, and CLI

**Files:**
- Create: `src/baseline/core.py`, `src/baseline/yolov5_adapter.py`, `src/baseline/cli.py`
- Modify: `src/cpu_baseline.py`
- Create: `tests/baseline/test_core.py`, `tests/baseline/test_yolov5_adapter.py`

**Interfaces:**
- `baseline.core` provides `iou_one_to_many`, `run_cpu`, and `verify`.
- `baseline.yolov5_adapter` provides `raw_yolo_predictions_to_candidates`, `load_raw_yolov5_model`, and `load_raw_yolo_candidates`.
- `baseline.cli.main()` retains the current argument and output contract.
- `cpu_baseline` re-exports the above plus `load_data(n, seed=0)` for V3 compatibility.

- [ ] **Step 1: Write a failing package-adapter compatibility test**

```python
def test_cpu_facade_reexports_package_yolo_adapter():
    import cpu_baseline
    from baseline.yolov5_adapter import raw_yolo_predictions_to_candidates

    assert cpu_baseline.raw_yolo_predictions_to_candidates is raw_yolo_predictions_to_candidates
```

- [ ] **Step 2: Verify RED against the rewritten façade branch**

Run: `pytest tests/baseline/test_yolov5_adapter.py::test_cpu_facade_reexports_package_yolo_adapter -q`

Expected: import failure for `baseline.yolov5_adapter`.

- [ ] **Step 3: Move CPU, detector, and argparse responsibilities into their named modules; implement the root façade**

```python
# src/cpu_baseline.py
from baseline.cli import main
from baseline.core import iou_one_to_many, run_cpu, verify
from baseline.yolov5_adapter import (
    load_raw_yolo_candidates,
    load_raw_yolov5_model,
    raw_yolo_predictions_to_candidates,
)
from common.candidates import load_synthetic_candidates

def load_data(n, seed=0):
    boxes, scores, _ = load_synthetic_candidates(n, seed=seed)
    return boxes, scores
```

- [ ] **Step 4: Verify CPU behaviour and CLI scope output**

Run: `pytest tests/baseline tests/compat/test_public_facades.py -q && python src/cpu_baseline.py --source synthetic --benchmark --verify`

Expected: CPU tests pass and stdout contains `benchmark_scope: nms_only_synthetic`.

- [ ] **Step 5: Commit baseline split**

```bash
git add src/baseline src/cpu_baseline.py tests/baseline tests/compat/test_public_facades.py
git commit -m "refactor: split CPU baseline responsibilities"
```

### Task 4: Extract V1 CUDA kernel, orchestration, and CLI

**Files:**
- Create: `src/v1/kernel.py`, `src/v1/core.py`, `src/v1/cli.py`
- Modify: `src/gpu_v1.py`
- Create: `tests/v1/test_v1.py`

**Interfaces:**
- `v1.kernel.compute_iou_matrix_gpu(boxes)` returns an `(N, N)` host `float32` array.
- `v1.core.run_gpu_v1(boxes, scores, class_ids=None, iou_threshold=0.5)` returns original indexes in stable score order.
- `gpu_v1` re-exports `compute_iou_matrix_gpu`, `run_gpu_v1`, and `benchmark`; `main()` delegates to `v1.cli.main`.

- [ ] **Step 1: Write a failing direct-core test**

```python
def test_v1_core_preserves_class_aware_output_without_cuda(monkeypatch):
    from v1 import core

    monkeypatch.setattr(
        core, "_run_single_class", lambda boxes, threshold: np.arange(len(boxes))
    )
    assert core.run_gpu_v1(BOXES, SCORES, CLASS_IDS).tolist() == [1, 3, 2, 0]
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/v1/test_v1.py::test_v1_core_preserves_class_aware_output_without_cuda -q`

Expected: import failure for `v1.core`.

- [ ] **Step 3: Move the CUDA matrix kernel into `v1.kernel`, retain only class partitioning and the CPU resolver in `v1.core`, and make `gpu_v1.py` a façade**

Use `load_synthetic_candidates` in new V1 CLI/benchmark code. Keep `load_data` only at the V3-compatible boundary.

- [ ] **Step 4: Verify focused CPU-level and CUDA-gated tests**

Run: `pytest tests/v1 tests/compat/test_public_facades.py -q`

Expected: non-CUDA façade/class-routing tests pass; hardware tests are skipped only when CUDA is unavailable.

- [ ] **Step 5: Commit V1 split**

```bash
git add src/v1 src/gpu_v1.py tests/v1 tests/compat/test_public_facades.py
git commit -m "refactor: separate V1 kernel and orchestration"
```

### Task 5: Extract V2 kernels, class-aware routing, and CLI

**Files:**
- Create: `src/v2/kernels.py`, `src/v2/core.py`, `src/v2/cli.py`
- Modify: `src/gpu_v2.py`
- Create: `tests/v2/test_v2.py`

**Interfaces:**
- `v2.kernels.compute_iou_matrix_gpu_v2(boxes)` retains the current host-array result.
- `v2.core.run_gpu_v2` accepts both `(N, 4)` and `(B, N, 4)` boxes and returns one index array or a list of index arrays.
- `v2.core.run_gpu_v2_batched` retains the class-aware batch contract.
- `gpu_v2` re-exports both runners, both benchmark helpers, and `compute_iou_matrix_gpu_v2`; `main()` delegates to `v2.cli.main`.

- [ ] **Step 1: Write a failing direct V2 routing test**

```python
def test_v2_direct_core_routes_multiclass_batch_without_cuda(monkeypatch):
    from v2 import core

    monkeypatch.setattr(
        core, "_run_single_class", lambda boxes, scores, threshold: np.arange(len(boxes))
    )
    result = core.run_gpu_v2(BATCH_BOXES, BATCH_SCORES, BATCH_CLASS_IDS)
    assert [item.tolist() for item in result] == [[1, 3, 2, 0], [1, 3, 2, 0]]
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/v2/test_v2.py::test_v2_direct_core_routes_multiclass_batch_without_cuda -q`

Expected: import failure for `v2.core`.

- [ ] **Step 3: Move both CUDA kernels to `v2.kernels`; keep input validation, class partitioning, host transfers, and packed-mask resolution in `v2.core`; move argparse/printing to `v2.cli`**

New V2 synthetic batch generation must use all three canonical arrays so its CLI, verification, and benchmark setup stay class-aware.

- [ ] **Step 4: Verify V2 contracts**

Run: `pytest tests/v2 tests/compat/test_public_facades.py tests/test_benchmarks.py -q`

Expected: batch/class-routing and report contracts pass; CUDA-only kernel comparisons skip only without CUDA.

- [ ] **Step 5: Commit V2 split**

```bash
git add src/v2 src/gpu_v2.py tests/v2 tests/compat/test_public_facades.py tests/test_benchmarks.py
git commit -m "refactor: separate V2 kernels and batch routing"
```

### Task 6: Reorganize the test tree and document canonical imports

**Files:**
- Create: `tests/conftest.py`, `tests/benchmarks/test_reports.py`
- Move: candidate, baseline, V1, and V2 tests from `tests/test_correctness.py` into their named package directories
- Move: `tests/test_benchmarks.py` to `tests/benchmarks/test_reports.py`
- Modify: `benchmarks/run_all.py`, `benchmarks/run_detector_pipeline.py`, `benchmarks/run_v2_batch.py` only if a façade import obscures the intended owner
- Modify: `docs/TECHNICAL_DOCUMENTATION.md`, `docs/HOW_TO_RUN.md`

**Interfaces:**
- Test collection remains `pytest tests -q`.
- Benchmark scripts keep their command-line arguments and JSON schema; their imports may point at canonical packages when this improves clarity without changing callable identity.
- V3 test assertions may be moved into `tests/v3/test_v3_legacy.py`, but `src/gpu_v3.py` and its notebook must not change.

- [ ] **Step 1: Move tests by responsibility without altering assertions, update only clarity-improving benchmark imports, and replace documentation file-tree text with the final layout**

Do not change timing fields, candidate-source labels, or evidence paths. Remove empty legacy test files only after all tests collect under the new tree.

- [ ] **Step 2: Verify complete collection and CLI help**

Run: `pytest tests -q && python src/cpu_baseline.py --help && python src/gpu_v1.py --help && python src/gpu_v2.py --help`

Expected: no collection failures; all three help commands exit zero even on a non-CUDA host.

- [ ] **Step 3: Commit tests, benchmark imports, and documentation**

```bash
git add tests benchmarks docs/TECHNICAL_DOCUMENTATION.md docs/HOW_TO_RUN.md
git commit -m "refactor: organize NMS tests and benchmark imports"
```

### Task 7: Perform final regression and repository hygiene checks

**Files:**
- Verify only: `src/gpu_v3.py`, `src/gpu_v3.ipynb`, all migrated source, tests, docs, and notebooks
- Remove only generated caches: `.pytest_cache/`, `**/__pycache__/`, `.DS_Store`

**Interfaces:**
- Every stated compatibility façade and CLI has a fresh verification result.
- V3 has no working-tree diff.

- [ ] **Step 1: Run the complete Python validation suite**

Run: `MPLCONFIGDIR=/private/tmp/mpl-cuda-nms .venv/bin/python -m pytest tests -q && .venv/bin/python -m compileall -q src benchmarks`

Expected: all non-CUDA tests pass; CUDA tests skip only when hardware is unavailable; compilation succeeds.

- [ ] **Step 2: Validate retained notebook JSON and protected V3 boundary**

Run: `.venv/bin/python -m json.tool src/gpu_v1.ipynb >/dev/null && .venv/bin/python -m json.tool src/gpu_v2.ipynb >/dev/null && git diff --exit-code -- src/gpu_v3.py src/gpu_v3.ipynb`

Expected: both notebook files parse; no V3 diff.

- [ ] **Step 3: Inspect tracked scope and remove generated cache only**

Run: `git status --short --ignored`

Expected: no new benchmark/evidence outputs; retain user-owned `.venv/`, local YOLO weights, and submission archive.

- [ ] **Step 4: Commit and push the verified refactor using an explicit file list**

```bash
git add src/common src/baseline src/v1 src/v2 src/nms_common.py \
  src/cpu_baseline.py src/gpu_v1.py src/gpu_v2.py \
  tests benchmarks docs/TECHNICAL_DOCUMENTATION.md docs/HOW_TO_RUN.md
git commit -m "refactor: organize baseline and GPU NMS versions"
git push origin main
```
