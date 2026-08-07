# One-click Colab GPU Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one Colab notebook that runs the verified Baseline/V1/V2 GPU test and benchmark workload from a fresh CUDA 13 runtime.

**Architecture:** The notebook orchestrates existing project scripts: it checks out a revision, builds an isolated CUDA 13 environment, and saves their outputs as evidence. A lightweight pytest test reads the notebook JSON to stop workflow regression without running Colab-only cells.

**Tech Stack:** Jupyter notebook format 4, Google Colab, Bash, Python `json`, pytest, `virtualenv`, `numba-cuda[cu13]`, CPU-only PyTorch.

## Global Constraints

- The default Run all path covers Baseline, V1, and V2; it must not import or execute V3.
- Use `/content/nms-cu13-venv`, NumPy 1.26.4, pytest 9.1.1, `numba-cuda[cu13]`, `torch==2.5.1`, and `torchvision==0.20.1` from the CPU index.
- Default to `COMMIT=main`, resolve and record the actual commit SHA, and allow replacing `main` with a full SHA.
- Disable detector timing by default; when manually enabled with `yolov5s.pt`, use `--max-candidates 11000`.
- Bash cells use `set -euo pipefail`; evidence lives in `/content/evidence/<short-sha>/` and is archived to ZIP.

---

### Task 1: Define the notebook contract with a failing structural test

**Files:**

- Create: `tests/test_colab_notebooks.py`
- Test: `tests/test_colab_notebooks.py`

**Interfaces:**

- Consumes: `collab/gpu_test_colab.ipynb` as an `nbformat == 4` JSON document.
- Produces: `test_colab_gpu_notebook_contains_complete_v1_v2_workflow()`.

- [ ] **Step 1: Write the failing test**

```python
payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
source = chr(10).join("".join(cell.get("source", [])) for cell in payload["cells"])
assert payload["nbformat"] == 4
for marker in ("COMMIT=main", "numba-cuda[cu13]", "nms-cu13-venv",
               "python -m pytest tests -q -rs", "benchmarks/run_all.py",
               "benchmarks/run_v2_batch.py", "files.download",
               "--max-candidates 11000"):
    assert marker in source
assert "gpu_v3" not in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_colab_notebooks.py -v`

Expected: `FileNotFoundError` because the notebook does not yet exist.

- [ ] **Step 3: Commit the red test**

```bash
git add tests/test_colab_notebooks.py
git commit -m "test: define Colab notebook workflow contract"
```

### Task 2: Create the self-contained notebook

**Files:**

- Create: `collab/gpu_test_colab.ipynb`
- Test: `tests/test_colab_notebooks.py`

**Interfaces:**

- Consumes: the marker contract from Task 1.
- Produces: an output-free notebook that creates a valid evidence ZIP in Colab.

- [ ] **Step 1: Add ordered code cells**

1. Display `nvidia-smi`.
2. Set `COMMIT=main`; clone/fetch/checkout `/content/cuda-nms-numba` and print its SHA.
3. Rebuild `/content/nms-cu13-venv` and install the required CUDA 13 packages.
4. Run the V1 two-box CUDA JIT smoke test.
5. Run `pytest tests -q -rs`, then `benchmarks/run_all.py` and `benchmarks/run_v2_batch.py`; tee output to the evidence directory.
6. Record environment metadata and ZIP the evidence directory.
7. Keep `RUN_DETECTOR=False` around the optional V2 detector command with `--max-candidates 11000`.
8. Add a separate manual-download cell using `files.download`, disabled unless explicitly toggled.

- [ ] **Step 2: Run the focused test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_colab_notebooks.py -v`

Expected: PASS.

- [ ] **Step 3: Validate JSON and whitespace**

Run: `.venv/bin/python -m json.tool collab/gpu_test_colab.ipynb >/dev/null && git diff --check`

Expected: exit status 0.

- [ ] **Step 4: Commit the implementation**

```bash
git add collab/gpu_test_colab.ipynb tests/test_colab_notebooks.py
git commit -m "feat: add one-click Colab GPU test notebook"
```

### Task 3: Link and verify the notebook

**Files:**

- Modify: `collab/readme.md`
- Verify: `tests/`

**Interfaces:**

- Consumes: the checked-in notebook path.
- Produces: a runbook that directs presenters to the notebook before the terminal troubleshooting instructions.

- [ ] **Step 1: Add a direct link to `gpu_test_colab.ipynb`**

State that selecting a T4 GPU and choosing Run all executes the required workflow; retain terminal steps as a fallback. Do not imply the optional detector runs by default.

- [ ] **Step 2: Run full local verification**

Run: `.venv/bin/python -m pytest tests -q -rs && .venv/bin/python -m json.tool collab/gpu_test_colab.ipynb >/dev/null && git diff --check`

Expected: all runnable local tests pass; CUDA tests may skip on non-NVIDIA hardware; JSON and whitespace validation pass.

- [ ] **Step 3: Commit documentation and verification-ready changes**

```bash
git add collab/readme.md collab/gpu_test_colab.ipynb tests/test_colab_notebooks.py
git commit -m "docs: link one-click Colab notebook"
```
