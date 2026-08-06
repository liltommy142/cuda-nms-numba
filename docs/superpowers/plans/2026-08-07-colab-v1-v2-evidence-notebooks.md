# Colab V1/V2 Evidence Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace obsolete V1/V2 GPU notebooks with two Google Colab notebooks
that record reproducible CUDA correctness and benchmark evidence.

**Architecture:** `collab/v1_gpu_colab.ipynb` and
`collab/v2_gpu_colab.ipynb` are self-contained runners: they clone a selected
commit, install dependencies, gate on CUDA, run focused tests, benchmark, and
download evidence. Existing Python modules and benchmarks remain the only
implementation source of truth.

**Tech Stack:** Jupyter Notebook JSON, Google Colab, Python 3, Numba CUDA,
pytest, existing `benchmarks/run_all.py` and `benchmarks/run_v2_batch.py`.

## Global Constraints

- V3 source and `src/gpu_v3.ipynb` are out of scope and must not change.
- Each Colab result must record a user-selected git SHA and CUDA environment.
- A CUDA skip/failure must stop before benchmark output is treated as evidence.
- No notebook may store a GitHub token or push to GitHub.
- Keep the user-modified `docs/INDEX.md` outside this change unless explicitly
  requested.

---

### Task 1: Add structural tests for the new Colab artifacts

**Files:**
- Create: `tests/test_colab_notebooks.py`
- Test: `tests/test_colab_notebooks.py`

**Interfaces:**
- Consumes: JSON notebook files under `collab/`.
- Produces: a local, GPU-free check that the two notebooks are valid, distinct
  and contain the required execution contract.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook_source(name: str) -> str:
    notebook = json.loads((ROOT / "collab" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_colab_v1_and_v2_have_required_evidence_flow():
    for name, marker in (("v1_gpu_colab.ipynb", "--versions cpu v1"),
                         ("v2_gpu_colab.ipynb", "--versions cpu v1 v2")):
        source = notebook_source(name)
        assert "numba.cuda.is_available()" in source
        assert "git rev-parse HEAD" in source
        assert marker in source
        assert "/content/evidence" in source
        assert "files.download" in source
        assert "gpu_v3" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_colab_notebooks.py -q`

Expected: FAIL because `collab/v1_gpu_colab.ipynb` and
`collab/v2_gpu_colab.ipynb` do not exist.

- [ ] **Step 3: Keep the test focused**

Do not assert timing values or actual CUDA availability; those require a
Colab runtime. The test only protects the notebook contract and prevents a
future accidental V3 coupling.

- [ ] **Step 4: Run the focused test after notebooks exist**

Run: `./.venv/bin/python -m pytest tests/test_colab_notebooks.py -q`

Expected: PASS.

### Task 2: Replace the obsolete V1 notebook

**Files:**
- Delete: `src/gpu_v1.ipynb`
- Create: `collab/v1_gpu_colab.ipynb`
- Test: `tests/test_colab_notebooks.py`

**Interfaces:**
- Consumes: public GitHub repository URL and `COMMIT` cell variable.
- Produces: `/content/evidence/v1/<commit>/` containing test output,
  `benchmark_v1.json` and `environment.txt`.

- [ ] **Step 1: Create a Colab metadata notebook with six labelled sections**

Use Python 3 kernelspec metadata and these markdown headings: `1. Runtime
gate`, `2. Source setup`, `3. Reference test`, `4. Warm-up`, `5. Benchmark`,
and `6. Evidence export`.

- [ ] **Step 2: Add source and CUDA-gate cells**

The setup cell must set `REPO_URL`, `COMMIT = "main"`, clone to
`/content/cuda-nms-numba`, checkout `$COMMIT`, install `requirements.txt`,
and export `PYTHONPATH=$PWD/src`. The runtime cell must run `nvidia-smi` and
raise `RuntimeError("CUDA GPU is required for this evidence notebook")` when
`numba.cuda.is_available()` is false.

- [ ] **Step 3: Add the V1 evidence commands**

The reference cell must run:

```bash
python -m pytest tests/test_correctness.py -k 'gpu_v1 or torchvision' -q -rs
```

and reject output containing `skipped`. The benchmark cell must first invoke
`run_gpu_v1` once with 64 synthetic boxes, then run:

```bash
python benchmarks/run_all.py --versions cpu v1 --n 100 1000 10000 \
  --warmup 2 --repeats 7 --json "$EVIDENCE_DIR/benchmark_v1.json"
```

- [ ] **Step 4: Add evidence packaging**

Capture `git rev-parse HEAD`, `nvidia-smi`, `python --version` and
`pip show numba numpy` in `environment.txt`; zip `$EVIDENCE_DIR` and call
`google.colab.files.download` on the ZIP.

- [ ] **Step 5: Run the structural test**

Run: `./.venv/bin/python -m pytest tests/test_colab_notebooks.py -q`

Expected: V1 is valid; V2 test remains failing until Task 3.

### Task 3: Replace the obsolete V2 notebook

**Files:**
- Delete: `src/gpu_v2.ipynb`
- Create: `collab/v2_gpu_colab.ipynb`
- Test: `tests/test_colab_notebooks.py`

**Interfaces:**
- Consumes: public GitHub repository URL and `COMMIT` cell variable.
- Produces: `/content/evidence/v2/<commit>/` containing test output,
  `benchmark_v2.json`, `batch_v2_b32.json` and `environment.txt`.

- [ ] **Step 1: Reuse the same six-section flow as V1**

Keep the same setup, CUDA gate, commit capture and evidence ZIP behavior so
both notebooks can be operated identically in a seminar.

- [ ] **Step 2: Add V2 correctness and warm-up**

The reference cell must run:

```bash
python -m pytest tests/test_correctness.py -k 'gpu_v2 or torchvision' -q -rs
```

and reject output containing `skipped`. Warm-up must call V2 once with a
64-box synthetic input before any timed command.

- [ ] **Step 3: Add V2 single-image and B=32 benchmark commands**

Run:

```bash
python benchmarks/run_all.py --versions cpu v1 v2 --n 100 1000 10000 \
  --warmup 2 --repeats 7 --json "$EVIDENCE_DIR/benchmark_v2.json"
python benchmarks/run_v2_batch.py --batch-size 32 --n 10000 \
  --warmup 2 --repeats 7 --json "$EVIDENCE_DIR/batch_v2_b32.json"
```

The notebook must label the second report as NMS-only end-to-end V2 call:
host sort, transfer, mask kernel, mask download and CPU greedy resolution.

- [ ] **Step 4: Run structural validation**

Run: `./.venv/bin/python -m pytest tests/test_colab_notebooks.py -q`

Expected: PASS.

### Task 4: Point documentation to the replacement notebooks

**Files:**
- Modify: `docs/HOW_TO_RUN.md:7-26`
- Test: `tests/test_colab_notebooks.py`

**Interfaces:**
- Consumes: GitHub-hosted notebooks under `collab/`.
- Produces: two Colab links and accurate evidence instructions for V1/V2.

- [ ] **Step 1: Replace the V1/V2 Colab URLs**

Point to `/blob/main/collab/v1_gpu_colab.ipynb` and
`/blob/main/collab/v2_gpu_colab.ipynb`. Retain V3 documentation unchanged.

- [ ] **Step 2: Update the run instructions**

State that users choose a NVIDIA GPU runtime, set the `COMMIT` cell, Run all,
and download the evidence ZIP. State that any skipped CUDA test invalidates
the evidence run and that notebooks do not push results.

- [ ] **Step 3: Run all local validation**

Run: `./.venv/bin/python -m pytest tests -q`

Expected: all CPU tests pass; CUDA tests remain skipped on macOS.

- [ ] **Step 4: Validate notebook JSON**

Run:

```bash
jq empty collab/v1_gpu_colab.ipynb collab/v2_gpu_colab.ipynb
```

Expected: exit code 0.

### Task 5: Review and publish

**Files:**
- Modify: files from Tasks 1-4 only.

- [ ] **Step 1: Inspect scope**

Run: `git status --short` and confirm no V3 file and no user-modified
`docs/INDEX.md` are staged.

- [ ] **Step 2: Commit intentionally**

Run:

```bash
git add collab tests/test_colab_notebooks.py docs/HOW_TO_RUN.md src/gpu_v1.ipynb src/gpu_v2.ipynb
git commit -m "feat: add reproducible Colab runners for V1 and V2"
```

- [ ] **Step 3: Push only with working GitHub credentials**

Run: `git push origin main`

Expected: remote accepts the commit. If credentials are absent, stop and ask
the user to run `gh auth login -h github.com`; never store a token in a
notebook.
