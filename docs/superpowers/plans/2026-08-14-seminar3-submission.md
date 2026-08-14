# Seminar 3 Submission Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a tested, reproducible, no-slides Seminar 3 hand-in containing a runnable notebook, current CUDA evidence, team plan, source snapshot, manifest, checksums, and upload-ready ZIP.

**Architecture:** Keep the existing class-aware CPU/V1/V2 implementations as the graded path. Add only the small public-helper guard required by the fresh CUDA suite, then generate evidence from an isolated Python 3.11/CUDA 13 environment. The submission notebook reads the committed evidence and independently runs a small CPU/V1/V2 parity check; packaging uses tracked files from one Git commit.

**Tech Stack:** Python 3.11, NumPy 1.26.4, pytest 9.1.1, numba-cuda 0.30.4 with CUDA 13, CPU-only PyTorch 2.5.1/torchvision 0.20.1, nbformat, nbclient, ipykernel, Git archive.

## Global Constraints

- No slide deck, speaker script, or presentation artifact.
- The primary claim covers class-aware hard NMS for CPU, V1, and V2 only.
- V3 Matrix NMS remains explicitly separate from hard-NMS parity and primary timing.
- Correctness must pass before timing; failed or missed targets are reported unchanged.
- Historical pre-restructure T4 files are never quoted as current evidence.
- Evidence records the exact tested source commit, GPU, driver, Python, and packages.
- No system-wide driver or CUDA toolkit changes.
- Missing optional YOLO weights are a documented limitation, not fabricated evidence.
- The final ZIP contains the report plus the exact source/tests/benchmarks needed to reproduce it.

---

## File Structure

- `requirements-cuda13.txt` — isolated CUDA 13 submission/test environment.
- `src/v1/kernel.py` — V1 full-IoU helper, including empty-input return.
- `src/v2/kernels.py` — V2 coalesced-IoU helper, including empty-input return.
- `tests/compat/test_public_facades.py` — public helper regression tests.
- `tests/test_submission_artifacts.py` — notebook execution, manifest, and checksum verification.
- `submission/seminar_3/FINAL_REPORT.ipynb` — runnable report and small parity demonstration.
- `submission/seminar_3/README.md` — grader-facing entry point and measured summary.
- `submission/seminar_3/TEAM_PLAN.md` — proposal-backed ownership and completion record.
- `submission/seminar_3/SUBMISSION_MANIFEST.txt` — scope, provenance, contents, limitations.
- `submission/seminar_3/SHA256SUMS.txt` — hashes for hand-in artifacts and evidence.
- `submission/seminar_3/evidence/*` — fresh raw logs, JSON measurements, environment metadata.
- `submission/seminar_3/output/Group11_Seminar3_CUDA_NMS.zip` — ignored upload artifact.

### Task 1: CUDA 13 environment contract and empty-helper correctness

**Files:**
- Create: `requirements-cuda13.txt`
- Modify: `tests/compat/test_public_facades.py`
- Modify: `src/v1/kernel.py`
- Modify: `src/v2/kernels.py`

**Interfaces:**
- Consumes: historic public facades `gpu_v1.compute_iou_matrix_gpu` and `gpu_v2.compute_iou_matrix_gpu_v2`.
- Produces: both helpers return an `np.float32` array with shape `(0, 0)` for a valid empty `(0, 4)` input without touching CUDA.

- [ ] **Step 1: Write the two failing public-behavior tests**

Append to `tests/compat/test_public_facades.py`:

```python
import numpy as np


def test_v1_public_iou_helper_returns_empty_matrix_without_cuda_launch():
    from gpu_v1 import compute_iou_matrix_gpu

    result = compute_iou_matrix_gpu(np.empty((0, 4), dtype=np.float32))
    assert result.shape == (0, 0)
    assert result.dtype == np.float32


def test_v2_public_iou_helper_returns_empty_matrix_without_cuda_launch():
    from gpu_v2 import compute_iou_matrix_gpu_v2

    result = compute_iou_matrix_gpu_v2(np.empty((0, 4), dtype=np.float32))
    assert result.shape == (0, 0)
    assert result.dtype == np.float32
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path src)
& .\.venv\Scripts\python.exe -m pytest tests\compat\test_public_facades.py -q
```

Expected: the two new tests fail because each helper attempts a zero-sized CUDA allocation/launch instead of returning early.

- [ ] **Step 3: Add the minimal empty-input guards**

At the start of each helper, normalize once and return before any `cuda.*` call:

```python
boxes = np.ascontiguousarray(boxes, dtype=np.float32)
if boxes.shape == (0, 4):
    return np.empty((0, 0), dtype=np.float32)
```

Keep every non-empty code path unchanged and remove the now-duplicate V2 normalization line.

- [ ] **Step 4: Add the reproducible CUDA 13 requirement set**

Create `requirements-cuda13.txt` with:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
numpy==1.26.4
pytest==9.1.1
numba-cuda[cu13]==0.30.4
torch==2.5.1+cpu
torchvision==0.20.1+cpu
nbformat==5.10.4
nbclient==0.10.2
ipykernel==6.30.1
```

- [ ] **Step 5: Verify GREEN and environment consistency**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-cuda13.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest tests\compat\test_public_facades.py -q
```

Expected: `pip check` reports no broken requirements and all facade tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add requirements-cuda13.txt src\v1\kernel.py src\v2\kernels.py tests\compat\test_public_facades.py
git commit -m "fix: make CUDA IoU helpers safe for empty inputs"
```

### Task 2: Fresh RTX CUDA correctness and benchmark evidence

**Files:**
- Create: `submission/seminar_3/evidence/pytest_cuda.txt`
- Create: `submission/seminar_3/evidence/benchmark_v1_v2.json`
- Create: `submission/seminar_3/evidence/benchmark_v1_v2.txt`
- Create: `submission/seminar_3/evidence/batch32_v2.json`
- Create: `submission/seminar_3/evidence/batch32_v2.txt`
- Create: `submission/seminar_3/evidence/environment.txt`

**Interfaces:**
- Consumes: Task 1 source and `.venv` dependency contract.
- Produces: commit-scoped correctness and timing evidence used verbatim by the notebook and README.

- [ ] **Step 1: Capture the tested source commit and CUDA environment**

Create the evidence directory, then generate `environment.txt` from actual commands:

```powershell
New-Item -ItemType Directory -Force submission\seminar_3\evidence | Out-Null
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$commit = git rev-parse HEAD
@(
  "source_commit=$commit"
  "captured_utc=$([DateTime]::UtcNow.ToString('o'))"
  (& nvidia-smi)
  (& $python --version)
  (& $python -m pip show numpy numba numba-cuda torch torchvision pytest)
) | Set-Content -Encoding utf8 submission\seminar_3\evidence\environment.txt
```

- [ ] **Step 2: Run a mandatory deterministic CUDA smoke test**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path src)
& .\.venv\Scripts\python.exe -c "import numpy as np; from numba import cuda; from cpu_baseline import run_cpu; from gpu_v1 import run_gpu_v1; from gpu_v2 import run_gpu_v2; b=np.array([[0,0,2,2],[0,0,2,2],[4,4,6,6]],dtype=np.float32); s=np.array([.9,.8,.7],dtype=np.float32); expected=run_cpu(b,s); assert cuda.is_available(); assert np.array_equal(run_gpu_v1(b,s),expected); assert np.array_equal(run_gpu_v2(b,s),expected); print('CUDA CPU/V1/V2 smoke: PASS')"
```

Expected: `CUDA CPU/V1/V2 smoke: PASS`. Stop Task 2 if it fails.

- [ ] **Step 3: Run the complete test suite and preserve the native exit code**

```powershell
$output = & .\.venv\Scripts\python.exe -m pytest tests -q -rs 2>&1
$exitCode = $LASTEXITCODE
$output | Tee-Object -FilePath submission\seminar_3\evidence\pytest_cuda.txt
if ($exitCode -ne 0) { exit $exitCode }
```

Expected: zero failed tests. CUDA V1/V2 tests must execute rather than skip.

- [ ] **Step 4: Run the repeated CPU/V1/V2 NMS-only sweep**

```powershell
$output = & .\.venv\Scripts\python.exe benchmarks\run_all.py --versions cpu v1 v2 --n 100 1000 10000 --warmup 2 --repeats 7 --seed 0 --json submission\seminar_3\evidence\benchmark_v1_v2.json 2>&1
$exitCode = $LASTEXITCODE
$output | Tee-Object -FilePath submission\seminar_3\evidence\benchmark_v1_v2.txt
if ($exitCode -ne 0) { exit $exitCode }
```

- [ ] **Step 5: Run the catalog batch workload**

```powershell
$output = & .\.venv\Scripts\python.exe benchmarks\run_v2_batch.py --batch-size 32 --n 10000 --warmup 2 --repeats 7 --seed 0 --json submission\seminar_3\evidence\batch32_v2.json 2>&1
$exitCode = $LASTEXITCODE
$output | Tee-Object -FilePath submission\seminar_3\evidence\batch32_v2.txt
if ($exitCode -ne 0) { exit $exitCode }
```

- [ ] **Step 6: Validate evidence structure and provenance**

Run:

```powershell
& .\.venv\Scripts\python.exe -c "import json,pathlib; p=pathlib.Path('submission/seminar_3/evidence'); a=json.loads((p/'benchmark_v1_v2.json').read_text()); b=json.loads((p/'batch32_v2.json').read_text()); assert set(a['configuration']['versions'])=={'cpu','v1','v2'}; assert set(a['results'])=={'100','1000','10000'}; assert b['configuration']['batch_size']==32 and b['configuration']['n']==10000; assert 'source_commit='+__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip() in (p/'environment.txt').read_text(encoding='utf-8-sig'); print('evidence validation: PASS')"
```

- [ ] **Step 7: Commit Task 2**

```powershell
git add submission\seminar_3\evidence
git commit -m "test: record Seminar 3 RTX CUDA evidence"
```

### Task 3: Runnable final report notebook and grader-facing documentation

**Files:**
- Create: `submission/seminar_3/FINAL_REPORT.ipynb`
- Create: `submission/seminar_3/README.md`
- Create: `submission/seminar_3/TEAM_PLAN.md`
- Create: `tests/test_submission_artifacts.py`

**Interfaces:**
- Consumes: Task 2 JSON/log evidence and CPU/V1/V2 public facades.
- Produces: a notebook that executes in the repository root using the current kernel and prints `GPU V1/V2 parity: PASS` plus evidence-derived metrics.

- [ ] **Step 1: Write the failing notebook execution test**

Create `tests/test_submission_artifacts.py`:

```python
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission" / "seminar_3"


def test_final_report_executes_against_submission_evidence():
    notebook = nbformat.read(SUBMISSION / "FINAL_REPORT.ipynb", as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    output = "\n".join(
        item.get("text", "")
        for cell in executed.cells
        for item in cell.get("outputs", [])
        if item.get("output_type") == "stream"
    )
    assert "GPU V1/V2 parity: PASS" in output
    assert "Evidence source commit:" in output
    assert "Batch-32 target status:" in output
```

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q
```

Expected: FAIL because `FINAL_REPORT.ipynb` does not exist.

- [ ] **Step 3: Create the notebook with the exact report flow**

Create a valid nbformat 4 notebook with these cells, in order:

1. Markdown title: `Seminar 3 - Real-Time Non-Maximum Suppression with CUDA/Numba` and Group 11 names/IDs.
2. Markdown scope: CPU/V1/V2 class-aware hard NMS; V3 separately labelled and excluded from primary comparison.
3. Code: find the ancestor containing `src/cpu_baseline.py`, prepend `src` to `sys.path`, and print the repository root.
4. Code: print Python, NumPy, Numba, CUDA availability, and GPU name; assert CUDA is available.
5. Markdown: explain the sequential greedy dependency and V1/V2 data flow.
6. Code: generate N=64 deterministic candidates, run CPU/V1/V2, assert exact index equality, and print `GPU V1/V2 parity: PASS`.
7. Code: load both Task 2 JSON files and `environment.txt`; print `Evidence source commit: <sha>`.
8. Code: print a compact N=100/1,000/10,000 median-ms and speedup table derived from JSON, then print batch median and `Batch-32 target status: MET` or `MISSED` against 5 ms.
9. Markdown: limitations - measured machine specificity, host greedy resolver, optional YOLO weight absent, V3 not hard-NMS parity.
10. Markdown: exact commands for installing `requirements-cuda13.txt`, running pytest, and reproducing both benchmarks.

Do not hard-code performance numbers in code cells; derive them from committed JSON.

- [ ] **Step 4: Create README and team plan from verified sources**

`README.md` must lead with the exact upload files, repeat measured values from Task 2 JSON, distinguish NMS-only from detector inference, state the target honestly, and give a five-command reproduction path.

`TEAM_PLAN.md` must use the proposal's recorded division of work:

- Phung Quoc Tuan (19127616): repository setup, CPU baseline, initial tests, proposal problem/background/risk work.
- Le Quang Tan (22127378): V1/V2 GPU implementation, GPU tests, proposal finalization.
- Shared: class-aware contract/restructure, evidence validation, report review, and mutual code explanation.

Describe shared work as team responsibility; do not invent hour counts or individual commits that Git history cannot prove.

- [ ] **Step 5: Verify GREEN and execute the notebook in place**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q
& .\.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; p='submission/seminar_3/FINAL_REPORT.ipynb'; n=nbformat.read(p,4); NotebookClient(n,timeout=300,kernel_name='python3',resources={'metadata':{'path':'.'}}).execute(); nbformat.write(n,p); print('executed notebook written')"
& .\.venv\Scripts\python.exe -c "import json; json.load(open('submission/seminar_3/FINAL_REPORT.ipynb',encoding='utf-8')); print('notebook JSON: PASS')"
```

- [ ] **Step 6: Commit Task 3**

```powershell
git add submission\seminar_3\FINAL_REPORT.ipynb submission\seminar_3\README.md submission\seminar_3\TEAM_PLAN.md tests\test_submission_artifacts.py
git commit -m "docs: add runnable Seminar 3 final report"
```

### Task 4: Manifest, checksums, and submission integrity test

**Files:**
- Modify: `tests/test_submission_artifacts.py`
- Create: `submission/seminar_3/SUBMISSION_MANIFEST.txt`
- Create: `submission/seminar_3/SHA256SUMS.txt`
- Modify: `README.md`
- Modify: `docs/INDEX.md`

**Interfaces:**
- Consumes: all Task 2 and Task 3 tracked artifacts.
- Produces: machine-verifiable hashes and discoverable repository entry points.

- [ ] **Step 1: Add the failing integrity tests**

Append:

```python
import hashlib


def test_submission_checksums_match_files():
    lines = (SUBMISSION / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        expected, relative = line.split("  ", 1)
        payload = (SUBMISSION / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected


def test_manifest_names_every_required_deliverable():
    manifest = (SUBMISSION / "SUBMISSION_MANIFEST.txt").read_text(encoding="utf-8")
    for name in (
        "FINAL_REPORT.ipynb",
        "README.md",
        "TEAM_PLAN.md",
        "evidence/pytest_cuda.txt",
        "evidence/benchmark_v1_v2.json",
        "evidence/batch32_v2.json",
        "evidence/environment.txt",
    ):
        assert name in manifest
```

- [ ] **Step 2: Run and verify RED**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q
```

Expected: checksum and manifest tests fail because the files do not exist.

- [ ] **Step 3: Write the manifest**

Create `SUBMISSION_MANIFEST.txt` with Group 11 identities, repository URL,
the `source_commit=` value copied from `evidence/environment.txt`, file list,
NMS-only scope, optional detector limitation, V3 separation, and measured
target status derived from Task 2 JSON. Do not call the ZIP commit its own hash.

- [ ] **Step 4: Generate stable SHA-256 entries**

Hash these paths relative to `submission/seminar_3/`, sorted lexically:

```text
FINAL_REPORT.ipynb
README.md
SUBMISSION_MANIFEST.txt
TEAM_PLAN.md
evidence/batch32_v2.json
evidence/batch32_v2.txt
evidence/benchmark_v1_v2.json
evidence/benchmark_v1_v2.txt
evidence/environment.txt
evidence/pytest_cuda.txt
```

Write one lowercase line per file as `<64-hex-digest><two spaces><relative path>`.
Do not include `SHA256SUMS.txt` in its own checksum list.

- [ ] **Step 5: Link the submission from repository documentation**

Add a `Seminar 3 submission` section to root `README.md` linking
`submission/seminar_3/README.md`, and add the same entry to `docs/INDEX.md`.
Do not add any slide link or copy numerical claims into `docs/INDEX.md`.

- [ ] **Step 6: Verify GREEN and check textual integrity**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_submission_artifacts.py -q
git diff --check
```

- [ ] **Step 7: Commit Task 4**

```powershell
git add README.md docs\INDEX.md submission\seminar_3\SUBMISSION_MANIFEST.txt submission\seminar_3\SHA256SUMS.txt tests\test_submission_artifacts.py
git commit -m "docs: finalize Seminar 3 submission manifest"
```

### Task 5: Upload ZIP and final verification

**Files:**
- Create (ignored): `submission/seminar_3/output/Group11_Seminar3_CUDA_NMS.zip`

**Interfaces:**
- Consumes: the clean Task 4 Git tree.
- Produces: the upload-ready archive and fresh final verification evidence.

- [ ] **Step 1: Run the full final verification from a clean tree**

```powershell
$env:PYTHONPATH = (Resolve-Path src)
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest tests -q -rs
git diff --check
git status --short
```

Expected: no broken packages, zero failed tests, no diff errors, and a clean tracked tree.

- [ ] **Step 2: Build the archive from tracked Git content**

```powershell
New-Item -ItemType Directory -Force submission\seminar_3\output | Out-Null
git archive --format=zip --prefix=Group11_Seminar3_CUDA_NMS/ --output=submission/seminar_3/output/Group11_Seminar3_CUDA_NMS.zip HEAD README.md requirements.txt requirements-cuda13.txt src benchmarks tests submission/seminar_3 docs/superpowers/specs/2026-08-14-seminar3-submission-design.md
```

- [ ] **Step 3: Inspect required ZIP entries and extract to a unique temp path**

```powershell
$zip = (Resolve-Path submission\seminar_3\output\Group11_Seminar3_CUDA_NMS.zip).Path
$entries = tar -tf $zip
$required = @(
  'Group11_Seminar3_CUDA_NMS/submission/seminar_3/FINAL_REPORT.ipynb',
  'Group11_Seminar3_CUDA_NMS/submission/seminar_3/SHA256SUMS.txt',
  'Group11_Seminar3_CUDA_NMS/src/gpu_v1.py',
  'Group11_Seminar3_CUDA_NMS/src/gpu_v2.py',
  'Group11_Seminar3_CUDA_NMS/tests/test_submission_artifacts.py'
)
foreach ($item in $required) { if ($item -notin $entries) { throw "missing ZIP entry: $item" } }
$extract = Join-Path $env:TEMP ("seminar3-" + [guid]::NewGuid())
Expand-Archive -LiteralPath $zip -DestinationPath $extract
Write-Output $extract
```

- [ ] **Step 4: Verify the extracted hand-in, including notebook execution**

From the printed extracted directory:

```powershell
$bundle = Join-Path $extract 'Group11_Seminar3_CUDA_NMS'
$env:PYTHONPATH = Join-Path $bundle 'src'
Push-Location $bundle
& 'D:\Study\HCMUS-APP\cuda-nms-numba\.worktrees\seminar3-submission\.venv\Scripts\python.exe' -m pytest tests\test_submission_artifacts.py -q
Pop-Location
```

Use only the isolated Python environment at the absolute path above.

- [ ] **Step 5: Record final hand-off facts**

Run and retain output for the final response:

```powershell
git log -5 --oneline
git status --short --branch
Get-Item submission\seminar_3\output\Group11_Seminar3_CUDA_NMS.zip | Select-Object FullName,Length,LastWriteTime
```

Expected: branch `codex/seminar3-submission`, clean tracked tree, and a non-empty ZIP.
