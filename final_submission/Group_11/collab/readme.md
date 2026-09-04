# Google Colab — GPU test runbook (T4, Baseline/V1/V2)

## Fast path: one notebook, Run all

Open [gpu_test_colab.ipynb](gpu_test_colab.ipynb) in Colab, choose
**Runtime → Change runtime type → T4 GPU**, then select **Run all**. It clones
or refreshes `main`, creates the verified CUDA 13 environment, runs the smoke
test, full pytest suite, V1/V2 benchmark sweep, and V2 B=32 benchmark, then
creates an evidence ZIP. The detector cell is intentionally skipped unless you
explicitly enable it and upload `yolov5s.pt`.

Use the terminal instructions below only for troubleshooting a failed cell.

This is the exact, terminal-first procedure that was verified on Google Colab
on 2026-08-07. It runs the project source at a chosen git commit, validates
CUDA JIT, runs the complete test suite, then records V1/V2 benchmark evidence.

Scope: Baseline, V1, V2 and the adaptive raw-YOLO candidate budget. Do not
change V3 for this run.

## Read this before running

- In Colab, select **Runtime → Change runtime type → T4 GPU**.
- Run commands either in the **Colab terminal** or in **notebook code cells**;
  the syntax differs:
  - Terminal: normal Bash only. Never paste `!`, `%cd`, or Python source into
    the terminal.
  - Code cell: use the `%%bash` cells below as-is.
- Do **not** run `pip install -r requirements.txt` in Colab's global Python.
  It pins legacy `numba==0.59.1`, which segfaults on the current CUDA 13
  Colab runtime, and its regular PyTorch wheel pulls CUDA 12 libraries that
  conflict with CUDA 13 linking.
- The working Colab environment is isolated at
  `/content/nms-cu13-venv`: `numba-cuda[cu13]`, NumPy 1.26.4,
  pytest 9.1.1, and CPU-only torch/torchvision for the reference oracle.
  NVIDIA's CUDA 13 installation route is `numba-cuda[cu13]`.

## Expected evidence

| Artifact | Meaning |
|---|---|
| `pytest_cuda.txt` | Complete test-suite result from the T4 runtime. |
| `benchmark_v1_v2.json` | CPU/V1/V2 synthetic NMS sweep at N=100, 1k, 10k. |
| `batch32_v2.json` | V2 end-to-end B=32, N=10k measurement. |
| `environment.txt` | Commit SHA, GPU, Python and package versions. |
| `detector_v2_cap11000.json` (optional) | Raw YOLO → 11k candidate budget → GPU V2 NMS. |

The first three benchmark artifacts measure **NMS only**. They do not measure
end-to-end object detector latency.

## A. Terminal workflow

Open **Tools → Terminal** in Colab and run the following blocks in order.

### A1. Check the GPU

~~~bash
nvidia-smi
~~~

The machine must show an NVIDIA GPU, normally Tesla T4.

### A2. Clone (or select) the exact source commit

Set `COMMIT` to the commit you intend to present. At the time this guide was
written, `c767fbd` contains the adaptive 11k-candidate feature.

~~~bash
COMMIT=c767fbd
REPO=/content/cuda-nms-numba

if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/liltommy142/cuda-nms-numba.git "$REPO"
fi

cd "$REPO"
git fetch origin
git checkout "$COMMIT"
git rev-parse HEAD
~~~

### A3. Create the CUDA 13 environment

~~~bash
python -m pip install -q virtualenv
python -m virtualenv --clear /content/nms-cu13-venv

/content/nms-cu13-venv/bin/python -m pip install --no-cache-dir \
  "numpy==1.26.4" "pytest==9.1.1" "numba-cuda[cu13]"

/content/nms-cu13-venv/bin/python -m pip install --no-cache-dir \
  "torch==2.5.1" "torchvision==0.20.1" \
  --index-url https://download.pytorch.org/whl/cpu
~~~

CPU-only PyTorch is deliberate: it provides `torchvision.ops.nms` for the
oracle without introducing a second CUDA toolkit into the Numba CUDA 13
environment.

### A4. CUDA smoke test — mandatory before pytest

~~~bash
cd /content/cuda-nms-numba
PYTHONPATH="$PWD/src" /content/nms-cu13-venv/bin/python - <<'PY'
import numpy as np
from numba import cuda
from gpu_v1 import compute_iou_matrix_gpu

print("CUDA module:", cuda.__file__)
print("CUDA available:", cuda.is_available())
assert cuda.is_available()

boxes = np.array([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=np.float32)
iou = compute_iou_matrix_gpu(boxes)
assert np.allclose(np.diag(iou), 1.0)
print("V1 CUDA JIT smoke test: PASS")
PY
~~~

A warning about low occupancy for this two-box smoke test is expected. Any
segmentation fault, `nvJitLinkError`, or failed assertion means stop; do not
record benchmark numbers.

### A5. Full test suite

~~~bash
set -o pipefail
COMMIT=$(git rev-parse --short HEAD)
EVIDENCE_DIR="/content/evidence/$COMMIT"
mkdir -p "$EVIDENCE_DIR"

PYTHONPATH="$PWD/src" /content/nms-cu13-venv/bin/python -m pytest tests -q -rs \
  2>&1 | tee "$EVIDENCE_DIR/pytest_cuda.txt"
test ${PIPESTATUS[0]} -eq 0
~~~

The verified run produced `78 passed, 1 skipped`. The single skip was the
live YOLO test because `yolov5s.pt` had not been uploaded; all CUDA tests ran.

### A6. Benchmark CPU, V1, V2

~~~bash
set -euo pipefail
PYTHONPATH="$PWD/src" /content/nms-cu13-venv/bin/python benchmarks/run_all.py \
  --versions cpu v1 v2 --n 100 1000 10000 \
  --warmup 2 --repeats 7 --seed 0 \
  --json "$EVIDENCE_DIR/benchmark_v1_v2.json" \
  | tee "$EVIDENCE_DIR/benchmark_v1_v2.txt"
~~~

### A7. Benchmark V2 batch size 32

This timing includes host sort, host→device copy, GPU bitmask kernel,
device→host copy, and CPU greedy-mask resolution.

~~~bash
set -euo pipefail
PYTHONPATH="$PWD/src" /content/nms-cu13-venv/bin/python benchmarks/run_v2_batch.py \
  --batch-size 32 --n 10000 --warmup 2 --repeats 7 --seed 0 \
  --json "$EVIDENCE_DIR/batch32_v2.json" \
  | tee "$EVIDENCE_DIR/batch32_v2.txt"
~~~

### A8. Save reproducibility metadata and list artifacts

~~~bash
{
  git rev-parse HEAD
  nvidia-smi
  /content/nms-cu13-venv/bin/python --version
  /content/nms-cu13-venv/bin/python -m pip show numba numba-cuda torch torchvision
} > "$EVIDENCE_DIR/environment.txt"

ls -lh "$EVIDENCE_DIR"
~~~

## B. Notebook code-cell workflow

Use this route if you want a clean notebook instead of the terminal. Insert
one **code cell** for each block below and execute from top to bottom.

### Cell 1 — runtime check

~~~python
!nvidia-smi
~~~

### Cell 2 — clone and checkout

~~~bash
%%bash
set -euo pipefail
COMMIT=c767fbd
REPO=/content/cuda-nms-numba

if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/liltommy142/cuda-nms-numba.git "$REPO"
fi

cd "$REPO"
git fetch origin
git checkout "$COMMIT"
git rev-parse HEAD
~~~

### Cell 3 — install the isolated CUDA 13 environment

~~~bash
%%bash
set -euo pipefail
python -m pip install -q virtualenv
python -m virtualenv --clear /content/nms-cu13-venv

/content/nms-cu13-venv/bin/python -m pip install --no-cache-dir \
  "numpy==1.26.4" "pytest==9.1.1" "numba-cuda[cu13]"

/content/nms-cu13-venv/bin/python -m pip install --no-cache-dir \
  "torch==2.5.1" "torchvision==0.20.1" \
  --index-url https://download.pytorch.org/whl/cpu
~~~

### Cell 4 — JIT smoke test

~~~bash
%%bash
set -euo pipefail
cd /content/cuda-nms-numba
PYTHONPATH="$PWD/src" /content/nms-cu13-venv/bin/python - <<'PY'
import numpy as np
from numba import cuda
from gpu_v1 import compute_iou_matrix_gpu

assert cuda.is_available()
boxes = np.array([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=np.float32)
assert np.allclose(np.diag(compute_iou_matrix_gpu(boxes)), 1.0)
print("V1 CUDA JIT smoke test: PASS")
PY
~~~

### Cell 5 — run tests and all required benchmarks

~~~bash
%%bash
set -euo pipefail
cd /content/cuda-nms-numba
COMMIT=$(git rev-parse --short HEAD)
EVIDENCE_DIR="/content/evidence/$COMMIT"
mkdir -p "$EVIDENCE_DIR"
export PYTHONPATH="$PWD/src"
PYTHON=/content/nms-cu13-venv/bin/python

"$PYTHON" -m pytest tests -q -rs 2>&1 | tee "$EVIDENCE_DIR/pytest_cuda.txt"

"$PYTHON" benchmarks/run_all.py \
  --versions cpu v1 v2 --n 100 1000 10000 \
  --warmup 2 --repeats 7 --seed 0 \
  --json "$EVIDENCE_DIR/benchmark_v1_v2.json" \
  | tee "$EVIDENCE_DIR/benchmark_v1_v2.txt"

"$PYTHON" benchmarks/run_v2_batch.py \
  --batch-size 32 --n 10000 --warmup 2 --repeats 7 --seed 0 \
  --json "$EVIDENCE_DIR/batch32_v2.json" \
  | tee "$EVIDENCE_DIR/batch32_v2.txt"

{
  git rev-parse HEAD
  nvidia-smi
  "$PYTHON" --version
  "$PYTHON" -m pip show numba numba-cuda torch torchvision
} > "$EVIDENCE_DIR/environment.txt"
~~~

### Cell 6 — download the evidence ZIP

~~~python
from pathlib import Path
from shutil import make_archive
from google.colab import files
import subprocess

commit = subprocess.check_output(
    ["git", "-C", "/content/cuda-nms-numba", "rev-parse", "--short", "HEAD"],
    text=True,
).strip()
evidence_dir = Path("/content/evidence") / commit
archive = make_archive(str(evidence_dir), "zip", evidence_dir)
files.download(archive)
~~~

## C. Optional detector + adaptive 11k candidate budget

This is separate from the NMS-only benchmarks. First upload trusted
`yolov5s.pt` to `/content/cuda-nms-numba/yolov5s.pt`. Then install the
detector-only dependencies into the same virtual environment, keeping NumPy,
Numba-CUDA, and CPU PyTorch already installed:

~~~bash
cd /content/cuda-nms-numba
/content/nms-cu13-venv/bin/python -m pip install --no-cache-dir \
  ultralytics pandas seaborn matplotlib "opencv-python<4.12" scipy gitpython tqdm thop
~~~

Run raw YOLO extraction with GPU V2 NMS and the adaptive budget:

~~~bash
set -euo pipefail
cd /content/cuda-nms-numba
test -f yolov5s.pt
COMMIT=$(git rev-parse --short HEAD)
EVIDENCE_DIR="/content/evidence/$COMMIT"
mkdir -p "$EVIDENCE_DIR"
PYTHONPATH="$PWD/src" /content/nms-cu13-venv/bin/python benchmarks/run_detector_pipeline.py \
  --image "https://ultralytics.com/images/zidane.jpg" \
  --runner v2 --conf-threshold 0.01 --max-candidates 11000 \
  --repeats 3 --warmup 1 \
  --json "$EVIDENCE_DIR/detector_v2_cap11000.json" \
  | tee "$EVIDENCE_DIR/detector_v2_cap11000.txt"
~~~

The JSON must show `raw_proposal_count`, `candidate_count <= 11000`,
`effective_conf_threshold`, `max_candidates: 11000`, and
`torchvision_parity: true`. Model candidate extraction is CPU-side in this
environment; only the selected NMS runner (`v2`) is GPU execution.

## Verified reference result

On the verified Tesla T4 run, commit `c767fbd`:

| N | CPU median | V1 median | V2 median |
|---:|---:|---:|---:|
| 100 | 3.050 ms | 2.697 ms | 5.572 ms |
| 1,000 | 33.741 ms | 6.069 ms | 8.168 ms |
| 10,000 | 790.201 ms | 83.917 ms | 32.021 ms |

V2 batch B=32, N=10,000: median batch time **1.116 s**; median per-image time
**34.876 ms**. Treat these as one T4 evidence run, not a universal performance
claim.
