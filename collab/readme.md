# Google Colab runbook — Baseline, V1 and V2

Use this runbook on a fresh NVIDIA Colab runtime to generate evidence for the
exact commit being presented. It covers Baseline, V1 and V2 only; V3 is out of
scope.

## Required evidence

| Run | Command output to save |
|---|---|
| CUDA correctness | `pytest_cuda.txt` |
| CPU/V1/V2, N=100/1k/10k | `benchmark_v1_v2.json` and `.txt` |
| V2, B=32, N=10k | `batch32_v2.json` and `.txt` |
| Raw YOLO candidate integration | `detector_cpu.json` and `.txt` |

Single-image and batch reports measure **synthetic NMS only**, never complete
detector inference latency.

## 1. Start with a real CUDA runtime

In Colab choose **Runtime → Change runtime type → T4 GPU** (or another NVIDIA
GPU), then run:

```python
!nvidia-smi

import sys
from numba import cuda

print("Python:", sys.version)
assert cuda.is_available(), "Reconnect with an NVIDIA GPU runtime."
device = cuda.get_current_device()
name = device.name.decode() if isinstance(device.name, bytes) else device.name
print("GPU:", name, "compute capability:", device.compute_capability)
```

The project pins `numpy==1.26.4` and `numba==0.59.1`; Python 3.11 is the
supported target. If Colab cannot install that Numba version, stop and record
the Python version instead of benchmarking a different toolchain.

## 2. Clone an already-pushed commit

Set `COMMIT` to the commit you will present. Do not run uncommitted local code
or reuse historical T4 evidence.

```python
COMMIT = "3fbf757"  # replace before a final evidence run
!git clone https://github.com/liltommy142/cuda-nms-numba.git /content/cuda-nms-numba
%cd /content/cuda-nms-numba
!git checkout {COMMIT}
!git rev-parse HEAD

!python -m pip install -q "numpy==1.26.4" "numba==0.59.1" "pytest==9.1.1"
!python -m pip install -q -r requirements.txt
```

Restart the runtime if pip asks. After a restart repeat the CUDA check and
clone/checkout cells.

## 3. Capture environment metadata

```python
import os
from pathlib import Path
EVIDENCE_DIR = Path("/content/evidence") / COMMIT
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["EVIDENCE_DIR"] = str(EVIDENCE_DIR)
```

```bash
%%bash
set -euo pipefail
cd /content/cuda-nms-numba
git rev-parse HEAD | tee "$EVIDENCE_DIR/commit.txt"
nvidia-smi | tee "$EVIDENCE_DIR/nvidia-smi.txt"
python --version | tee "$EVIDENCE_DIR/python.txt"
python -m pip show numpy numba torch torchvision | tee "$EVIDENCE_DIR/packages.txt"
```

## 4. Required: full CUDA correctness

```bash
%%bash
set -o pipefail
cd /content/cuda-nms-numba
python -m pytest tests -q -rs 2>&1 | tee "$EVIDENCE_DIR/pytest_cuda.txt"
test ${PIPESTATUS[0]} -eq 0
```

Any failed or CUDA-skipped test invalidates the GPU evidence run. If a detector
test skips because `yolov5s.pt` is absent, do not claim the detector path is
verified until the weight is provided and the suite is rerun.

## 5. Required: V1/V2 single-image benchmark

```bash
%%bash
set -euo pipefail
cd /content/cuda-nms-numba
python benchmarks/run_all.py \
  --versions cpu v1 v2 --n 100 1000 10000 \
  --warmup 2 --repeats 7 --seed 0 \
  --json "$EVIDENCE_DIR/benchmark_v1_v2.json" \
  | tee "$EVIDENCE_DIR/benchmark_v1_v2.txt"
```

## 6. Required: V2 batch-size-32 benchmark

This includes host sort, transfer, GPU mask kernel, mask copy-back and CPU
greedy-mask resolution.

```bash
%%bash
set -euo pipefail
cd /content/cuda-nms-numba
python benchmarks/run_v2_batch.py \
  --batch-size 32 --n 10000 --warmup 2 --repeats 7 --seed 0 \
  --json "$EVIDENCE_DIR/batch32_v2.json" \
  | tee "$EVIDENCE_DIR/batch32_v2.txt"
```

## 7. Required for a detector-integration claim

Upload a trusted `yolov5s.pt` file to `/content/cuda-nms-numba/yolov5s.pt`,
then use a fixed public image URL or local image path:

```bash
%%bash
set -euo pipefail
cd /content/cuda-nms-numba
test -f yolov5s.pt
python benchmarks/run_detector_pipeline.py \
  --image "IMAGE_URL_OR_LOCAL_PATH" --runner cpu --repeats 3 --warmup 1 \
  --json "$EVIDENCE_DIR/detector_cpu.json" \
  | tee "$EVIDENCE_DIR/detector_cpu.txt"
```

The report times raw candidate extraction and NMS separately and checks NMS
against per-class `torchvision.ops.nms`.

## 8. Optional: CPU hotspot profile

`cProfile` explains the CPU bottleneck; it does not measure CUDA kernel speed.

```bash
%%bash
set -euo pipefail
cd /content/cuda-nms-numba
python -m cProfile -o "$EVIDENCE_DIR/cpu_n10000.prof" \
  src/cpu_baseline.py --source synthetic --n 10000
python -c "import pstats; pstats.Stats('$EVIDENCE_DIR/cpu_n10000.prof').strip_dirs().sort_stats('cumulative').print_stats('run_cpu|_run_single_class|iou_one_to_many')" \
  | tee "$EVIDENCE_DIR/cpu_n10000_profile.txt"
```

## 9. Download evidence

Inspect the test log, commit SHA and both benchmark JSON files. Then download
the ZIP and commit selected evidence under `presentation/seminar_2/evidence/`.

```python
from google.colab import files
import shutil

archive = shutil.make_archive(str(EVIDENCE_DIR), "zip", EVIDENCE_DIR)
files.download(archive)
```

Never store a GitHub token in Colab and never copy historical T4 numbers into a
new report.
