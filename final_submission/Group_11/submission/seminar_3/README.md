# Seminar 3 submission — CUDA/Numba hard NMS

## Start here

The main thing to open is `FINAL_REPORT.ipynb`. The notebook, source code, tests,
benchmark scripts, and the evidence folder are kept together so the result is
easy to check. `TEAM_PLAN.md` records what each member actually handled. The
presentation deck is not inside this hand-in because this ZIP is the code/report
package for Seminar 3.

## What we implemented

We implemented class-aware greedy hard non-maximum suppression (NMS) for one
image in three versions: a NumPy CPU baseline, CUDA/Numba V1 with dense pairwise
IoU, and CUDA/Numba V2 with packed suppression masks. V3 Matrix NMS is kept as a
separate experiment; we do not call it hard-NMS parity.

The timings below use deterministic synthetic boxes and measure NMS only. They
do not include detector inference, image preprocessing, or model loading. The
GPU evidence was collected from source commit
`7ee76cd5f6e12b87ddee247d58c9fd6ac866245b` on the recorded NVIDIA GeForce RTX
4060 Ti environment. See `evidence/environment.txt` for full provenance.

## Verified results

We ran two warm-ups and then seven samples. The table shows medians in
milliseconds (lower is better); speedup is CPU median divided by GPU median.

| Candidates/image | CPU | V1 | V2 | V1 speedup | V2 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.378 | 2.467 | 5.065 | 0.56× | 0.27× |
| 1,000 | 16.259 | 4.501 | 7.625 | 3.61× | 2.13× |
| 10,000 | 308.965 | 113.991 | 33.327 | 2.71× | 9.27× |

For V2 batch-32 with 10,000 candidates per image, the measured median was
947.180 ms for the whole batch (29.599 ms/image). So the catalog stretch target
of **<5 ms/batch is MISSED** on this machine. We leave that result visible
instead of presenting the per-image number as if it were a batch result. This is
still NMS-only timing, not end-to-end detector latency.

## Five-command reproduction

From the repository root, with an NVIDIA driver compatible with CUDA 13, the
same run can be reproduced with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda13.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe benchmarks\run_all.py --n 100 1000 10000 --versions cpu v1 v2 --warmup 2 --repeats 7 --seed 0 --json submission\seminar_3\evidence\benchmark_v1_v2.json
.\.venv\Scripts\python.exe benchmarks\run_v2_batch.py --batch-size 32 --n 10000 --warmup 2 --repeats 7 --seed 0 --json submission\seminar_3\evidence\batch32_v2.json
```

Running `FINAL_REPORT.ipynb` in that environment checks CPU/V1/V2 parity and
shows the saved evidence. The optional YOLOv5 checkpoint is not included, so we
do not make a detector-inference claim.

## Evidence files

- `evidence/pytest_cuda.txt` — complete CUDA pytest output.
- `evidence/benchmark_v1_v2.json` and `.txt` — N=100/1,000/10,000 sweep.
- `evidence/batch32_v2.json` and `.txt` — V2 B=32, N=10,000 measurement.
- `evidence/environment.txt` — source commit, hardware, driver, Python, and
  installed package provenance.
