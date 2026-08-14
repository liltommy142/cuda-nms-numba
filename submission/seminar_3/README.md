# Seminar 3 submission — CUDA/Numba hard NMS

## Upload these files

Upload `FINAL_REPORT.ipynb`, `TEAM_PLAN.md`, and the GitHub repository/commit
history. For convenience, the final package also includes this README, the
fresh evidence logs, manifest, checksums, source, tests, and benchmark scripts.
No presentation deck is included because this is a submission-only hand-in.

## What was evaluated

The primary path is class-aware greedy hard non-maximum suppression (NMS) for
one image: NumPy CPU baseline, CUDA/Numba V1 dense pairwise IoU, and CUDA/Numba
V2 packed suppression masks. V3 Matrix NMS is separate and is not presented as
hard-NMS parity.

All reported timings are deterministic synthetic **NMS-only** measurements:
they exclude detector inference, image preprocessing, and model loading. The
evidence was collected from source commit
`7ee76cd5f6e12b87ddee247d58c9fd6ac866245b` on the recorded NVIDIA GeForce RTX
4060 Ti environment. See `evidence/environment.txt` for full provenance.

## Verified results

Seven repeated samples after two warm-ups; medians in milliseconds (lower is
better). Speedups are CPU median divided by the GPU median.

| Candidates/image | CPU | V1 | V2 | V1 speedup | V2 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.378 | 2.467 | 5.065 | 0.56× | 0.27× |
| 1,000 | 16.259 | 4.501 | 7.625 | 3.61× | 2.13× |
| 10,000 | 308.965 | 113.991 | 33.327 | 2.71× | 9.27× |

V2 batch-32 at 10,000 candidates/image measured 947.180 ms per batch, or
29.599 ms/image. Therefore the catalog stretch target of **<5 ms/image is
MISSED** on this environment. This result is not detector end-to-end latency.

## Five-command reproduction

Run from the repository root with an NVIDIA driver compatible with CUDA 13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda13.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe benchmarks\run_all.py --n 100 1000 10000 --warmup 2 --repeats 7 --json submission\seminar_3\evidence\benchmark_v1_v2.json
.\.venv\Scripts\python.exe benchmarks\run_batch_v2.py --batch-size 32 --n 10000 --warmup 2 --repeats 7 --json submission\seminar_3\evidence\batch32_v2.json
```

Execute `FINAL_REPORT.ipynb` in the same environment to run deterministic
CPU/V1/V2 parity and render the saved evidence. The optional YOLOv5 checkpoint
is absent from this repository; the submission makes no detector-inference
claim without that external asset.

## Evidence files

- `evidence/pytest_cuda.txt` — complete CUDA pytest output.
- `evidence/benchmark_v1_v2.json` and `.txt` — N=100/1,000/10,000 sweep.
- `evidence/batch32_v2.json` and `.txt` — V2 B=32, N=10,000 measurement.
- `evidence/environment.txt` — source commit, hardware, driver, Python, and
  installed package provenance.
