# Local validation record

Date: 2026-08-06

## Environment

- Platform: macOS (no CUDA-capable NVIDIA GPU available)
- Python: 3.14.6
- CUDA/Numba GPU runtime: unavailable

## Commands run

```bash
python -m compileall -q src benchmarks
python -m pytest tests -q
```

## Result

- Source/benchmark compilation: passed.
- Result: **30 passed, 41 skipped, 0 failed**.
- Every GPU test was skipped because CUDA/Numba is unavailable locally.

This record proves only host-side import, syntax and CPU/reference-oracle
coverage. It is not GPU correctness or performance evidence. Follow
[`../SUBMISSION_CHECKLIST.md`](../SUBMISSION_CHECKLIST.md) on a CUDA machine
before presenting V2/V3 results.
