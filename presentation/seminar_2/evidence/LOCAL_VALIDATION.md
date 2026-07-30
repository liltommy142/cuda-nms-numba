# Local validation record

Date: 2026-07-31

## Environment

- Platform: macOS (no CUDA-capable NVIDIA GPU available)
- Python: 3.14.6
- CUDA/Numba GPU runtime: unavailable

## Commands run

```bash
python3 -m py_compile src/gpu_v2.py tests/test_correctness.py
python3 -m pytest tests -v
```

## Result

- Python syntax check: passed.
- Test collection: 49 tests.
- Result: **9 passed, 40 skipped, 0 failed**.
- Every GPU test was skipped because CUDA/Numba is unavailable locally.

This record proves only host-side import, syntax and CPU/reference-oracle
coverage. It is not GPU correctness or performance evidence. Follow
[`../SUBMISSION_CHECKLIST.md`](../SUBMISSION_CHECKLIST.md) on a CUDA machine
before presenting V2/V3 results.
