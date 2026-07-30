# cuda-nms-numba

GPU-accelerated Non-Maximum Suppression using CUDA (Numba) — from naive parallel IoU to Matrix NMS.

CSC14116 — Applied Parallel Programming, Topic A4. Group 11: Lê Quang Tân (22127378), Phùng Quốc Tuấn (19127616).

## Usage

```bash
pip install -r requirements.txt
python src/cpu_baseline.py --benchmark
python src/gpu_v1.py --benchmark   # requires a CUDA GPU
pytest tests/

# Repeated measurements: median/stddev, raw samples, machine/GPU metadata
python benchmarks/run_all.py --versions cpu v1 v2 v3 --repeats 7 \
    --json benchmarks/results/measurement.json
```

The CUDA implementations currently process one image / one set of boxes per
call. Fused batch-size-32 execution from the A4 catalog is not implemented.

See `CSC14116 - Proposal.docx` for the full project proposal.

## Documentation

🧭 Start at [`docs/INDEX.md`](docs/INDEX.md) — a map of every doc in this repo, organized by what you're trying to do (understand the project, run it yourself, prep the talk, look up a term). Individually: [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md) (run code/tests without AI help), [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) (full technical writeup), [`docs/GLOSSARY.md`](docs/GLOSSARY.md) (term lookup).
