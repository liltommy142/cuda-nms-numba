# cuda-nms-numba

GPU-accelerated Non-Maximum Suppression using CUDA (Numba) — from naive parallel IoU to Matrix NMS.

CSC14116 — Applied Parallel Programming, Topic A4. Group 11: Lê Quang Tân (22127378), Phùng Quốc Tuấn (19127616).

## Usage

```bash
pip install -r requirements.txt
python src/cpu_baseline.py --source synthetic --benchmark --verify
python src/gpu_v1.py --benchmark   # requires a CUDA GPU
pytest tests/

# Repeated measurements: median/stddev, raw samples, machine/GPU metadata
python benchmarks/run_all.py --versions cpu v1 v2 v3 --repeats 7 \
    --json benchmarks/results/measurement.json

# V2 end-to-end batch-32 measurement (one CUDA mask launch for 32 images)
python benchmarks/run_v2_batch.py --batch-size 32 --n 10000 --warmup 2 --repeats 7 \
    --json benchmarks/results/v2_batch32.json
```

Baseline, V1 and V2 share a class-aware hard-NMS contract: `boxes (N,4)`,
`scores (N,)`, `class_ids (N,)`. V2 retains its fused one-launch batch path
when every candidate belongs to one class; real multi-class candidates are
partitioned by class and each partition uses the same SoA packed-mask kernel.
The final greedy mask resolution remains on the CPU. V3 is an unchanged Matrix
NMS experiment and is not a hard-NMS parity implementation.

`benchmarks/run_all.py` and `run_v2_batch.py` measure **synthetic NMS only**.
`benchmarks/run_detector_pipeline.py` measures raw YOLO candidate extraction
and NMS separately. Historical T4 values under `presentation/seminar_2/evidence/`
are pre-restructure until the current commit is rerun on a CUDA runtime.

See `CSC14116 - Proposal.docx` for the full project proposal.

## Documentation

🧭 Start at [`docs/INDEX.md`](docs/INDEX.md) — a map of every doc in this repo, organized by what you're trying to do (understand the project, run it yourself, prep the talk, look up a term). Individually: [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md) (run code/tests without AI help), [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) (full technical writeup), [`docs/GLOSSARY.md`](docs/GLOSSARY.md) (term lookup).
