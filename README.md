# cuda-nms-numba

GPU-accelerated Non-Maximum Suppression using CUDA (Numba) — from naive parallel IoU to Matrix NMS.

CSC14116 — Applied Parallel Programming, Topic A4. Group 11: Lê Quang Tân (22127378), Phùng Quốc Tuấn (19127616).

## Usage

```bash
pip install -r requirements.txt
python src/cpu_baseline.py --benchmark
python src/gpu_v1.py --benchmark   # requires a CUDA GPU
pytest tests/
```

See `CSC14116 - Proposal.docx` for the full project proposal.
