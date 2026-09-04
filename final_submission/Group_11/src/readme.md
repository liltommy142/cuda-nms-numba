# Source code map

The code is organized by responsibility and NMS version. Start with
[`common/explain.md`](common/explain.md) for the candidate contract, then read
the baseline before V1 and V2.

```text
common/    shared input contract and verification oracle
baseline/  serial CPU hard-NMS and raw YOLOv5 adapter
v1/        naive GPU full-IoU-matrix hard-NMS
v2/        SoA + packed-mask batched hard-NMS
gpu_v3.py  legacy Matrix NMS, intentionally unchanged
```

`cpu_baseline.py`, `gpu_v1.py`, and `gpu_v2.py` are compatibility façades.
They preserve the documented CLI and imports; new code should use the focused
modules in the corresponding directory.
