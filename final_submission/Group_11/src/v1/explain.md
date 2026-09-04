# GPU V1: full IoU matrix

V1 demonstrates direct pairwise parallelism, not a production-optimal NMS
design. It keeps hard-NMS semantics identical to the CPU baseline.

| File | Responsibility |
|---|---|
| `kernel.py` | Launch one CUDA thread per box pair and return a full `N × N` IoU matrix. |
| `core.py` | Partition classes, call the kernel, and resolve greedy suppression on the CPU. |
| `cli.py` | V1 CLI and synthetic CPU-vs-V1 benchmark. |

For each score-sorted class, V1 uploads boxes, computes all pair IoUs on CUDA,
downloads the dense matrix, and applies the dependency-sensitive greedy loop on
the host. Its main limitation is quadratic device memory and transfer cost.

`gpu_v1.py` is a compatibility façade. It remains the command users run; the
three files above are the implementation to read or modify.
