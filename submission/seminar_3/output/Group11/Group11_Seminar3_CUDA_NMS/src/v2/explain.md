# GPU V2: SoA packed-mask NMS

V2 retains greedy hard-NMS results while reducing pairwise-storage cost. It is
the version designed for the catalog's batched NMS experiment.

| File | Responsibility |
|---|---|
| `kernels.py` | Coalesced structure-of-arrays IoU helper and CUDA construction of packed `uint64` suppression masks. |
| `core.py` | Validate/partition input, transfer data, launch masks, and perform the final host greedy-mask resolution. |
| `cli.py` | Single-image and batch synthetic benchmark commands. |

For a one-class batch, V2 sorts every image on the host and makes one CUDA
mask launch. For multi-class candidates it partitions by image and class, then
uses the same kernel for each partition. The resolver stays serial because
greedy keep decisions depend on earlier keep decisions; that is a known V2
limitation, not a hidden GPU-only claim.

`gpu_v2.py` is the compatibility façade for existing notebooks, tests, and CLI
commands. New implementation work belongs in this directory.
