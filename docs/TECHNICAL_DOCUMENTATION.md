# Technical documentation — CUDA NMS with Numba

## 1. Scope and current status

This is Topic A4 for CSC14116: accelerate the NMS post-processing stage of an
object detector. The repository has two deliberately separate experiments:

- **NMS-only synthetic stress** at `N = 100, 1,000, 10,000`.
- **Real detector integration**: raw YOLOv5 candidates, then this project's
  class-aware hard NMS.

The current local verification is **26 passed, 41 CUDA-skipped**. V1/V2 CUDA
parity and all new performance numbers must be rerun on a CUDA runtime before
they are cited. Existing files under `presentation/seminar_2/evidence/` are
historical pre-restructure evidence only.

## 2. Candidate contract

`src/common/candidates.py` is the shared source of truth. `src/nms_common.py`
is retained only as a compatibility façade for existing imports.

```text
boxes     float32 (N, 4), xyxy with x2 > x1 and y2 > y1
scores    float32 (N,)
class_ids int32   (N,)
```

`validate_candidates()` normalizes and checks the contract. The hard-NMS paths
partition candidates by class, apply greedy NMS inside each partition, then
return original indexes in stable descending-score order. This avoids
suppressing two overlapping boxes that belong to different detector classes.

`torchvision_class_aware_nms()` is the correctness oracle: it applies
`torchvision.ops.nms` independently for every class and restores the same
stable global order.

## 3. Implementations

| Module | Role | Main limitation |
|---|---|---|
| `baseline/{core,yolov5_adapter,cli}.py` (`cpu_baseline.py` façade) | Serial, class-aware greedy hard NMS; raw detector adapter; canonical baseline | O(N²) pairwise work |
| `v1/{kernel,core,cli}.py` (`gpu_v1.py` façade) | One CUDA thread per IoU pair; downloads full N×N matrix; CPU resolves greedy decisions | O(N²) device memory/transfer |
| `v2/{kernels,core,cli}.py` (`gpu_v2.py` façade) | SoA coordinates and packed `uint64` suppression masks; CPU resolves greedy decisions | Greedy dependency still remains on host |
| `gpu_v3.py` | Matrix NMS / soft score decay | Different semantics from hard NMS; intentionally untouched |

### CPU baseline and raw detector

`run_cpu(boxes, scores, class_ids=None, iou_threshold=0.5)` is the hard-NMS
reference. `load_synthetic_candidates()` supplies deterministic multi-class
stress inputs. `load_raw_yolo_candidates()` loads legacy `yolov5s.pt` through
the matching YOLOv5 `torch.hub` code, directly forwards a normalized tensor,
and decodes raw `xywh + objectness + class probability` predictions. It does
not use AutoShape or post-NMS `Results.boxes`.

### V1

`run_gpu_v1()` keeps the intentionally naive design. Each class partition is
score-sorted, copied to CUDA, processed by the full IoU-matrix kernel, copied
back, and resolved greedily on CPU. It demonstrates pairwise parallelism but
not an efficient production NMS design.

### V2

`run_gpu_v2()` keeps the coalesced SoA and packed-mask kernel. One-class batch
input uses its fused batch launch. Multi-class input is partitioned by
image/class and each partition uses the same kernel; returned indexes are
restored to the original image order. The final greedy mask resolver is still
serial CPU work.

### V3

V3 is Matrix NMS: it decays scores instead of reproducing greedy hard-NMS
keep indexes. Its correct reference is `matrix_nms_reference()`, not the
torchvision hard-NMS oracle. Do not claim V3 hard-NMS parity.

## 4. Tests and benchmarks

| Purpose | Command |
|---|---|
| Local correctness | `pytest tests -q` |
| CPU synthetic benchmark | `python benchmarks/run_all.py --versions cpu --n 100 1000 10000 --json /tmp/nms.json` |
| CUDA V1/V2 benchmark | `python benchmarks/run_all.py --versions cpu v1 v2 --n 100 1000 10000 --warmup 2 --repeats 7 --json benchmarks/results/t4.json` |
| V2 batch benchmark | `python benchmarks/run_v2_batch.py --batch-size 32 --n 10000 --warmup 2 --repeats 7 --json benchmarks/results/v2_batch32.json` |
| Detector + NMS report | `python benchmarks/run_detector_pipeline.py --image IMAGE --runner cpu --json benchmarks/results/detector.json` |

`run_all.py` and `run_v2_batch.py` report:

```text
benchmark_scope: nms_only_synthetic
candidate_source: deterministic_synthetic
timing_scope: candidate NMS only; excludes model inference
```

`run_detector_pipeline.py` reports `detector_plus_nms_real` and records raw
candidate-extraction time and NMS time independently. Do not compare those two
timings as though they measure the same scope.

## 5. Seminar-safe claims

- Hard-NMS correctness applies to CPU, V1 and V2 through the per-class
  torchvision oracle.
- V3 is a Matrix-NMS trade-off, not a hard-NMS replacement with identical
  indexes.
- The catalog target `32 × 10,000 boxes under 5 ms` remains unverified for the
  restructured code until a current T4 run exists.
- Use `presentation/seminar_2/README.md` as the single status page for the
  seminar; use evidence JSON/logs only for numbers actually measured.
