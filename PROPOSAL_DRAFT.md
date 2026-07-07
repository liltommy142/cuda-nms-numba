# Proposal Draft — A4: GPU-Accelerated Non-Maximum Suppression (NMS)

> Draft content ready to copy into `CSC14116 - Proposal.docx`. Placeholders marked `[TODO]`.
> Confirmed with instructor: submission format = **both** `.py` and `.ipynb`; template = `Proposal_template.docx`; team size = 2; deadline = **July 10, 2026**.

---

## Project Name
GPU-Accelerated Non-Maximum Suppression for Real-Time Object Detection

## Topic / Track
Catalog Topic A4 — Real-Time Non-Maximum Suppression for Object Detection (Track A — Computer Vision)

## Git Repository URL
<https://github.com/liltommy142/cuda-nms-numba>

## Group name
[TODO]

## List of members
- [TODO — Full name, Student ID]
- [TODO — Full name, Student ID]

## Keywords
CUDA, Numba, Non-Maximum Suppression, Object Detection, IoU Parallelization

## List of references
- Bodla, N. et al. (2017). *Soft-NMS — Improving Object Detection With One Line of Code*. ICCV 2017.
- Wang, X. et al. (2020). *SOLOv2: Dynamic and Fast Instance Segmentation*. NeurIPS 2020. (Matrix NMS, Section 3.3)
- Hosang, J. et al. (2017). *Learning Non-Maximum Suppression*. CVPR 2017.
- `torchvision.ops.nms` / `box_iou` source (reference for correctness verification only, not copied)
- NVIDIA TensorRT NMS plugin (reference only, not copied)

---

## Content

### Problem Statement

**Problem:**
Modern object detectors (YOLO, SSD, Faster R-CNN) output thousands of candidate bounding boxes per image. Non-Maximum Suppression (NMS) filters these down to the final detections by suppressing boxes that overlap heavily with higher-confidence boxes. On CPU, greedy NMS is a serial O(n²) operation and becomes the throughput bottleneck at inference time, especially at large batch sizes or with images containing many objects. Computing the IoU between every pair of boxes is entirely independent across pairs, making this problem well-suited to GPU parallelism.

**Dataset / Input:**
- Dataset name and source: synthetic bounding boxes generated with `numpy.random`, plus real detection outputs from a pretrained **YOLOv5s** model run on a subset of COCO validation images.
- Input size for benchmarking: N ∈ {100, 1,000, 10,000} boxes, batch size 32.
- How to load: `torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)` for real boxes; `numpy.random.uniform` for synthetic stress-test boxes.

**Why GPU-suitable:**
The central operation is computing IoU (Intersection over Union) between every pair (box_i, box_j) — this is fully independent across pairs and can be assigned one GPU thread per pair (N² threads for N boxes). This is a textbook "embarrassingly parallel" problem, analogous to computing pairwise distances between points.

---

### Background

Pipeline / pseudocode:
```
Input: boxes[N], scores[N], IoU_threshold
1. Sort boxes by score, descending
2. For each box (in descending score order):
   - If box is not yet suppressed:
       - Keep this box (final detection)
       - For every remaining box with lower score:
           Compute IoU(current_box, other_box)
           If IoU > threshold: mark other_box as suppressed
Output: list of kept boxes
```

The part that parallelizes well: computing the N×N IoU matrix — every pair is independent and can be computed simultaneously on GPU (one thread per pair). The part that resists naive parallelization: the sequential suppression decision (whether box B survives depends on whether box A, a higher-scoring box, was already kept) — this is the challenge addressed below.

**Measured CPU baseline timing** (`python src/cpu_baseline.py --benchmark`, `run_cpu`, synthetic boxes):

```text
       N |   time (s)
---------------------
     100 |     0.0008
    1000 |     0.0103
   10000 |     0.2846
```

Runtime grows in line with the algorithm's O(n²) worst case as N increases.

**Profiling** (`cProfile`, N=10,000, see `profile_output/cprofile_N10000.txt`): total runtime 0.289s, with ~65% (`0.189s`) spent directly inside the `run_cpu` suppression loop and ~34% (`0.098s`) inside `iou_one_to_many` — together over 99% of runtime is the NMS step itself, confirming it as the bottleneck to target with GPU kernels.

---

### The Challenge

The core difficulty is not computing IoU (embarrassingly parallel) but the serial suppression step: whether to keep or discard box B depends on whether box A (higher score) has already been kept — a data-dependency chain that cannot be naively parallelized. To resolve this, the team will implement **Matrix NMS** (Wang et al., 2020), which replaces hard sequential suppression with a fully parallel soft-suppression computation based on a decay factor rather than hard elimination. This is also the primary learning goal: transforming an algorithm with sequential dependencies into a formulation that can be computed entirely in parallel.

---

### Optimization Plan

| Version | Technique | Expected speedup vs. CPU baseline (N=10,000) |
|---|---|---|
| CPU baseline | Serial greedy NMS, pure NumPy (`run_cpu` in `src/cpu_baseline.py`) | 1× (reference) |
| GPU V1 | Naive parallel IoU matrix kernel — 1 thread per (box_i, box_j) pair; suppression mask still resolved sequentially on host | ~5–10× (IoU computation parallelized, suppression loop still the bottleneck) |
| GPU V2 | GPU V1 + batched NMS using parallel reduction to build the suppression mask + coalesced box coordinate reads | ≥15× (100% target) |
| GPU V3 (stretch) | Matrix NMS (Wang et al., 2020) — replaces sequential suppression with a fully parallel soft-suppression pass based on a decay factor | 30–80× (125% target), <5ms at N=10,000/batch 32 |

Each version is verified against `torchvision.ops.nms` before moving to the next (see `tests/test_correctness.py`), so a regression in V2/V3 is caught immediately rather than at the end.

---

### Resources

- **Starting code base**: from scratch for the CPU baseline (NumPy); `torchvision.ops.nms` and `torchvision.ops.box_iou` used only as a reference to verify correctness (not copied).
- **Pretrained model**: YOLOv5s (`torch.hub`) for realistic box distributions.
- **Reference papers**: Bodla et al. 2017 (Soft-NMS); Wang et al. 2020 (Matrix NMS, Section 3.3) — used for GPU V3 design.
- **Compute**: Google Colab / Kaggle Notebook (free GPU).
- **GPU language/library**: Numba (`@cuda.jit`) — the course's official GPU tool (no raw CUDA C/C++).
- **Open item**: GPU profiling on Colab via `nvprof` may be restricted by permissions — needs early testing; `torch.cuda.Event` timing or Nsight tools may be a fallback.

---

### Goals and Deliverables

**Performance Target: ≥15× speedup at 100% level, 30–80× at 125% stretch level, measured at N=10,000 boxes, batch size 32.**

**100% (must achieve):**
- CPU baseline correct, verified to match `torchvision.ops.nms` (tolerance 1e-4)
- GPU V1: parallel IoU matrix kernel (1 thread per pair) — correct, not yet optimized for speed
- GPU V2: batched NMS with parallel reduction for the suppression mask, coalesced memory access
- At least **15× speedup** over CPU NMS baseline at N=10,000 boxes (conservative target, below the catalog's official 30–80×, to ensure achievability)
- *Justification*: since the IoU step is embarrassingly parallel, simply moving from a Python loop to a GPU kernel computing thousands of pairs simultaneously already yields substantial speedup at large N.

**125% (stretch, if ahead of schedule):**
- GPU V3: full Matrix NMS (Wang et al. 2020), reaching the catalog's official target of **30–80× speedup**, processing 10,000 boxes/batch 32 in under 5ms
- Additional comparison of Soft-NMS vs. Matrix NMS speed/accuracy trade-off

**75% (if behind schedule):**
- Only GPU V1 (naive IoU kernel) completed, with correctness + baseline speedup measured, memory/compute not yet optimized
- Sufficient data collected to analyze why the target wasn't reached and what would improve it

**Demo at seminar:**
- Latency comparison chart: CPU vs. GPU V1 vs. V2 (vs. V3 if completed) across N ∈ {100, 1,000, 10,000}
- Visual demo: image with many overlapping boxes → after NMS only correct boxes remain, running in real time if the target is met
- Table comparing final detection results between CPU and GPU (proving correctness)

---

### Risk Analysis

1. **GPU profiling on Colab may be permission-restricted.** Colab's shared VMs frequently block `nvprof` / Nsight Compute due to non-root access, which would prevent kernel-level profiling (occupancy, memory throughput) needed to justify GPU V2/V3 design decisions.
   *Mitigation*: fall back to wall-clock timing via `torch.cuda.Event(enable_timing=True)` / `numba.cuda.event`, which works without elevated privileges; test `nvprof` access in Week 1 (not at the profiling deadline) so there is time to switch approach if it fails.

2. **Floating-point mismatch between the hand-written Numba kernel and `torchvision.ops.nms`.** IoU computed in a CUDA kernel (single precision, different summation order than PyTorch's C++/CUDA backend) can disagree with the reference by a few ULPs, and — more importantly — ties in box scores can flip the greedy suppression order (since `torchvision.ops.nms` sorts stably while a naive kernel/argsort may not), producing a different *set* of kept boxes even when every individual IoU value matches within tolerance.
   *Mitigation*: verify IoU values within a numerical tolerance (e.g. 1e-4) rather than requiring bit-exact floats; use a stable sort (`kind='stable'` in NumPy, an explicit stable sort on GPU) so tie-breaking matches the reference; report the kept-box-set match rate in addition to IoU error, and treat near-ties (score difference below a small epsilon) as an expected, documented source of disagreement rather than a bug.

3. **GPU V3 (Matrix NMS) may not be finished in time.** It is the hardest deliverable — replacing the sequential suppression dependency with a fully parallel soft-suppression formulation (Wang et al., 2020) — and is scheduled last, so any slippage in V1/V2 directly eats into V3's time budget.
   *Mitigation*: V3 is explicitly scoped as the 125% stretch goal, not a 100% requirement — the 75%/100% goals (V1, V2) are achievable without it; start reading the Matrix NMS paper (Section 3.3) in parallel with V1/V2 implementation instead of after, so design work isn't blocked by leftover schedule; if V3 is not completed, the 75% fallback plan (V1 only, with correctness + baseline speedup data and a written analysis of the gap) is already accounted for in Goals and Deliverables above.

---

### Division of Work

- **Person A** — owns the code path: CPU baseline, profiling, all three GPU kernel versions (V1 naive IoU, V2 batched + memory optimization, V3 Matrix NMS), and the correctness/benchmark test suite (`tests/test_correctness.py`).
- **Person B** — owns the writing/verification path: proposal document, weekly progress report, independently re-running `tests/test_correctness.py` and `benchmark()` against each GPU version Person A delivers (a second pair of eyes on correctness before it's marked done), report and slide deck for the final seminar.
- Both members are expected to be able to explain every part of the code, per the course's participation requirement — work is split for pace, not to silo knowledge.

---

### Weekly schedule
*(Confirm exact week numbers with instructor — course slides show two numbering schemes: Week 6–13 presentations vs. Week 5–10 "Level ladder.")*

| Week | Person A | Person B |
|---|---|---|
| 5–6 | CPU baseline + profiling | Write proposal + Git setup |
| 7 | GPU V1 (naive IoU kernel) | Verify V1 vs. torchvision |
| 8 | GPU V2 (memory optimization) | GPU V3 (Matrix NMS) |
| 9–10 | Combined benchmarking | Report + slides + demo |

---

## ⏰ Revised sprint plan — deadline July 10, 2026 (today = July 8)

`plan.md`'s original sprint assumed ~4 days remaining; only **2 days** remain as of today. Compressed plan:

### Today — July 8
**Person A (code)**
- [x] Repo restructured to match the Catalog's required layout — `src/cpu_baseline.py` (+ `.ipynb`), `tests/test_correctness.py`, `requirements.txt`, `README.md` at repo root
- [x] Write `cpu_baseline.py` **and** a parallel `.ipynb` version (both formats confirmed) — done, verified with `--n 500` and `--benchmark` (100/1,000/10,000 boxes)
- [x] Run `cProfile`, capture output showing NMS as the bottleneck — see `profile_output/cprofile_N10000.txt` and Background section above
- [x] Write `tests/test_correctness.py` — 5 passed / 3 skipped locally (torchvision tests skip without torch installed; run again on Colab to execute them)
- [ ] Confirm Git repo pushed (repo currently local only — not yet pushed)
- [ ] Run `yolov5s` on a few COCO images to get real sample boxes (`load_real_boxes` in `cpu_baseline.py`/`.ipynb`) — needs a torch/torchvision environment (e.g. Colab)

**Person B (writing)**
- [x] Draft full proposal content (Problem Statement, Background, Challenge, Optimization Plan, Resources, Goals and Deliverables, Risk Analysis, Division of Work, Weekly schedule) — see `PROPOSAL_DRAFT.md`
- [ ] Copy the content above into `Proposal_template.docx` (or confirm with instructor whether `project_proposal_template.md` from the Catalog is the actual required format — see open question below)

### July 9 (final full day)
**Person A (code)**
- [ ] Push code + README to Git
- [ ] Run `tests/test_correctness.py` on Colab/Kaggle with torch installed to actually execute the torchvision comparison tests (currently only skipped, never run)
- [ ] Confirm CPU baseline runs on a machine without GPU (mandatory per catalog — 0 points if it fails)

**Person B (writing)**
- [ ] Fill in Weekly schedule with real team names/dates
- [ ] Fill in Group name + member names/IDs (currently placeholders above)
- [ ] Ask instructor: `.docx` template vs. Catalog's `project_proposal_template.md`, and proposal deadline (Catalog itself says both "end of Week 6" and "before Week 5" in different places)
- [ ] Finalize full proposal document, spell-check, export PDF/Word

### July 10 — deadline day
- [ ] Submit early, several hours before the deadline
- [ ] Double-check Git repo is public/accessible to the instructor
- [ ] Confirm submission succeeded on Moodle
