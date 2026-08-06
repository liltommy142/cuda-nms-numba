# NMS Source Restructure Design

**Status:** Approved by project owner on 2026-08-07.

## Goal

Rewrite and reorganize the baseline, V1, and V2 implementations so each
version is readable in isolation while preserving all current hard-NMS
behaviour, public Python APIs, CLI commands, benchmark semantics, and notebook
entry points. `src/gpu_v3.py` is explicitly out of scope and must remain
byte-for-byte unchanged.

## Scope and non-goals

- In scope: `common`, baseline, V1, V2 source modules; their tests; and small
  documentation updates needed to point to the retained entry points.
- In scope: compatibility façades at `src/cpu_baseline.py`, `src/gpu_v1.py`,
  and `src/gpu_v2.py` so existing imports and `python src/<file>.py` commands
  continue to work.
- Out of scope: a new NMS algorithm, changed benchmark targets, dependency
  changes, a packaging/install migration, and any production-code change to
  V3.

## Target layout

```text
src/
  common/
    candidates.py       canonical candidate contract, validation, ordering, synthetic data
    oracle.py           torchvision class-aware reference NMS
  baseline/
    core.py             CPU IoU and serial greedy hard NMS
    yolov5_adapter.py   raw YOLOv5 output to canonical candidates
    cli.py              baseline argument parsing and presentation
  v1/
    kernel.py           full NxN CUDA IoU kernel
    core.py             V1 class-aware orchestration and CPU resolver
    cli.py              V1 argument parsing and presentation
  v2/
    kernels.py          SoA IoU and packed-bitmask CUDA kernels
    core.py             V2 routing, transfers, and greedy mask resolution
    cli.py              V2 argument parsing and presentation
  cpu_baseline.py       compatibility façade
  gpu_v1.py             compatibility façade
  gpu_v2.py             compatibility façade
  gpu_v3.py             untouched legacy V3
```

Each package module owns one responsibility. CUDA device functions remain with
the version that executes them; shared data semantics do not depend on a GPU
version.

## Interfaces and compatibility

The façades re-export the functions consumed by tests, benchmarks, notebooks,
and V3. In particular, `run_cpu`, `run_gpu_v1`, `run_gpu_v2`, IoU helper
functions, benchmark functions, YOLO raw-candidate helpers, and legacy
`load_data` remain importable under their current module names. `load_data`
remains only as an explicitly documented one-class compatibility adapter for
V3; new baseline/V1/V2 code uses the canonical three-array candidate contract.

CLI behaviour stays at the existing command paths and keeps synthetic NMS-only
timing separate from real raw-detector candidate extraction. No GPU benchmark
claim is created or changed by this refactor.

## Test and verification strategy

1. Add façade regression tests before moving implementation code; these test
   the preserved public imports and representative calls.
2. Move tests into `common`, `baseline`, `v1`, and `v2` suites only after the
   relevant module is green. V3 tests may be relocated without changing their
   assertions or production module.
3. Run the full test suite after each version migration. CUDA tests may skip on
   a host without NVIDIA CUDA; CPU and import-level tests must pass.
4. Validate each CLI help command and JSON notebook syntax.
5. Before commit, run `git diff --exit-code` against `src/gpu_v3.py` and its
   notebook so the protected V3 path is demonstrably unchanged.

## Error handling and invariants

Validation remains centralized: boxes are finite `float32 (N, 4)` `xyxy`,
scores are finite `float32 (N,)`, class IDs are `int32 (N,)`, and valid box
area is positive. NMS output remains original indices in stable descending
score order, with suppression only inside a class. CUDA-unavailable paths keep
their current clear failure/skip behaviour.

## Self-review

This design has no placeholders, preserves the user-selected V3 boundary, and
does not introduce packaging or performance scope beyond the requested source
clarity. The module layout maps every existing responsibility to exactly one
owner while the façades limit compatibility churn.
