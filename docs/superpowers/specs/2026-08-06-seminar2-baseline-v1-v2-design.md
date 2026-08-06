# Seminar 2 baseline, V1 and V2 restructuring

## Goal

Make the project demonstrably match Track A4 for Seminar 2 while preserving
`src/gpu_v3.py` and its Matrix-NMS tests and evidence unchanged.  The revised
project must show both a realistic detector-to-NMS path and controlled
synthetic stress tests.

## Scope

### Preserve unchanged

- `src/gpu_v3.py`, `src/gpu_v3.ipynb`, and V3-specific tests.
- Existing Tesla T4 evidence, which remains historical evidence for the
  pre-restructure implementation unless rerun after the new work.
- The distinction that V3 uses Matrix NMS and is not greedy-NMS parity.

### Rebuild

- CPU baseline and shared input contract.
- V1 and V2 implementation boundaries, tests, notebooks and benchmarks.
- Seminar 2 documentation and slides so their claims exactly match the new
  code and any newly produced evidence.

## Canonical contract

All hard-NMS implementations consume a validated candidate set:

```text
boxes:     float32 [N, 4]  in xyxy coordinates
scores:    float32 [N]
class_ids: int32   [N]
```

They return the original candidate indices kept by class-aware greedy NMS in
stable descending-score order.  NMS is run independently per class, matching
the normal detector-postprocessing contract and `torchvision.ops.nms`.

Invalid geometry, incompatible shapes, non-finite inputs, invalid thresholds,
and empty inputs have explicit, tested behavior.

## Input sources

### Synthetic stress source

Generate deterministic, class-labelled candidates at N = 100, 1,000 and
10,000.  This source is used only to measure NMS scaling, because a normal
detector may not produce 10,000 post-threshold candidates for one image.

### Real detector source

Run a pretrained detector on a supplied/demo image and extract pre-NMS raw
candidate boxes, scores and class ids.  The code must not use a detector API
whose returned detections have already passed NMS.  It produces a small,
auditable end-to-end demonstration:

```text
image -> model raw predictions -> canonical candidates -> CPU/V1/V2 NMS
```

This path demonstrates realistic integration; its timing is reported
separately from the NMS-only stress benchmark.

## Components

1. `src/nms_common.py`
   - input validation, stable class-aware ordering and candidate-source
     interfaces;
   - synthetic generation plus raw-detector extraction;
   - torchvision per-class oracle and small result summaries.

2. `src/cpu_baseline.py`
   - pure NumPy class-aware greedy NMS;
   - commands for synthetic benchmark, correctness verification and real
     detector demo/profile;
   - no CUDA imports.

3. `src/gpu_v1.py`
   - retains the intentionally naive all-pairs IoU design;
   - processes a single class partition at a time and delegates sorting,
     input handling and result mapping to shared utilities;
   - its limitation is explicit: full IoU transfer and CPU greedy resolution.

4. `src/gpu_v2.py`
   - uses a packed, class-partitioned suppression mask and supports a batch of
     independent candidate sets;
   - preserves hard-NMS output parity with CPU and torchvision;
   - reports the remaining host resolution as a known limitation rather than
     claiming fully device-resident greedy NMS.

5. Tests and benchmarks
   - CPU tests run without CUDA and cover validation, multi-class suppression,
     synthetic determinism and detector-output adaptation;
   - CUDA tests cover V1/V2 parity with CPU and torchvision on small,
     multi-class inputs; batch semantics and partial mask words remain tested;
   - NMS-only and full-pipeline benchmark commands write separate JSON
     artifacts with source, timing scope, environment, warmup/repeats and
     median/stddev.

## Seminar 2 claims

The deck will state:

- Synthetic inputs are controlled stress tests at fixed N; real detector
  outputs demonstrate integration.
- CPU, V1 and V2 are hard-NMS and compare to the same class-aware torchvision
  oracle.
- V3 is untouched Matrix NMS and compares only to its Matrix-NMS oracle.
- No performance target is claimed without evidence generated from the
  corresponding revised code.
- If the revised T4 run is unavailable before the seminar, slides show the
  architecture and verified correctness status only; historical results are
  visibly labelled as pre-restructure evidence.

## Acceptance criteria

- `python src/cpu_baseline.py --benchmark` runs without CUDA.
- A real-image command produces raw detector candidates, CPU NMS results and a
  torchvision parity result without relying on detector-internal NMS output.
- V1 and V2 use the canonical contract and preserve greedy NMS output per
  class.
- Unit tests distinguish CPU-only coverage from CUDA-required coverage.
- Benchmark JSON distinguishes `nms_only_synthetic` from
  `detector_plus_nms_real`.
- V3 files and implementation hashes are unchanged.
- The Seminar 2 deck and script contain no stale or contradictory claims.

## Risks and mitigation

- A Colab runtime may not expose a compatible Numba version: run a kernel smoke
  test first and use the existing documented fallback pin if necessary.
- Downloading model weights may fail or be slow: retain a checked-in demo-image
  path and a clear fallback that uses saved raw candidate fixtures, labelled as
  fixture data rather than live inference.
- Rebuilding V1/V2 can invalidate old T4 measurements: preserve the old
  evidence and re-run only claims intended for the new implementation.
- The batch-32 <5 ms target may remain unmet because hard greedy resolution has
  inherent dependency and the current V2 returns masks to the host: report the
  measured result honestly and explain the architectural limit.
