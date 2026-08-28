# Seminar 3 Submission Bundle Design

## Scope

The graded path is the catalog-aligned, class-aware hard-NMS pipeline:

`candidates -> CPU baseline -> GPU V1 -> GPU V2 -> correctness -> timing`

- CPU is the transparent NumPy baseline.
- V1 computes the dense pairwise IoU relation on CUDA and resolves greedy NMS
  on the host.
- V2 uses SoA reads and packed 64-bit suppression masks, including batch 32.
- V3 remains a separately labelled Matrix-NMS experiment. It must not be
  presented as hard-NMS parity or mixed into the primary performance claim.
- The detector integration is included when its external model asset is
  available, but missing YOLO weights must not invalidate the synthetic NMS
  correctness and performance submission.

## Deliverables

Tracked files live under `submission/seminar_3/`:

- `FINAL_REPORT.ipynb`: audience-facing runnable report and experiment driver.
- `README.md`: exact run instructions, scope, and verified result summary.
- `TEAM_PLAN.md`: proposal-backed division of work plus completion evidence.
- `SUBMISSION_MANIFEST.txt`: source commit, contents, claims, and limitations.
- `SHA256SUMS.txt`: checksums for every tracked submission artifact and fresh
  evidence file.
- `evidence/pytest_cuda.txt`: complete current CUDA test output.
- `evidence/benchmark_v1_v2.json` and `.txt`: repeated NMS-only sweep.
- `evidence/batch32_v2.json` and `.txt`: repeated B=32, N=10,000 measurement.
- `evidence/environment.txt`: commit, GPU, driver, Python, and package versions.

The final untracked ZIP is written to `submission/seminar_3/output/` and
contains the tracked submission files plus the source, tests, and benchmark
scripts needed by the notebook. The ZIP is a hand-in convenience artifact;
Git remains the source of truth.

## Runtime and provenance

- Use Python 3.11 in an isolated local environment.
- Use NumPy 1.26.4, pytest 9.1.1, `numba-cuda[cu13]`, and CPU-only
  torch/torchvision 2.5.1/0.20.1 for the trusted hard-NMS oracle.
- Run a CUDA smoke test before the full suite.
- Record the exact commit used for evidence. If packaging introduces later
  documentation-only commits, the manifest distinguishes the tested source
  commit from the package commit.
- Do not copy historical pre-restructure T4 numbers into current claims.

## Error handling and honesty rules

- Environment, correctness, and performance failures are reported separately.
- A failed CUDA smoke test stops GPU evidence generation.
- Correctness must pass before any timing is accepted.
- Benchmark commands use warm-up and repeated samples and write JSON.
- Missing optional detector weights are reported as an explicit limitation.
- The `<5 ms` catalog target is reported as met or missed from measurements;
  the package never edits or cherry-picks numbers to make the target look met.

## Verification

1. Validate the isolated environment and CUDA availability.
2. Run a small deterministic CPU/CUDA smoke comparison.
3. Run the complete pytest suite with zero failures.
4. Run V1/V2 repeated benchmarks at N=100, 1,000, and 10,000.
5. Run V2 B=32, N=10,000 repeated benchmark.
6. Execute the final notebook in the submission environment.
7. Validate notebook JSON, manifest entries, hashes, and ZIP contents.
8. Re-run the complete suite after any product-code fix.

## Non-goals

- No slide deck, speaker script, or presentation rehearsal material.
- No fabricated detector evidence without an available model checkpoint.
- No claim that V3 Matrix NMS is equivalent to torchvision hard NMS.
- No system-wide CUDA driver or toolkit changes.
