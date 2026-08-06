# Colab V1/V2 Evidence Notebooks Design

**Goal:** Replace the disorganized GPU notebooks with two reproducible Google
Colab notebooks that produce current CUDA correctness and benchmark evidence
for V1 and V2.

## Scope

- Replace the old `src/gpu_v1.ipynb` and `src/gpu_v2.ipynb` notebooks.
- Create `collab/v1_gpu_colab.ipynb` and `collab/v2_gpu_colab.ipynb`.
- Update the Colab links and instructions in `docs/HOW_TO_RUN.md`.
- Keep Baseline as the CPU/reference path invoked by both notebooks.
- Do not modify V3 source code or its notebook.

## Notebook Contract

Both notebooks run top-to-bottom on a fresh Colab runtime with a NVIDIA GPU.
Each notebook has these six labelled sections:

1. **Runtime gate** — fail early when `numba.cuda.is_available()` is false;
   print `nvidia-smi`, Python, Numba and the selected commit SHA.
2. **Source setup** — clone `liltommy142/cuda-nms-numba`, checkout an explicit
   `COMMIT` variable, install project dependencies and expose `src` through
   `PYTHONPATH`.
3. **Reference test** — run the focused CPU/torchvision and target-version
   correctness tests; a CUDA skip is treated as a failed evidence run.
4. **Warm-up** — run the selected GPU version once outside timing.
5. **Benchmark** — run reproducible NMS-only benchmarks for N=100, 1,000 and
   10,000. V2 additionally runs B=32 with N=100, 1,000 and 10,000.
6. **Evidence export** — write stdout/stderr, environment metadata and JSON
   results to `/content/evidence/<version>/`; offer the directory as a ZIP for
   manual download and later commit.

## Version-Specific Behavior

### V1

Runs the `v1` correctness tests and the CPU/V1 benchmark command. It reports
the full-IoU-matrix approach and does not describe V1 as fully parallel NMS.

### V2

Runs the `v2` correctness tests, the CPU/V1/V2 benchmark command, and the
batch-B=32 command. It reports packed-mask construction separately from the
host greedy resolver.

## Evidence and Safety Rules

- A result is valid only when the notebook records the exact commit SHA and
  reports a CUDA-capable NVIDIA device.
- The notebooks never reuse historical T4 numbers and never push to GitHub.
- A failed/skip test stops the notebook before performance numbers are shown.
- The user manually inspects and commits downloaded evidence; no token is
  stored in a notebook.

## Acceptance Checks

- Both notebook files open in Google Colab from the GitHub links.
- On a fresh T4 runtime, each notebook completes setup and detects CUDA.
- V1 executes focused parity tests and benchmarks N=100/1k/10k.
- V2 executes focused parity tests plus single-image and B=32 benchmarks.
- Each run creates a self-contained evidence ZIP with commit and environment
  metadata.
