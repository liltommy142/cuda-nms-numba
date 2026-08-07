# One-click Colab GPU Notebook Design

## Goal

Provide one checked-in Colab notebook that a presenter can open and run from
top to bottom to prepare the verified CUDA 13 environment, run the complete
Baseline/V1/V2 validation workload, and collect reproducible evidence.

## Scope

- Create `collab/gpu_test_colab.ipynb` as the single runnable GPU-test entry
  point.
- Update `collab/readme.md` so the notebook is the recommended route.
- Add a repository test that validates the notebook's required workflow
  markers without executing Colab-only cells.
- Preserve V3: the notebook neither imports nor runs `gpu_v3`.

## Notebook workflow

The notebook contains ordered cells with one responsibility each:

1. Show `nvidia-smi` and fail early if no NVIDIA GPU is available.
2. Configure `COMMIT="main"`, clone or update `/content/cuda-nms-numba`, fetch
   `origin`, check out the configured revision, and print the resolved SHA.
   `main` gives a classroom run the newest pushed notebook-compatible source;
   replacing it with a full SHA freezes a reproducible run.
3. Build `/content/nms-cu13-venv` using `virtualenv`, then install NumPy
   1.26.4, pytest 9.1.1, `numba-cuda[cu13]`, and CPU-only PyTorch/torchvision.
   The project requirements file is deliberately not installed into Colab's
   global Python because its legacy Numba stack is incompatible with the
   current CUDA 13 runtime.
4. Run the V1 CUDA JIT smoke test.
5. Run `pytest tests -q -rs`.
6. Run `benchmarks/run_all.py` for CPU, V1 and V2 at 100, 1k, and 10k boxes.
7. Run `benchmarks/run_v2_batch.py` with B=32 and N=10k.
8. Write the resolved SHA, GPU description, Python version and key package
   versions to `/content/evidence/<short-sha>/environment.txt`; archive that
   directory as a ZIP. A separate manual-download cell uses `files.download`
   only when the presenter chooses to download the ZIP.

The detector integration cell is present but is disabled by
`RUN_DETECTOR = False`. When enabled after supplying `yolov5s.pt`, it runs V2
with `--max-candidates 11000`. Its absence or an unavailable model weight must
not make the default Run all workflow fail.

## Error handling

All Bash execution cells use `set -euo pipefail`. Failure in environment setup,
CUDA availability, smoke testing, pytest, or either required benchmark stops
the notebook before evidence is treated as valid. The notebook prints the
evidence directory at completion so it can be inspected or downloaded.

## Verification

`tests/test_colab_notebooks.py` parses the `.ipynb` JSON and asserts that the
required setup, CUDA 13 packages, full tests, both benchmark commands,
evidence archive, manual download helper, optional 11k detector budget, and
V3 exclusion are all represented. The normal local test suite validates this
structural contract; execution on an actual T4 is recorded by the notebook's
own evidence files.
