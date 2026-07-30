# Seminar 2 submission checklist

Complete these steps in order. A later step must not replace missing evidence
from an earlier one.

## 1. Freeze the runnable code

- [ ] Commit or record the exact commit hash for the presentation build.
- [ ] Install the pinned CUDA-compatible environment on a NVIDIA GPU.
- [ ] Record Python, NumPy, Numba, CUDA driver/runtime and GPU model.

## 2. Prove correctness before timing

- [x] Run `python -m pytest tests -v` on CUDA and save the full log in `evidence/`.
- [x] Confirm V1 and V2 match CPU greedy NMS, including `N=10,000`.
- [x] Confirm V2 batch input includes a partial 64-box block and matches CPU per image.
- [x] Confirm V3 matches `matrix_nms_reference` for linear and Gaussian decay.

## 3. Produce reproducible measurements

- [x] Run the repeated single-image benchmark and save JSON plus terminal log.
- [x] Run V2 with `--batch-size 32`; record end-to-end latency per batch and per image.
- [x] Use warm-up, at least seven repeats, median and standard deviation.
- [x] Label any historical one-shot V1 result as historical, not final evidence.

## 4. Prepare the final story

- [x] Update every performance number in the final deck from saved artifacts.
- [x] Add result slides: correctness status, median latency, variability and environment.
- [x] Add one limitation slide: V2 still resolves greedy keep decisions on CPU; V3 changes the algorithm.
- [x] Do not claim V3 has identical outputs to torchvision greedy NMS.
- [x] Add a final slide with contributions and reproducibility references.

## 5. Rehearse and package

- [ ] Rehearse the full deck against the script; both members can explain V1, V2 and V3.
- [ ] Run the demo from a clean runtime or record a fallback video/screenshots.
- [x] Export final PDF/PPTX and prepare the local evidence bundle.
- [ ] Commit/push the final code and upload the prepared submission bundle.
