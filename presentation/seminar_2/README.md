# Seminar 2 — Implementation and Results

This folder is the single source of truth for Seminar 2 preparation. T4 test
and benchmark artifacts are now recorded under `evidence/`.

## Read in this order

1. [Code explanation](CODE_EXPLANATION_VI.md) — understand the current implementation.
2. [Colab manual test](COLAB_MANUAL_TEST.md) — correctness before performance.
3. [Submission checklist](SUBMISSION_CHECKLIST.md) — complete the evidence gate first.
4. [Slide update plan](SLIDE_UPDATE_PLAN.md) — update claims only after evidence exists.
5. [Submission list](submission/list.md) — final hand-in and GitHub check.
6. [Slide outline](OUTLINE_AND_CONTENT.md) — audience-facing content.
7. [Speaker script](SCRIPT.md) — rehearsal sequence and transitions.
8. [Q&A preparation](QA_PREP.md) — technical defence of design choices.
9. [Cross-group lessons](CROSS_GROUP_LESSONS.md) — likely feedback and pitfalls.
10. [Evidence folder](evidence/README.md) — where test logs and benchmark JSON belong.

## Current verified state (after baseline/V1/V2 restructure)

| Item | Status | What may be claimed now |
|---|---|---|
| Local CPU/reference suite | 25 passed, 41 CUDA-skipped | CPU baseline, raw YOLOv5 adapter, report metadata and non-GPU V1/V2 wrappers verified |
| CPU baseline | class-aware hard NMS + raw YOLOv5 pre-NMS adapter | synthetic N=100/1k/10k and real detector path are separate |
| GPU V1 / V2 | class-aware hard-NMS code + T4 parity tests added | requires rerun on the current commit before quoting a GPU number |
| GPU V3 | untouched Matrix NMS code/evidence | do not present it as hard-NMS parity |

### Historical T4 values — do not use as post-restructure results

The values in the old evidence files (V1 `226.107 ms`, V2 `31.599 ms`, V3
`4.092 ms` at N=10,000; V2 batch-32 `1.002 s`) were produced **before** the
class-aware rebuild. Keep them only as historical context. Rerun the current
commit in Colab/T4 to generate `*_restructured_t4.*` before putting any new
performance claim on the deck.

V1/V2 are class-aware hard greedy NMS and must be compared with the per-class
CPU/torchvision reference. V3 is Matrix NMS, so it must be compared with
`matrix_nms_reference`, not claimed to match greedy NMS.

Evidence: [`pytest_t4_final.txt`](evidence/pytest_t4_final.txt),
[`benchmark_t4_single.json`](evidence/benchmark_t4_single.json) and
[`batch32_t4.json`](evidence/batch32_t4.json). Environment: Tesla T4,
Python 3.12.13, NumPy 2.0.2 and Numba 0.60.0.

## Deck and source material

- [Final Seminar 2 deck — PPTX](Seminar_2_Final_T4.pptx)
- [Final Seminar 2 deck — PDF](Seminar_2_Final_T4.pdf)
- [Original working deck](Seminar_2_CUDA_NMS_Numba.pptx)
- [Local CPU profile](cprofile_N10000_local.txt)
- [Legacy preparation notes](LEGACY_STATUS_NOTES.md) — retained for history;
  verify every claim against this README and the evidence folder before reuse.

The automated B=32 correctness test uses N=50 to cover the partially filled
final 64-thread block. The separate latency benchmark uses B=32 and N=10,000.
