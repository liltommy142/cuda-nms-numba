# Seminar 2 — Implementation and Results

This folder is the single source of truth for Seminar 2 preparation. T4 test
and benchmark artifacts are now recorded under `evidence/`.

## Read in this order

1. [Code explanation](CODE_EXPLANATION_VI.md) — understand the current implementation.
2. [Slide outline](OUTLINE_AND_CONTENT.md) — audience-facing content.
3. [Speaker script](SCRIPT.md) — rehearsal sequence and transitions.
4. [Q&A preparation](QA_PREP.md) — technical defence of design choices.
5. [Submission checklist](SUBMISSION_CHECKLIST.md) — evidence gate before hand-in.
6. [Evidence folder](evidence/README.md) — where test logs and benchmark JSON belong.

## Current verified state (after baseline/V1/V2 restructure)

| Item | Status | What may be claimed now |
|---|---|---|
| Local CPU/reference suite | 30 passed, 41 CUDA-skipped | CPU baseline, raw YOLOv5 adapter, report metadata and non-GPU V1/V2 wrappers verified |
| CPU baseline | class-aware hard NMS + raw YOLOv5 pre-NMS adapter | synthetic N=100/1k/10k and real detector path are separate |
| GPU V1 / V2 | class-aware hard-NMS code + T4 parity tests added | requires rerun on the current commit before quoting a GPU number |

### Historical T4 values — do not use as post-restructure results

The old evidence files were produced **before** the class-aware rebuild. Keep
them only as historical context. Rerun the current commit in Colab/T4 to
generate `*_restructured_t4.*` before putting any new performance claim on the
deck.

V1/V2 are class-aware hard greedy NMS and must be compared with the per-class
CPU/torchvision reference. Seminar 2 is intentionally limited to Baseline,
V1 and V2.

Evidence: [`pytest_t4_final.txt`](evidence/pytest_t4_final.txt),
[`benchmark_t4_single.json`](evidence/benchmark_t4_single.json) and
[`batch32_t4.json`](evidence/batch32_t4.json). Environment: Tesla T4,
Python 3.12.13, NumPy 2.0.2 and Numba 0.60.0.

## Deck and source material

- [Final Seminar 2 deck — PPTX](Seminar_2_Final_T4.pptx) — 12-slide
  Baseline/V1/V2 narrative; no historical T4 number is presented as current.
- [Final Seminar 2 deck — PDF](Seminar_2_Final_T4.pdf) — exported from the
  same deck.
- [Local CPU profile](cprofile_N10000_local.txt)

The automated B=32 correctness test uses N=50 to cover the partially filled
final 64-thread block. The separate latency benchmark uses B=32 and N=10,000.
