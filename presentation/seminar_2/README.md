# Seminar 2 — Implementation and Results

This folder is the single source of truth for Seminar 2 preparation. Do not
present an unverified speedup or latency as a measured result.

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

## Current verified state

| Item | Status | What may be claimed now |
|---|---|---|
| CPU greedy NMS | CPU tests pass locally | CPU reference and synthetic benchmark only |
| GPU V1 | Historical T4 single-run result exists | Historical result only; rerun for final evidence |
| GPU V2 | Batched mask implementation exists | Architecture only; CUDA correctness and timing pending |
| GPU V3 | Matrix-NMS implementation exists | Algorithm/design only; CUDA correctness and timing pending |

V1/V2 are hard greedy NMS and must be compared with the CPU/torchvision
reference. V3 is Matrix NMS, so it must be compared with
`matrix_nms_reference`, not claimed to match greedy NMS.

## Deck and source material

- [Seminar 2 deck](Seminar_2_CUDA_NMS_Numba.pptx)
- [Local CPU profile](cprofile_N10000_local.txt)
- [Legacy preparation notes](LEGACY_STATUS_NOTES.md) — retained for history;
  verify every claim against this README and the evidence folder before reuse.
