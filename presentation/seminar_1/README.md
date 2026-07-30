# Seminar 1 — Proposal

## Submitted artifacts

- [Proposal deck](Slide_Proposal.pptx)
- [Written proposal](../../CSC14116%20-%20Proposal.docx)

## Agreed scope

The proposal commits the team to a CPU greedy-NMS reference, GPU V1 pairwise
IoU, GPU V2 coalesced suppression masks with batch-size-32 processing, and a
Matrix-NMS V3 stretch goal. It also requires correctness evidence and measured
benchmark results for `N = 100, 1,000, 10,000`.

The proposal's timing/profile values remain historical Seminar 1 evidence.
Seminar 2 must label its own runtime, GPU, package versions and measurement
method rather than reusing those values as final results.
