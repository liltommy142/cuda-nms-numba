# Dàn ý Seminar 2 — CUDA NMS with Numba

> Chỉ dùng số performance từ evidence sinh bởi **commit đang trình bày**. T4
> evidence hiện có là historical pre-restructure, không đặt lên slide mới.

## Slide 1 — Bài toán

Detector tạo rất nhiều bounding-box candidates. NMS loại candidate trùng lặp
để giữ box có score tốt hơn.

## Slide 2 — Pipeline theo catalog

`YOLO raw candidates → class-aware hard NMS → final detections`.

Phân biệt rõ demo detector thật với benchmark synthetic N=100/1k/10k.

## Slide 3 — Vì sao NMS khó song song

IoU giữa các cặp box độc lập, nhưng quyết định greedy keep/suppress phụ thuộc
thứ tự score. Đây là nút thắt của hard NMS.

## Slide 4 — CPU baseline

Contract chung: `boxes`, `scores`, `class_ids`. CPU chạy greedy theo từng
class; đây là oracle nội bộ để kiểm tra V1/V2.

## Slide 5 — V1: full IoU matrix

Một thread cho một cặp IoU. GPU tạo ma trận đầy đủ; CPU dùng ma trận đó để
resolve greedy. Hạn chế: memory và transfer `O(N²)`.

## Slide 6 — V2: SoA + bitmask

SoA giúp coalesced read; bitmask `uint64` nén quan hệ suppress. Greedy resolver
trên CPU vẫn còn. Đừng gọi bitmask packing là parallel reduction.

## Slide 7 — Multi-class và batch

NMS chạy độc lập theo class. Batch một class dùng fused V2 launch; multi-class
partition theo class để bảo toàn semantics.

## Slide 8 — Đúng đắn

CPU/V1/V2 đối chiếu per-class `torchvision.ops.nms`; IoU kernels so với CPU
trong tolerance `1e-4`. Tie-break dùng stable descending score order.

## Slide 9 — Benchmark đúng scope

- Synthetic NMS: scaling của post-processing.
- Detector + NMS: raw candidate extraction và NMS được đo tách.
- Không gọi số NMS-only là latency inference end-to-end.

## Slide 10 — Evidence gate cho batch benchmark

Protocol batch là `B=32, N=10.000`, nhưng local Mac không có NVIDIA CUDA.
Deck chỉ báo `30 passed, 41 CUDA-skipped`; correctness parity và timing CUDA
phải được rerun từ chính commit sẽ trình bày trước khi công bố speedup.

## Slide 11 — Giới hạn hiện tại

V1 trả dense relation GPU → CPU; V2 nén relation thành `uint64` bitmask nhưng
greedy resolver cuối vẫn ở CPU. Đây là giới hạn còn lại của hard NMS hiện tại.

## Slide 12 — Kết luận

Seminar chỉ trình bày Baseline, V1 và V2. V1 minh họa pairwise parallelism;
V2 giảm traffic/memory; hard greedy decision vẫn là giới hạn. Các hướng NMS
khác nằm ngoài scope buổi này.
