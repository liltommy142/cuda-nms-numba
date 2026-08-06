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

## Slide 10 — V3 Matrix NMS

V3 soft-decay score để bỏ dependency greedy, nhưng không cùng semantics hard
NMS. So với Matrix-NMS reference, không claim torchvision parity.

## Slide 11 — Trạng thái hiện tại

Local suite: 26 passed, CUDA tests skipped trên Mac. V1/V2 class-aware và T4
parity tests đã sẵn sàng; rerun CUDA là evidence gate trước số speedup mới.

## Slide 12 — Kết luận

V1 minh họa pairwise parallelism; V2 giảm traffic/memory; hard greedy decision
vẫn là giới hạn. V3 là trade-off thuật toán, không phải drop-in hard NMS.
