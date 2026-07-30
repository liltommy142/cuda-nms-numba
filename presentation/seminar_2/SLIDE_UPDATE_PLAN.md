# Việc cần làm với slide Seminar 2

Deck hiện tại có thể giữ phần giải thích bài toán và kiến trúc. Evidence CUDA
đã có, nên bước còn lại là đưa các số liệu dưới đây vào deck final.

## Giữ

- Problem, IoU/greedy-NMS flow và lý do GPU phù hợp.
- Sơ đồ V1: IoU matrix GPU, greedy host.
- Sơ đồ V2: SoA, shared target tile, packed uint64 mask.
- Sơ đồ V3: Matrix-NMS soft suppression.

## Sửa trước khi trình bày

1. Thay mọi số tốc độ “expected”, `[CHỜ COLAB]` hoặc historical one-shot bằng
   median/stddev từ `evidence/`.
2. Thêm slide **Kết quả và correctness**: GPU model, package version, test
   result, bảng latency N=100/1k/10k và nguồn artifact.
3. Thêm slide **V2 batch 32**: input `(32, N, 4)`, một grid 3D mask launch,
   latency batch và latency/image. Không gọi bitmask word-64 là batch 32.
4. Thêm slide **Giới hạn/trade-off**: V1 truyền ma trận IoU lớn; V2 còn CPU
   greedy resolution; V3 đổi algorithm nên không khớp hard NMS.
5. Thêm slide cuối **Reproducibility & contribution**: commit, command test,
   command benchmark, phân công hai thành viên.

## Số liệu phải dùng

- Tesla T4; Python 3.12.13; NumPy 2.0.2; Numba 0.60.0.
- Full CUDA correctness suite: **50 passed**.
- N=10,000 single-image median: CPU 1125.272 ms; V1 226.107 ms; V2 31.599
  ms; V3 4.092 ms.
- V2 batch 32 × 10,000: **1002.239 ms/batch**, 31.320 ms/image, stddev
  214.264 ms; target `<5 ms/batch` is not met.

## Không được nói trên slide

- “V3 đúng như torchvision NMS”.
- “V2 fully on-device greedy NMS”.
- Trộn số CPU proposal, Colab lịch sử và benchmark mới trong một biểu đồ.

## Việc còn lại

1. Cập nhật deck theo `OUTLINE_AND_CONTENT.md` và `SCRIPT.md`, giữ font/layout đã có.
3. Render, kiểm tra không tràn chữ và rehearsal theo `SCRIPT.md`.
4. Lưu PPTX/PDF final cạnh artifact evidence.
