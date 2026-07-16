# Dàn ý & nội dung slide — Nhóm 11: Real-Time NMS on GPU

> 10 slide, mỗi slide chỉ giữ bullet ngắn + gợi ý hình minh hoạ — chi tiết kỹ thuật đầy đủ nằm ở `SCRIPT.md` (lời nói) và `QA_PREP.md` (hỏi-đáp). Lý do tách như vậy: xem mục "Nhóm 5 — Transformer Attention" trong `CROSS_GROUP_LESSONS.md` (2 lần bị chê "quá chi tiết cho 1 buổi proposal").
>
> Chỗ nào ghi `[CHỜ COLAB]` nghĩa là số liệu chưa có thật — **phải điền số thật trước khi thuyết trình**, xem `README.md` mục "Trạng thái số liệu".

---

## Slide 1 — Mở đầu

**Real-Time Non-Maximum Suppression trên GPU**
CUDA (Numba) · CSC14116 Topic A4 · Nhóm 11
Lê Quang Tân (22127378) · Phùng Quốc Tuấn (19127616)

[Visual: 1 ảnh có vật thể với 5+ box chồng lấp xung quanh — dùng để mở đầu bằng hình ảnh trực quan trước khi giải thích]

---

## Slide 2 — Vấn đề

Detector (YOLO/SSD/Faster R-CNN) → hàng nghìn box ứng viên / ảnh
NMS lọc trùng → nhưng **NMS tuần tự = O(n²)**

Số liệu thật (cProfile, N=10.000, xem `cpu_baseline.ipynb`): tính IoU (`iou_one_to_many`) chiếm **~65%** thời gian, phần vòng lặp suppression (sort + bookkeeping, không tính IoU) chiếm **~35%** — tức là chính phần **song song hoá được** (IoU) mới là phần tốn thời gian nhất

[Visual: before/after — ảnh nhiều box chồng lấp → ảnh chỉ còn 1 box đúng]

---

## Slide 3 — Vì sao GPU phù hợp

Bài toán có 2 nửa rất khác nhau:
- Tính IoU giữa mọi cặp box → **độc lập hoàn toàn** (embarrassingly parallel)
- Quyết định giữ/loại box → **phụ thuộc tuần tự** (box sau phụ thuộc quyết định box trước)

→ Đây là **thử thách xuyên suốt cả 3 version** GPU của nhóm: mỗi version tấn công phần "tuần tự" này theo 1 cách khác.

[Visual: bên trái — lưới mũi tên song song (IoU); bên phải — 1 chuỗi mũi tên nối tiếp (suppression)]

---

## Slide 4 — Kiến trúc V1: song song hoá phần IoU

**Ý tưởng**: N² thread, mỗi thread tính đúng 1 cặp `IoU(i, j)` → ma trận IoU N×N tính xong trong 1 lần gọi kernel.
**Suppression**: vẫn chạy trên CPU, nhưng giờ chỉ là *tra bảng* (đọc ma trận có sẵn), không tính lại IoU như CPU baseline.

Bottleneck còn lại: tải cả ma trận N×N (float32) từ GPU về CPU qua PCIe → tăng theo O(n²).

[Visual: sơ đồ 3 bước — Host (sort) → Device (kernel N×N thread, song song) → Host (vòng lặp tuần tự, tra bảng)]

---

## Slide 5 — Kiến trúc V2: nén dữ liệu + đọc bộ nhớ liên tục

2 cải tiến so với V1:
1. **Coalesced memory (SoA)**: 4 mảng `x1,y1,x2,y2` riêng thay vì 1 mảng box gộp → thread trong cùng warp đọc liền kề nhau, gom lại thành 1 lần đọc bộ nhớ thay vì nhiều lần.
2. **Bitmask suppression trên GPU**: thay vì tải cả ma trận IoU float32 (N²×4 byte), GPU nén luôn kết quả "ai suppress ai" thành **bitmask 64-bit** (N²/64 phần tử) rồi mới tải về → giảm PCIe traffic ~64 lần so với V1.

Vòng lặp CPU cuối vẫn còn (1 lần/rank), nhưng mỗi lần chỉ OR 2 mảng ngắn (N/64 phần tử) thay vì so sánh cả hàng N phần tử như V1.

**Tốc độ (N=10.000)**: `[CHỜ COLAB]`× so với CPU — kỳ vọng ≥15× (mốc 100%, xem docstring `gpu_v2.py`)

[Visual: sơ đồ so sánh AoS (V1, đọc lệch) vs SoA (V2, đọc liền mạch); bên cạnh — khối bitmask nén thay vì ma trận float đầy]

---

## Slide 6 — Kiến trúc V3: Matrix NMS — bỏ hẳn vòng lặp CPU

Đổi hẳn thuật toán: từ **hard suppression** (giữ/loại nhị phân, phải làm tuần tự theo thứ tự score) sang **soft suppression** (giảm dần điểm số theo mức chồng lấp — Wang et al. 2020).

2 kernel, chạy hoàn toàn song song, **không còn vòng lặp CPU nào**:
1. Mỗi box tính `iou_max` với các box điểm cao hơn nó — song song cho **mọi box cùng lúc**.
2. Mỗi box tự tính hệ số suy giảm điểm số dựa trên `iou_max` đã có — cũng song song cho **mọi box cùng lúc**.

**Không chỉ nhanh hơn — còn sửa một lỗi thật của hard NMS**: khi 2 vật thể khác nhau đứng sát nhau (IoU giữa 2 box thật cao dù là 2 vật thể riêng biệt), V1/V2 (hard suppression) có thể xoá nhầm 1 trong 2 box đúng. V3 (soft suppression) chỉ giảm điểm chứ không xoá hẳn, nên giữ được cả 2 nếu điểm còn đủ cao sau khi giảm — đây là ví dụ cụ thể nên đưa ra khi trình bày, không chỉ nói chung chung "đánh đổi tốc độ".

**Đánh đổi cần nói rõ**: vì đổi thuật toán, tập box "giữ lại" của V3 **không còn khớp y hệt** CPU baseline/V1/V2 (khác kiểu so sánh — ngưỡng điểm số, không phải index) — đây là trade-off tốc độ/thiết kế, không phải bug.

**Tốc độ (N=10.000)**: `[CHỜ COLAB]`× so với CPU — kỳ vọng 30-80× (mốc 125%, xem docstring `gpu_v3.py`)

[Visual: sơ đồ 2 kernel nối tiếp, mỗi kernel có N khối chạy song song, không có mũi tên vòng lặp quay lại CPU giữa chừng]

---

## Slide 7 — Kết quả đo thật

| N | CPU (s) | GPU V1 (s) | Speedup V1 |
|---|---|---|---|
| 100 | 0.0069 | 0.0057 | **1.2×** |
| 1.000 | 0.1513 | 0.0146 | **10.3×** |
| 10.000 | 2.4918 | 0.2557 | **9.7×** |

(CPU và GPU đo cùng 1 lần chạy trên Colab T4, xem `src/gpu_v1.ipynb` — đảm bảo so sánh công bằng, cùng điều kiện máy.)

Đã đối chiếu 100% với `torchvision.ops.nms` (ground truth bên ngoài).
GPU V2 / V3: code xong, **đang chờ đo thật trên Colab T4** — `[CHỜ COLAB]`

[Visual: bar chart CPU vs GPU V1, 3 nhóm N=100/1.000/10.000 — nếu có số V2/V3 thật, thêm 2 cột mỗi nhóm]

---

## Slide 8 — Đang ở đâu (trung thực)

✅ CPU baseline — đúng, đã test tự động
✅ GPU V1 — đúng, đã đo tốc độ thật
✅ GPU V2 — code xong, test tự động đã viết, **đang chờ verify + benchmark thật trên Colab**
✅ GPU V3 — code xong (Matrix NMS), **đang chờ verify + benchmark thật trên Colab**
⏳ Batch size 32 (theo target catalog A4) — **chưa implement** ở cả 3 version, mỗi lần chạy vẫn xử lý 1 tập box

[Visual: checklist 5 dòng, dấu ✅/⏳ như trên]

---

## Slide 9 — Mục tiêu

| Mốc | Điều kiện | Trạng thái |
|---|---|---|
| 75% | GPU V1 đúng + benchmark | ✅ Đạt |
| 100% | + GPU V2, ≥15× tại N=10.000 | ⏳ Code xong, chờ đo |
| 125% (stretch) | + GPU V3, 30-80×, <5ms | ⏳ Code xong, chờ đo |

[Visual: bậc thang 3 tầng 75/100/125%, đánh dấu tầng nào đã/đang/chưa đạt]

---

## Slide 10 — Phân công & Kết

Phùng Quốc Tuấn — CPU baseline, test suite, repo
Lê Quang Tân — GPU V1/V2/V3 kernel, benchmark

Cảm ơn thầy/cô và các bạn — sẵn sàng nhận câu hỏi.

[Visual: 2 avatar/card đơn giản]
