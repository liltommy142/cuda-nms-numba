# Ghi chú chuẩn bị thuyết trình proposal

> Tổng hợp từ `CSC14116 - Proposal.docx`, `Project Topic Catalog.pdf` (đề A4), và code/docs hiện có trong repo.

## 1. Những điểm đáng lưu ý để lên slide

### Vấn đề & lý do dùng GPU
- Detector (YOLO/SSD/Faster R-CNN) sinh hàng nghìn box/ảnh → NMS lọc lại, nhưng NMS tuần tự là **O(n²)** → bottleneck ở batch size lớn.
- Số liệu profiling thật (`cProfile`, N=10.000, xem [profile_output/cprofile_N10000.txt](../profile_output/cprofile_N10000.txt)): tổng 0.289s, trong đó **65% ở vòng lặp suppression**, **34% ở `iou_one_to_many`** → >99% thời gian là chính NMS, không phải I/O. Đây là số liệu mạnh nhất để mở đầu slide.
- Tính IoU giữa mọi cặp box **độc lập hoàn toàn** (embarrassingly parallel) → giao cho N² thread GPU; nhưng bước suppression có **phụ thuộc tuần tự** → đây là "cái khó" xuyên suốt dự án.

### Kế hoạch tối ưu — bảng V0→V3
| Version | Kỹ thuật | Speedup kỳ vọng (N=10.000) |
|---|---|---|
| CPU baseline | Greedy NMS tuần tự, NumPy thuần | 1× |
| GPU V1 | Kernel IoU song song (1 thread/cặp), suppression vẫn tuần tự trên host | ~5–10× |
| GPU V2 | + batched NMS, parallel reduction cho suppression mask, coalesced memory | ≥15× (mốc 100%) |
| GPU V3 (stretch) | Matrix NMS (Wang et al. 2020) — soft-suppression hoàn toàn song song | 30–80× (mốc 125%), <5ms @N=10.000/batch 32 |

**Số liệu thật đã đo trên Colab T4** (lưu trong `gpu_v1.ipynb`) so với dự đoán trong proposal:
- N=100 → **1.2×** (thấp hơn dự đoán — N nhỏ, overhead JIT-compile + PCIe transfer lấn át lợi ích song song)
- N=1.000 → **10.3×**
- N=10.000 → **9.7×**

→ Khớp đúng khoảng dự đoán (~5–10×) ở N lớn, không đạt ở N nhỏ — slide tốt để nói "dự đoán vs thực đo", và giải thích được *tại sao* GPU không phải lúc nào cũng nhanh hơn.

### Đã đạt mục tiêu <5ms chưa? — **Chưa, còn cách khá xa**
Mục tiêu "<5ms tại N=10.000, batch 32" là mốc **125% (stretch)**, gắn với **GPU V3 (Matrix NMS)** — phần này **chưa có code**, nên chưa thể đạt được.

Số liệu GPU V1 hiện tại (đã đo thật trên Colab T4, N=10.000, **chưa có chiều batch** — chỉ 1 tập box, không phải batch 32):
- **GPU V1: 255.7 ms** (0.2557s) — so với mục tiêu **<5 ms** → **chậm hơn ~51 lần** so với mục tiêu cuối (V3).
- Đây là con số hoàn toàn bình thường ở giai đoạn này: GPU V1 chỉ nhắm mốc 75% (naive, chưa tối ưu), tiêu chí `<5ms` chỉ áp dụng cho V3 — **không nên nhầm là dự án đang "trễ tiến độ"**.

Nếu bị hỏi "vậy đã đạt mục tiêu chưa" trong buổi thuyết trình proposal, câu trả lời an toàn: *"Hiện tại nhóm em mới hoàn thành GPU V1 (mốc 75%) với kết quả thật đo trên Colab — 255.7ms tại N=10.000, đúng như kỳ vọng ở giai đoạn naive. Mục tiêu <5ms/30-80× là của GPU V3, dự kiến hoàn thành ở tuần 9-10 theo lịch đã đề ra."* Đừng để bị hiểu nhầm là đang so sánh GPU V1 với mục tiêu V3 — 2 con số này thuộc 2 mốc khác nhau (75% vs 125%).

### Trạng thái thực tế vs. proposal (nên nói rõ, tránh bị hỏi bất ngờ)
- **Đã hoàn thành**: CPU baseline + GPU V1 (đúng tiến độ tuần 7).
- **Chưa làm**: GPU V2, GPU V3 — vẫn là kế hoạch, chưa có code (đúng lịch tuần 8-10, chưa tới hạn).
- **Gap**: proposal ghi benchmark ở "batch size 32", nhưng **CPU baseline lẫn GPU V1 hiện tại chưa có chiều batch** — chỉ xử lý 1 tập box/lần gọi.

### Risk Analysis — 3 rủi ro đã xác định
1. **Colab hạn chế quyền profiling** (`nvprof`/Nsight cần root) → mitigation: dùng `torch.cuda.Event`/`numba.cuda.event` (wall-clock timing), test sớm từ tuần 1.
2. **Sai số floating-point + tie-break khác nhau** giữa kernel tự viết và `torchvision.ops.nms` → có thể ra tập box giữ khác nhau dù IoU gần giống hệt. Mitigation: so khớp trong dung sai 1e-4, dùng stable sort cả 2 bên, báo "kept-box-set match rate".
3. **GPU V3 có thể không kịp tiến độ** (khó nhất, xếp cuối lịch) → đã scope là mục tiêu 125% (stretch); có fallback 75% (chỉ V1 + phân tích gap).

### Mục tiêu & tiêu chí đạt
- **75%**: chỉ GPU V1, đo correctness + speedup baseline, chưa tối ưu.
- **100%**: CPU baseline đúng + GPU V1 đúng + GPU V2 + đạt **≥15× speedup** tại N=10.000.
- **125% (stretch)**: GPU V3 đạt 30–80×, <5ms; thêm so sánh Soft-NMS vs Matrix NMS.
- **Demo tại seminar**: biểu đồ latency CPU vs GPU V1 vs V2 (vs V3), demo ảnh nhiều box chồng lấp → sau NMS chỉ còn box đúng, bảng so sánh CPU vs GPU.

### Phân công & tiến độ
- **Phùng Quốc Tuấn (19127616)**: dựng repo Git, CPU baseline + notebook + test suite ban đầu, viết draft proposal.
- **Lê Quang Tân (22127378)**: GPU V1 (kernel + fix vectorize suppression loop), phần test GPU trong `test_correctness.py`, hoàn thiện proposal nộp.
- Lưu ý: proposal ghi rõ **cả 2 thành viên phải giải thích được toàn bộ code** — yêu cầu tham gia của môn học.

### Tài liệu tham khảo
Bodla et al. 2017 (Soft-NMS), Wang et al. 2020 (Matrix NMS, mục 3.3 — nền tảng V3), Hosang et al. 2017 (Learning NMS), `torchvision.ops.nms`/`box_iou` (chỉ đối chiếu, không copy), NVIDIA TensorRT NMS plugin (chỉ tham khảo).

### Vì sao batch size = 32? (dễ bị hỏi, cần trả lời đúng)
**32 không phải do nhóm tự chọn** — đây là con số do **`Project Topic Catalog.pdf`, đề A4** quy định sẵn:
> "Performance target: Process 10,000 boxes at **batch size 32** in under 5 ms. Target 30–80× speedup over the CPU NMS step."

Proposal của nhóm cũng ghi rõ đang bám theo *"the catalog's official target"*. Câu trả lời chuẩn nếu bị hỏi: *"Đây là mốc chuẩn do catalog đề tài A4 quy định, nhóm em bám theo để kết quả so sánh được trực tiếp với chuẩn chấm điểm của môn — không phải nhóm tự đặt ra."*

Lý do kỹ thuật chung khiến 32 là lựa chọn phổ biến trong ML/GPU (bổ sung nếu bị hỏi sâu hơn):
- Bội số của **warp size = 32** trên GPU NVIDIA → tận dụng phần cứng tốt, không lãng phí thread (cùng logic với `_TPB=(16,16)=256` trong `gpu_v1.py`, cũng là bội số của 32).
- Cân bằng giữa batch nhỏ (mô phỏng real-time, ít tận dụng song song) và batch lớn (tận dụng GPU tối đa nhưng tốn VRAM, không còn giống real-time).

### Lưu ý khác từ catalog (Git Repository requirement)
Catalog ghi rõ: **`cpu_baseline.py` phải chạy được end-to-end không cần GPU** — nếu không chạy được, deliverable tuần 6 = **0 điểm**. Đây là lý do phần CPU baseline (đã verify chạy hoàn hảo trên máy này) quan trọng hơn cả GPU V1 về mặt chấm điểm nền tảng.

---

## 2. Bảng thuật ngữ — giải thích dễ hiểu

### Nhóm "đo lường / benchmark"

**Profiling** — gắn đồng hồ bấm giờ vào **từng hàm** khi chạy code, để biết hàm nào ăn nhiều thời gian nhất. Ví dụ: `cProfile` trên `run_cpu` N=10.000 → 65% thời gian ở vòng lặp suppression, 34% ở hàm tính IoU. Giống bấm giờ từng bước nấu phở để biết bước nào lâu nhất.

**Bottleneck (nút thắt cổ chai)** — bước **chậm nhất**, quyết định tốc độ tổng thể. Biết bottleneck ở đâu mới biết nên tối ưu chỗ nào.

**Wall-clock timing** — đo thời gian thật trôi qua (như `time.perf_counter()`), khác với đo "chỉ riêng GPU tính toán mất bao lâu" (cần công cụ chuyên biệt như `nvprof`).

**`nvprof` / Nsight** — công cụ NVIDIA để "mổ xẻ" bên trong GPU (occupancy, memory throughput). Colab có thể chặn quyền dùng công cụ này → phương án dự phòng là wall-clock timing.

### Nhóm "GPU / song song"

**Kernel** — "công thức" mà GPU phát cho hàng nghìn thread cùng lúc, mỗi thread tự biết mình là ai và chỉ làm phần việc của mình. Trong project: `_iou_matrix_kernel` — mỗi thread tính IoU của đúng 1 cặp box.

**Thread** — 1 "công nhân" nhỏ trên GPU. GPU có hàng nghìn thread chạy song song, khác CPU chỉ vài lõi tuần tự.

**Embarrassingly parallel** — bài toán mà các phần việc hoàn toàn không liên quan nhau, ai làm phần nấy. Tính IoU giữa các cặp box là dạng này.

**JIT-compile (Just-In-Time)** — Numba biên dịch kernel thành mã máy **ngay lúc chạy lần đầu**. Cần "warm-up" (chạy thử nhỏ trước) để không tính nhầm thời gian compile vào benchmark.

**PCIe / overhead truyền dữ liệu** — đường ống nối CPU (RAM) và GPU (VRAM), chậm hơn nhiều so với GPU tự tính nội bộ. Ma trận IoU N×N lớn → thời gian "đi qua ống" có thể vượt cả thời gian tính toán.

**Overhead** — chi phí phụ cố định phải trả thêm mỗi lần gọi GPU (khởi động kernel, copy dữ liệu). N nhỏ → overhead lớn hơn cả lợi ích song song (lý do N=100 chỉ nhanh 1.2× chứ không phải 5-10×).

**Coalesced memory access** *(GPU V2, chưa code)* — các thread cạnh nhau đọc dữ liệu ở vị trí bộ nhớ gần nhau → GPU gom lại đọc 1 lần cho nhiều thread (nhanh hơn). Giống đi chợ mua đồ ở các sạp liền kề thay vì rải rác.

**Parallel reduction** *(GPU V2, chưa code)* — gộp nhiều giá trị thành 1 kết quả bằng cách chia nhỏ và gộp dần theo cặp trên GPU, thay vì gộp tuần tự. Ví dụ: 8 người cộng tiền — chia 4 cặp cộng song song, rồi gộp tiếp, chỉ 3 bước thay vì 7.

### Nhóm "thuật toán / độ chính xác"

**IoU (Intersection over Union)** — tỉ lệ phần "chồng lấp" giữa 2 khung hình. Giống hệt → IoU=1. Không chạm → IoU=0.

**Suppression** — bước loại bỏ khung hình bị coi là trùng lặp (IoU vượt ngưỡng so với khung đã giữ).

**O(n²) / Big-O** — cách nói độ phức tạp tăng theo cỡ dữ liệu N. O(n²): N tăng gấp đôi → thời gian tăng ~4 lần.

**Stable sort (sắp xếp ổn định)** — khi 2 phần tử bằng nhau, giữ nguyên thứ tự gốc. Quan trọng để kết quả tất định (chạy nhiều lần ra cùng 1 kết quả).

**Tie-break** — cách xử lý khi 2 giá trị bằng nhau tuyệt đối, cần quy tắc rõ ràng để quyết định ai "thắng".

**Floating-point mismatch / ULP** — máy tính lưu số thập phân không tuyệt đối chính xác; 2 cách tính khác thứ tự cộng trừ có thể lệch nhau vài ULP (đơn vị sai số nhỏ nhất). Vì vậy so sánh chỉ cần khớp trong sai số nhỏ (1e-4), không cần khớp tuyệt đối.

**Batch size** — số lượng "tập dữ liệu" xử lý cùng lúc trong 1 lần chạy. Xem mục "Vì sao batch size = 32" ở trên.

**Mitigation** — biện pháp phòng ngừa/giảm nhẹ rủi ro, chuẩn bị phương án B trước.

---

*File này để tra cứu nhanh trước buổi thuyết trình — xem thêm giải thích kỹ thuật đầy đủ tại [docs/TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) và hướng dẫn chạy code tại [docs/HOW_TO_RUN.md](HOW_TO_RUN.md).*
