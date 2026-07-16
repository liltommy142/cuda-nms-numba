# Bài học từ feedback 12 nhóm khác (nguồn: `docs/ALPP_22KHMT+KHDL-SeminarList.xlsx`)

> File xlsx có 13 sheet: 1 sheet "Seminar" (danh sách lớp — nhóm mình là **Nhóm 11**, đề tài "Real-Time Non-Maximum Suppression for Object Detection", 2 thành viên) + 12 sheet feedback/Q&A cho từng nhóm (2 → 13). **Sheet của Nhóm 11 hiện trống** — nghĩa là buổi thuyết trình của nhóm mình chưa diễn ra, các nhóm dưới đây đã trình bày trước và để lại dữ liệu thật để mình học theo/tránh.

## Tóm tắt nhanh — điều cần làm khác

| # | Nhóm / đề tài | Điều xảy ra | Việc cụ thể nhóm mình cần làm |
|---|---|---|---|
| 1 | Nhóm 4 — Movie Recommendation | Bị chê thẳng: *"The presenters do not seem to understand their own work, including the dataset, the methodology, or whether TF-IDF is suitable for GPU acceleration—which it generally is not."* | Phải tự trả lời chắc chắn: **vì sao NMS phù hợp GPU** (IoU độc lập từng cặp = embarrassingly parallel) — không học thuộc câu trả lời, phải hiểu để trả lời biến thể câu hỏi. Cả 2 thành viên phải trả lời được, không chỉ 1 người "hiểu sâu". |
| 2 | Nhóm 8 — Image Processing Pipeline | Comment: *"chưa hiểu quy trình song song hóa (cả v1 và v2)"* dù nhóm đã trình bày 2 version | Slide kiến trúc phải **vẽ/diễn giải rõ luồng dữ liệu + ai làm gì trên GPU vs CPU** cho từng version, không chỉ liệt kê "nhanh hơn X lần". Xem slide 8/10/12 (kiến trúc V1/V2/V3) trong `OUTLINE_AND_CONTENT.md` — giờ có thêm slide 9/11 nói rõ hạn chế còn tồn tại của V1/V2, càng làm rõ luồng dữ liệu hơn bản cũ. |
| 3 | Nhóm 5 — Transformer Attention | 2 lần bị chê: *"Too technically detailed for a proposal, which should focus more on the overall plan"*, *"A proposal should present the solution at a higher level and remain more general"* — dù nội dung đúng và được khen "clearly, good, well prepared" | Slide chính giữ **cao cấp** (kiến trúc + kết quả), không nhồi công thức/dòng code vào slide. Chi tiết (công thức IoU, bounds guard, warp shuffle...) chỉ nói khi được hỏi, dựa vào `QA_PREP.md`. |
| 4 | Nhóm 9 — PageRank | *"Trình bày chưa rõ ràng lắm"* dù phần trả lời Q&A rất sâu và đúng | Cấu trúc bài nói phải có tín hiệu chuyển ý rõ ràng ("tiếp theo là...", "so với V1 thì V2 khác ở chỗ..."). Xem `SCRIPT.md` — mỗi slide có câu mở/chuyển ý riêng. |
| 5 | Nhóm 3 — 3D Gaussian Splatting | Bị hỏi dồn: bottleneck compute-bound hay memory-bound? mốc 100% cụ thể là optimization level nào (naive CUDA/tiled/optimized sort)? Có đo cả chất lượng output không (PSNR/SSIM) hay chỉ tốc độ? | Nhóm mình đã có sẵn: PCIe O(n²) là bottleneck ở N lớn (ghi rõ trong `docs/TECHNICAL_DOCUMENTATION.md` mục 3.3). Cần nói rõ **mốc 75/100/125% ứng với version nào** (V1/V2/V3) — bản rút gọn xem slide 6 "Roadmap", bảng chi tiết xem mục "Mục tiêu" ở cuối `OUTLINE_AND_CONTENT.md` (slide bổ sung, hiện chưa có trong pptx — xem cảnh báo đầu file đó). Về "chất lượng output": NMS không có metric ảnh như PSNR, nhưng V3 đổi thuật toán (soft-suppression) nên tập box giữ lại khác V1/V2 — đây là câu cần chuẩn bị kỹ, xem mục "Trade-off chất lượng V3" trong `QA_PREP.md`. |
| 6 | Nhóm 7 — Fast Neural Style | Câu hỏi: *"What is the trade-off between speed and image quality?"*, *"V1 GPU có thể chậm hơn CPU không?"* — nhóm trả lời tốt, được khen | Chuẩn bị sẵn câu trả lời cho **N nhỏ → GPU có thể không nhanh hơn** (đã có: N=100 chỉ 1.2× do overhead JIT + PCIe). Và trade-off tốc độ/chất lượng — áp dụng y hệt cho V3 (soft-NMS). |
| 7 | Nhóm 9 — PageRank | Bị hỏi rất sâu về thiết kế: tại sao chọn CSR thay vì adjacency matrix, thread có phụ thuộc nhau không, warp shuffle cải thiện gì | Chuẩn bị câu trả lời "vì sao chọn thiết kế X thay vì Y" cho từng version — đã có trong `QA_PREP.md` (vì sao AoS→SoA ở V2, vì sao bitmask thay vì mask bool thường, vì sao 2 kernel riêng ở V3 thay vì 1 kernel). |
| 8 | Nhiều nhóm (3, 4, 8, 9...) | Bị hỏi về **metric đánh giá đúng-sai** rõ ràng (Precision/Recall, PSNR/SSIM, so khớp bao nhiêu %...) | Nhóm mình đã có metric rõ: so khớp *tập box giữ lại* với `torchvision.ops.nms` (ground truth), dung sai IoU 1e-4. Cần nói thêm: V3 dùng metric khác (score threshold, không so khớp index trực tiếp) vì bản chất thuật toán khác — đừng để bị hỏi bí. |

## Bài học tổng hợp áp dụng vào cấu trúc bài nói

1. **Hiểu sâu, không học thuộc.** Câu trả lời hay nhất trong toàn bộ feedback (Nhóm 5, câu hỏi Q&A B13) không phải trả lời đúng công thức, mà giải thích được *lý do thiết kế* ("tại sao 1 thread chỉ tính 1 phần tử output" → giải thích cả về bản chất GPU lẫn vì sao tăng khối lượng/thread sẽ làm chậm đi). Nhóm mình cần trả lời kiểu đó cho câu tương tự: "vì sao 1 thread tính đúng 1 cặp IoU (i,j)" — đã có sẵn trong `QA_PREP.md`.
2. **Trung thực về phần chưa làm/chưa đo tốt hơn là né tránh.** Không nhóm nào trong feedback bị chê vì nói "phần này em chưa đo/chưa làm" — chỉ bị chê khi tỏ ra không hiểu hoặc trả lời sai. Nhóm mình sẽ nói rõ V2/V3 **code xong nhưng benchmark thật đang chờ chạy Colab**, batch size 32 **chưa implement**.
3. **1 slide kiến trúc / 1 version, không gộp chung.** Đây là khác biệt lớn nhất so với slide cũ (bản cũ gộp V1/V2/V3 vào 1 slide roadmap duy nhất, chỉ có số speedup). Giờ có code thật cho cả 3 version nên có thể (và nên) tách riêng, dùng đúng bài học của Nhóm 8.
4. **Chuẩn bị câu hỏi "trade-off"** — mọi nhóm có sản phẩm nhiều version đều bị hỏi kiểu này (tốc độ đổi lấy gì). Với NMS: V1→V2 đổi bộ nhớ/độ phức tạp code lấy tốc độ (không đổi độ chính xác — vẫn hard NMS, khớp 100% CPU). V2→V3 đổi **cả thuật toán** (hard→soft suppression) — tốc độ tăng nhưng tập box giữ lại không còn giống hệt CPU baseline nữa, đây là trade-off thật cần nói chủ động, đừng đợi bị hỏi.

## Dữ liệu gốc (dịch/trích các câu hỏi tiêu biểu, đầy đủ hơn xem trực tiếp file xlsx)

<details>
<summary>Nhóm 3 — 3D Gaussian Splatting</summary>

- Chi tiết về bottleneck hiện tại, kỹ thuật song song hóa sẽ áp dụng
- How do you handle Gaussians that overlap many screen-space tiles? Is the bottleneck more memory-bound or compute-bound? What optimization level is your 100% target: naive CUDA, tiled rendering, or optimized sorting/blending?
- Do you measure both FPS and image quality metrics such as PSNR or SSIM?
- Vì sao Sort vẫn là bottleneck dù dùng GPU? Memory-bound khác Compute-bound thế nào?
</details>

<details>
<summary>Nhóm 4 — Movie Recommendation System</summary>

- Comment: "The presenters do not seem to understand their own work, including the dataset, the methodology, or whether TF-IDF is suitable for GPU acceleration—which it generally is not."
- Do you think the TF-IDF is GPU-friendly? Đối tượng được chia cho các thread là gì?
</details>

<details>
<summary>Nhóm 5 — Transformer Attention Mechanism</summary>

- Comment (x2): "Too technically detailed for a proposal, which should focus more on the overall plan" / "A proposal should present the solution at a higher level and remain more general."
- Comment (positive, x3): "Clearly, good, well prepared" / "Thuyết trình tốt" / "The presentation covers all the fundamentals of the chosen topic."
- Tại sao một thread chỉ tính một output element? Vì sao phải tạo ba ma trận Q, K, V? Softmax cũng O(N²), sao vẫn tối ưu?
</details>

<details>
<summary>Nhóm 7 — Fast Neural Style</summary>

- Comment: "Nhóm trình bày tốt, đầy đủ" / "The presentation covers all the fundamental knowledge of the topic; it also details the project planning."
- V1 GPU có thể chậm hơn CPU không? What is the trade-off between speed and image quality? Vì sao Memory Optimization trước Compute Optimization?
</details>

<details>
<summary>Nhóm 8 — Image Processing Pipeline</summary>

- Comment: "chưa hiểu quy trình song song hóa (cả v1 và v2)"
- Phase nào có thể tối ưu nhiều thời gian nhất và tại sao? Một hàm kernel Numba CUDA khác gì hàm Python thường về `return`?
</details>

<details>
<summary>Nhóm 9 — PageRank</summary>

- Comment: "Trình bày chưa rõ ràng lắm"
- Chi tiết về CPU cơ bản và các phiên bản GPU (V1/V2/V3 Pull/V3 Push) — nhóm trả lời rất kỹ, đúng, dùng làm mẫu tham khảo cách trả lời "giải thích thiết kế từng version" cho nhóm mình.
- Tăng số thread có luôn làm chương trình nhanh hơn không? (không — tranh chấp atomic, register/shared memory tăng)
</details>
