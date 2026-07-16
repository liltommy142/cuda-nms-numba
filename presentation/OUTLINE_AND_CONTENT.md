# Dàn ý & nội dung slide — Nhóm 11: Real-Time NMS on GPU

> **Đã đối chiếu với `Slide_Proposal.pptx`** (bản pptx thật của Tân, commit `10f3026`, 15 slide) — file này giờ theo đúng thứ tự/tiêu đề 15 slide của bản pptx, giữ lại toàn bộ nội dung chi tiết bản cũ (10 slide) bằng cách gộp vào đúng slide tương ứng. Không xoá nội dung cũ nào — phần nào pptx chưa có slide riêng (kết quả đo thật, trạng thái, mục tiêu, phân công) được giữ nguyên ở cuối, đánh dấu **[CHƯA CÓ TRONG PPTX]** để 2 bạn bổ sung thêm slide trước khi thuyết trình. Xem thêm ghi chú đối chiếu đầy đủ trong `README.md`.
>
> Mỗi slide chỉ giữ bullet ngắn + gợi ý hình minh hoạ — chi tiết kỹ thuật đầy đủ nằm ở `SCRIPT.md` (lời nói) và `QA_PREP.md` (hỏi-đáp). Lý do tách như vậy: xem mục "Nhóm 5 — Transformer Attention" trong `CROSS_GROUP_LESSONS.md` (2 lần bị chê "quá chi tiết cho 1 buổi proposal").
>
> Chỗ nào ghi `[CHỜ COLAB]` nghĩa là số liệu chưa có thật — **phải điền số thật trước khi thuyết trình**, xem `README.md` mục "Trạng thái số liệu".

---

## Slide 1 — Mở đầu

**Real-Time Non-Maximum Suppression for Object Detection**
Numba-CUDA & Matrix NMS Accelerated Pipeline on GPU
CSC14116 Topic A4 · Nhóm 11 · Lê Quang Tân (22127378) · Phùng Quốc Tuấn (19127616)

> Bản pptx thật để tên nhóm/tác giả ở **slide cuối** (Slide 15), không phải slide mở đầu — giữ lại dòng tên tác giả ở đây vì hữu ích khi đọc riêng file này, nhưng khi trình bày theo đúng pptx thì phần giới thiệu tên nói bằng lời (xem `SCRIPT.md`), không hiện trên slide 1.

[Visual: 1 ảnh có vật thể với 5+ box chồng lấp xung quanh (người, xe, đèn giao thông, chó...) — dùng để mở đầu bằng hình ảnh trực quan trước khi giải thích. Ảnh thật đã có trong pptx.]

---

## Slide 2 — Vấn đề (Problem)

Các mô hình Object Detection hiện nay như YOLO, SSD hay Faster R-CNN tạo ra hàng ngàn bounding box ứng viên cho mỗi khung hình. NMS lọc các hộp trùng lặp thông qua so khớp độ phủ IoU nhằm giữ lại những hộp tối ưu nhất.

Trên CPU, Greedy NMS truyền thống thực thi tuần tự với độ phức tạp **O(n²)** — điểm nghẽn nghiêm trọng khi suy luận (inference) ở batch size lớn hoặc ảnh có nhiều vật thể.

[Visual: ảnh nhiều box chồng lấp quanh người/xe/vật thể — dùng chung ảnh mở đầu ở Slide 1, hoặc before/after nếu có]

---

## Slide 3 — NMS Greedy (lưu đồ thuật toán)

*(Slide mới trong bản pptx — chưa có trong bản outline cũ, bổ sung ở đây)*

Lưu đồ (flowchart) thuật toán Greedy NMS truyền thống, các bước:
1. Nhập danh sách bounding boxes + scores → sắp xếp theo điểm tin cậy giảm dần.
2. Lặp: nếu danh sách rỗng → xuất Keep List, kết thúc. Nếu còn box → chọn box điểm cao nhất, thêm vào Keep List.
3. Duyệt qua từng box còn lại (CPU loop): tính `IoU(current_max, other_box)`. Nếu `IoU > ngưỡng` → loại bỏ box; nếu không → giữ lại cho vòng sau.
4. Lặp lại bước 2-3 cho đến khi duyệt hết.

→ Đây chính là bước tuần tự (vòng lặp CPU) sẽ được phân tích kỹ ở Slide 5 ("Thách Thức") và là mục tiêu cả 3 version GPU nhắm vào giải quyết.

[Visual: lưu đồ đầy đủ có trong pptx — Start → Nhập box+scores → Sort → check rỗng → chọn box cao nhất → thêm Keep List → duyệt box còn lại → tính IoU → so ngưỡng → loại bỏ/giữ lại → lặp]

---

## Slide 4 — CPU Baseline thực tế

- Khi số lượng box lên **1.000**, bắt đầu xuất hiện độ trễ.
- Khi số lượng box lên **10.000**, hệ thống đã nghẽn:
  - **65% thời gian (~0.189s)**: tiêu tốn trực tiếp trong vòng lặp khử trùng lặp tuần tự (suppression loop).
  - **34% thời gian (~0.098s)**: nằm ở hàm tính khoảng cách IoU giữa các bounding box.

| N | Thời gian CPU (đo trong proposal) |
|---|---|
| 100 | 0.0008 s |
| 1.000 | 0.0103 s |
| 10.000 | 0.2846 s |

> ⚠️ **Cần lưu ý khi trình bày (chưa thống nhất giữa 2 bạn)**: bảng số + tỉ lệ 65/34% trên là **số đo ở giai đoạn viết proposal ban đầu** (đúng như pptx ghi chú "đo trong proposal"), dùng để minh hoạ động lực bài toán (tại sao cần tăng tốc). Đây **không phải** số liệu Colab đã verify dùng ở phần kết quả (Slide bổ sung "Kết quả đo thật" bên dưới, số thật: CPU N=10.000 = **2.4918s** trên Colab T4). Hai bộ số fine để cùng tồn tại nếu nói rõ ngữ cảnh khác nhau (proposal ban đầu vs benchmark Colab sau này), nhưng **đừng trộn lẫn khi trả lời câu hỏi** — nếu bị hỏi "vậy CPU N=10.000 mất bao lâu, 0.28s hay 2.5s?", trả lời: "0.28s là số đo nhanh lúc viết proposal trên máy khác; 2.49s là số đo lại kỹ trên Colab T4 dùng để so sánh tốc độ GPU — chênh lệch do khác phần cứng/điều kiện đo, không phải sai số". Xem thêm `QA_PREP.md` mục H và `README.md`.

[Visual: bảng số liệu + biểu đồ cột thời gian CPU theo N — đã có sẵn trong pptx]

---

## Slide 5 — Thách Thức

Bước quyết định triệt tiêu (suppression) mang tính **tuần tự nghiêm ngặt**. Hộp B có được giữ lại hay không phụ thuộc trực tiếp vào việc một hộp A có điểm cao hơn trước đó đã bị xoá hay được giữ.

→ Đây chính là **thử thách xuyên suốt cả 3 version GPU** của nhóm: mỗi version tấn công phần "tuần tự" này theo 1 cách khác.

[Visual: sơ đồ chuỗi phụ thuộc A → B → C..., mỗi bước phải biết kết quả bước trước]

---

## Slide 6 — Roadmap (lộ trình tăng tốc)

*(Slide mới trong bản pptx — bản đồ tổng quan tốc độ, đặt sớm trong bài để người nghe nắm lộ trình trước khi đi vào chi tiết từng version)*

CPU → V1 → V2 → V3
Tốc độ: **1× → 5-10× → 15× → 30-80×**

- **CPU**: Baseline performance — 1×
- **V1**: Initial acceleration — 5-10×
- **V2**: Optimized throughput — 15×
- **V3**: High-performance tier — 30-80×

> Khớp với bảng "Mục tiêu 75/100/125%" ở slide bổ sung bên dưới — đây là bản trình bày trực quan (mũi tên) của cùng 1 dữ liệu mục tiêu, nên dùng nhất quán 1 bộ số khi trả lời câu hỏi.

[Visual: sơ đồ roadmap dạng sóng/mũi tên nối tiếp CPU→V1→V2→V3, đã có sẵn trong pptx]

---

## Slide 7 — Why GPU? (Vì sao GPU phù hợp)

Bài toán có 2 nửa rất khác nhau — 3 lý do bài toán IoU hợp GPU:

1. **Phép Toán IoU Độc Lập**: tính toán mức độ giao nhau giữa từng cặp hộp (box_i, box_j) hoàn toàn độc lập với nhau — không có ràng buộc dữ liệu chéo nào ở bước tính IoU.
2. **Số Lượng Thread Khổng Lồ**: với N hộp ứng viên, có thể phân bổ 1 thread GPU cho mỗi cặp hộp → hệ thống tính đồng thời N² thread song song.
3. **Độ Song Song Cực Đại**: đây là bài toán song song hoá điển hình (*embarrassingly parallel*), cực kỳ phù hợp với kiến trúc hàng nghìn nhân tính toán của GPU.

Ngược lại, quyết định giữ/loại box lại **phụ thuộc tuần tự** (box sau phụ thuộc quyết định box trước, xem Slide 5) — đây là phần khó, không tự động hợp GPU, và là lý do cần 3 version tấn công theo 3 cách khác nhau.

[Visual: bên trái — lưới mũi tên song song (IoU); bên phải — 1 chuỗi mũi tên nối tiếp (suppression)]

---

## Slide 8 — GPU V1: Naive Parallel IoU Matrix Kernel

**Ý tưởng**: N² thread, mỗi thread tính đúng 1 cặp `IoU(i, j)` → ma trận IoU N×N tính xong trong 1 lần gọi kernel.
**Suppression**: vẫn chạy trên CPU, nhưng giờ chỉ là *tra bảng* (đọc ma trận có sẵn), không tính lại IoU như CPU baseline.

Cơ chế hoạt động GPU V1 — 3 trụ cột: **1. Tính toán độc lập · 2. Phân bổ luồng · 3. Xử lý đồng thời**

[Visual: sơ đồ 3 bước — Host (sort) → Device (kernel N×N thread, song song) → Host (vòng lặp tuần tự, tra bảng); kèm sơ đồ "Cơ chế hoạt động GPU V1" 3 nhánh đã có trong pptx]

---

## Slide 9 — Hạn chế còn tồn tại của V1 → chuyển sang V2

Nút thắt cổ chai của V1:
1. **Nghẽn truyền tải dữ liệu (PCIe Bottleneck)**: tải cả ma trận N×N (float32) từ GPU về CPU qua PCIe → tăng theo O(n²) — ở N=10.000 là khoảng 400MB.
2. **Vòng lặp loại bỏ tuần tự trên CPU (Sequential Suppression Loop)**: suppression vẫn còn 1 vòng lặp Python chạy trên CPU.

**=> Cần tiếp tục cải tiến với GPU v2: Batched NMS & Hardware Optimization**

[Visual: sơ đồ "Hạn chế còn tồn tại (Nút thắt cổ chai của V1)" — 2 nhánh PCIe Bottleneck / Sequential Suppression Loop, đã có trong pptx]

---

## Slide 10 — GPU v2: Batched NMS & Hardware Optimization

2 cải tiến so với V1:
1. **Coalesced Memory Access**: 4 mảng `x1,y1,x2,y2` riêng (SoA) thay vì 1 mảng box gộp (AoS) → tối ưu hoá cách đọc toạ độ hộp để các luồng liên tiếp truy cập bộ nhớ liền kề, tiết kiệm tối đa băng thông VRAM.
2. **Batched NMS & Parallel Reduction**: gom cụm các hộp thành khối 64 (do dùng số nguyên 64-bit) và áp dụng kỹ thuật rút gọn song song (parallel reduction) cùng phép toán logic cấp bit để dựng **bitmask suppression** trực tiếp trên GPU — giảm PCIe traffic ~64 lần so với V1.

> ⚠️ **Lưu ý phân biệt chữ "Batched" ở đây**: "Batched NMS" trong tên slide này nghĩa là *gom nhóm 64 box/khối* để nén bitmask (khớp đúng thiết kế `_nms_bitmask_kernel` đã có trong code) — **khác** với "batch size 32" theo catalog đề tài A4 (xử lý nhiều ảnh/nhiều tập box cùng lúc trong 1 lần gọi), phần đó **vẫn chưa được implement** ở bất kỳ version nào (xem Slide "Đang ở đâu" bên dưới). Cần nói rõ khác biệt này nếu bị hỏi, tránh gây hiểu lầm là đã làm xong batch size.

Vòng lặp CPU cuối vẫn còn (1 lần/rank), nhưng mỗi lần chỉ OR 2 mảng ngắn (N/64 phần tử) thay vì so sánh cả một hàng dài như V1.

**Tốc độ (N=10.000)**: `[CHỜ COLAB]`× so với CPU — kỳ vọng ≥15× (mốc 100%, xem docstring `gpu_v2.py`)

[Visual: sơ đồ "GPU V2: Batched NMS & Hardware Optimization" 2 nhánh Coalesced Memory Access / Batched NMS & Parallel Reduction, đã có trong pptx]

---

## Slide 11 — Hạn chế của GPU V2 (Nút thắt thuật toán)

*(Nội dung mới từ pptx — chưa có trong bản outline cũ, quan trọng: nói rõ V2 vẫn CHƯA giải quyết xong bài toán, chuẩn bị lý do cho V3)*

1. **Vẫn mang bản chất Greedy NMS**: dù đã gom cụm (batches) và tối ưu phần cứng, GPU V2 vẫn phải giải quyết chuỗi phụ thuộc dữ liệu tuần tự (quyết định giữ/xoá hộp B vẫn phụ thuộc vào việc hộp điểm cao hơn đã bị xoá hay chưa).
2. **Chưa song song hoá triệt để**: việc dựng mặt nạ triệt tiêu (suppression mask) bằng parallel reduction chỉ **giảm thiểu độ trễ**, chứ **chưa triệt tiêu được** tư duy so sánh tuần tự của thuật toán gốc.

→ Đây là lý do V3 phải **đổi hẳn thuật toán** (soft suppression) thay vì tiếp tục tối ưu phần cứng như V1→V2.

[Visual: sơ đồ "Hạn chế của GPU V2 (Nút thắt thuật toán)" — 2 nhánh, đã có trong pptx]

---

## Slide 12 — GPU v3: Matrix NMS

Giải pháp này chuyển đổi việc triệt tiêu cứng tuần tự thành cơ chế tính toán độ suy giảm mềm (**soft decay**) song song hoàn toàn.

Đổi hẳn thuật toán: từ **hard suppression** (giữ/loại nhị phân, phải làm tuần tự theo thứ tự score) sang **soft suppression** (giảm dần điểm số theo mức chồng lấp — Wang et al. 2020).

**Ưu điểm vượt trội của Matrix NMS** (theo pptx):
1. **Tốc độ "bàn thờ"** — nhanh vượt trội so với 2 bản trước.
2. **Độ chính xác cao** — không xoá nhầm box do chỉ giảm điểm thay vì loại hẳn (xem Slide 13).
3. **Không tốn tài nguyên huấn luyện** — là kỹ thuật hậu xử lý (post-processing), không cần train lại model.

2 kernel, chạy hoàn toàn song song, **không còn vòng lặp CPU nào**:
1. Mỗi box tính `iou_max` với các box điểm cao hơn nó — song song cho **mọi box cùng lúc**.
2. Mỗi box tự tính hệ số suy giảm điểm số dựa trên `iou_max` đã có — cũng song song cho **mọi box cùng lúc**.

**Tốc độ (N=10.000)**: `[CHỜ COLAB]`× so với CPU — kỳ vọng 30-80× (mốc 125%, xem docstring `gpu_v3.py`)

[Visual: sơ đồ 2 kernel nối tiếp + sơ đồ "Ưu điểm vượt trội của Matrix NMS" 3 nhánh, đã có trong pptx]

---

## Slide 13 — V3 giải quyết đúng 1 lỗi thật của Hard NMS

**=> Giải quyết được bài toán 2 vật thể đứng quá sát nhau dẫn tới độ chồng lấn (IoU) cao và bị xoá**

**Không chỉ nhanh hơn — còn sửa một lỗi thật của hard NMS**: khi 2 vật thể khác nhau đứng sát nhau (IoU giữa 2 box thật cao dù là 2 vật thể riêng biệt), V1/V2 (hard suppression) có thể xoá nhầm 1 trong 2 box đúng. V3 (soft suppression) chỉ giảm điểm chứ không xoá hẳn, nên giữ được cả 2 nếu điểm còn đủ cao sau khi giảm.

Khái niệm "Soft Decay" (Suy giảm mềm): thay vì `IoU(A,B) > Threshold → Score_B = 0` (xoá sổ lập tức) như Hard Suppression, Soft Decay dùng công thức `Score_new = Score_old × Decay_Factor` — hạ thấp điểm số của các hộp đè chồng lên nhau một cách mịn màng, không xoá ngay.

**Đánh đổi cần nói rõ**: vì đổi thuật toán, tập box "giữ lại" của V3 **không còn khớp y hệt** CPU baseline/V1/V2 (khác kiểu so sánh — ngưỡng điểm số, không phải index) — đây là trade-off tốc độ/thiết kế, không phải bug.

[Visual: sơ đồ "Khái niệm Soft Decay" — so sánh Hard Suppression (A/B, Score_B=0) vs Soft Decay (A/B, Score_B giảm dần), đã có trong pptx]

---

## Slide 14 — Tại sao Soft Decay là chìa khoá song song hoá 100% (GPU V3)

1. **Phá vỡ chuỗi phụ thuộc tuần tự**: trong Greedy NMS, phải biết hộp A có bị xoá hay không mới quyết định được số phận hộp B. Matrix NMS bỏ hẳn yêu cầu này.
2. **Tính toán song song độc lập**: với Matrix NMS (Wang et al., 2020), hệ số `Decay_Factor` của một hộp được tính hoàn toàn độc lập, dựa trên mức độ chồng lấn lớn nhất của nó với tất cả các hộp có điểm số cao hơn — `Decay_Factor_i = f(max IoU_i)`.
3. **Thực thi trên GPU**: phép toán này quy về 1 bước duyệt ma trận IoU song song duy nhất + 1 phép nhân ma trận điểm số đồng loạt trên GPU — đạt tốc độ vượt trội **30-80×**.

[Visual: sơ đồ 3 nhánh "Tại sao Soft Decay là chìa khoá..." — đã có trong pptx, kèm minh hoạ ma trận IoU (N×N) × vector điểm số → vector điểm số mới]

---

## Slide 15 — Kết

**-Lê Quang Tân, Phùng Quốc Tuấn-**
Thanks for your listening

[Visual: ảnh nền/closing card đã có trong pptx]

---

## Slide bổ sung — nội dung có trong bản outline cũ nhưng **[CHƯA CÓ TRONG PPTX]**

> pptx 15-slide hiện tại **chưa có slide riêng** cho: kết quả đo thật (bảng số CPU vs GPU V1), trạng thái từng version (checklist ✅/⏳), bảng mục tiêu 75/100/125% dạng chi tiết (chỉ có bản rút gọn ở Slide 6 "Roadmap"), và phân công công việc. Giữ nguyên các mục này ở đây — **2 bạn cần thống nhất có thêm slide cho phần này vào pptx hay không** trước khi thuyết trình, vì đây là nội dung quan trọng (số liệu thật + trạng thái trung thực + phân công) mà catalog đề tài thường yêu cầu.

### Kết quả đo thật

| N | CPU (s) | GPU V1 (s) | Speedup V1 |
|---|---|---|---|
| 100 | 0.0069 | 0.0057 | **1.2×** |
| 1.000 | 0.1513 | 0.0146 | **10.3×** |
| 10.000 | 2.4918 | 0.2557 | **9.7×** |

(CPU và GPU đo cùng 1 lần chạy trên Colab T4, xem `src/gpu_v1.ipynb` — đảm bảo so sánh công bằng, cùng điều kiện máy. **Khác với bảng "đo trong proposal" ở Slide 4** — xem ghi chú ở đó.)

Đã đối chiếu 100% với `torchvision.ops.nms` (ground truth bên ngoài).
GPU V2 / V3: code xong, **đang chờ đo thật trên Colab T4** — `[CHỜ COLAB]`

[Visual: bar chart CPU vs GPU V1, 3 nhóm N=100/1.000/10.000 — nếu có số V2/V3 thật, thêm 2 cột mỗi nhóm]

### Đang ở đâu (trung thực)

✅ CPU baseline — đúng, đã test tự động
✅ GPU V1 — đúng, đã đo tốc độ thật
✅ GPU V2 — code xong, test tự động đã viết, **đang chờ verify + benchmark thật trên Colab**
✅ GPU V3 — code xong (Matrix NMS), **đang chờ verify + benchmark thật trên Colab**
⏳ Batch size 32 (theo target catalog A4) — **chưa implement** ở cả 3 version, mỗi lần chạy vẫn xử lý 1 tập box (không nhầm với "Batched NMS" ở Slide 10 — xem ghi chú ở đó)

[Visual: checklist 5 dòng, dấu ✅/⏳ như trên]

### Mục tiêu (bản chi tiết — bản rút gọn xem Slide 6 "Roadmap")

| Mốc | Điều kiện | Trạng thái |
|---|---|---|
| 75% | GPU V1 đúng + benchmark | ✅ Đạt |
| 100% | + GPU V2, ≥15× tại N=10.000 | ⏳ Code xong, chờ đo |
| 125% (stretch) | + GPU V3, 30-80×, <5ms | ⏳ Code xong, chờ đo |

[Visual: bậc thang 3 tầng 75/100/125%, đánh dấu tầng nào đã/đang/chưa đạt]

### Phân công & lời cảm ơn

Phùng Quốc Tuấn — CPU baseline, test suite, repo
Lê Quang Tân — GPU V1/V2/V3 kernel, benchmark

Cảm ơn thầy/cô và các bạn — sẵn sàng nhận câu hỏi. (Slide 15 của pptx đã có câu chào tiếng Anh "Thanks for your listening" + tên 2 bạn — có thể dùng làm slide phân công luôn nếu không muốn thêm slide riêng, chỉ cần nói thêm phần phân công bằng lời.)

[Visual: 2 avatar/card đơn giản]
