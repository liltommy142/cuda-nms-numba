# Chuẩn bị thuyết trình seminar — Nhóm 11 (Real-Time NMS on GPU)

> 🧭 Về [docs/INDEX.md](../docs/INDEX.md) · Giải thích thuật ngữ xem [docs/GLOSSARY.md](../docs/GLOSSARY.md), giải thích code/kiến trúc chi tiết xem [docs/TECHNICAL_DOCUMENTATION.md](../docs/TECHNICAL_DOCUMENTATION.md).
>
> Thay thế hoàn toàn 4 file cũ trong `docs/` (`SLIDE_CONTENT.md`, `PRESENT_SCRIPT.md`, `PRESENTATION_NOTES.md`, `QA_PREP.md`) — các file đó viết từ lúc chỉ có CPU baseline + GPU V1, giờ V2 và V3 đã có code nên nội dung cũ sai ("V2 ⏳", "V3 ⏳ chưa có code"). Đây là bộ tài liệu viết lại từ đầu, dựa trên đúng trạng thái code hiện tại + bài học rút ra từ feedback thật của 12 nhóm khác trong lớp.

## Các file trong folder này

| File | Dùng để làm gì | Khi nào đọc |
|---|---|---|
| [`OUTLINE_AND_CONTENT.md`](OUTLINE_AND_CONTENT.md) | Dàn ý + nội dung từng slide (bullet ngắn, gợi ý hình minh hoạ) | Copy vào Google Slides |
| [`SCRIPT.md`](SCRIPT.md) | Kịch bản nói đầy đủ, khớp từng slide, có ước lượng thời lượng | Học/luyện nói trước buổi |
| [`QA_PREP.md`](QA_PREP.md) | Hỏi-đáp chuẩn bị sẵn — khái niệm cơ bản, thiết kế V1/V2/V3, trade-off, rủi ro | Đọc kỹ trước, không cần học thuộc |
| [`CROSS_GROUP_LESSONS.md`](CROSS_GROUP_LESSONS.md) | Tổng hợp feedback thật của 12 nhóm khác từ `docs/ALPP_22KHMT+KHDL-SeminarList.xlsx`, map sang rủi ro cụ thể của nhóm mình | Đọc trước để biết giảng viên/lớp hay hỏi kiểu gì |

Google Slides hiện có (từ bản cũ, **cần cập nhật lại nội dung theo bộ tài liệu mới này** trước khi thuyết trình): https://docs.google.com/presentation/d/10enq7N5OOawB4k1h8jwsGlImZjNQYfnySDcQV4pCLOE/edit?usp=sharing

**✅ Đã đối chiếu với `Slide_Proposal.pptx`** (commit `10f3026`, 15 slide, đã có đủ 3 phần V1/V2/V3 kể cả sơ đồ kiến trúc dạng ảnh). `OUTLINE_AND_CONTENT.md` và `SCRIPT.md` đã được viết lại để khớp đúng thứ tự/tiêu đề 15 slide của bản pptx — không xoá nội dung cũ, chỉ renumber + gộp. Dùng bản pptx của Tân làm slide chính khi trình bày, `SCRIPT.md`/`QA_PREP.md`/`CROSS_GROUP_LESSONS.md` làm phần chuẩn bị nói + hỏi-đáp đi kèm.

Các điểm đáng chú ý phát hiện khi đối chiếu trực tiếp với nội dung pptx (kể cả các slide chỉ có hình, đã trích xuất ảnh để đọc):

1. **V3 giải quyết lỗi hard NMS xoá nhầm 2 vật thể sát nhau** — đã có sẵn trong pptx (Slide 13), khớp với những gì đã bổ sung trước đó vào `QA_PREP.md` mục F.
2. **2 slide "hạn chế" mới, chưa từng có trong bộ tài liệu cũ**: Slide 9 (hạn chế V1: PCIe bottleneck + sequential suppression loop) và **Slide 11 (hạn chế V2 — quan trọng, dễ bị hỏi)**: dù đã batch + tối ưu phần cứng, V2 *vẫn* mang bản chất Greedy NMS (vẫn phụ thuộc tuần tự), parallel reduction chỉ giảm độ trễ chứ chưa loại bỏ được tư duy so sánh tuần tự — đây là lý do chính đáng cho việc cần đổi hẳn thuật toán ở V3. Đã thêm vào `OUTLINE_AND_CONTENT.md`/`SCRIPT.md`.
3. **Matrix NMS có 3 ưu điểm được liệt kê rõ trong pptx** (Slide 12): tốc độ, độ chính xác cao (không xoá nhầm), và không tốn tài nguyên huấn luyện (là bước hậu xử lý, không cần train lại model) — bổ sung thêm ý thứ 3 này, trước đây bộ tài liệu chỉ nhấn tốc độ + đúng đắn.
4. **pptx chưa có slide riêng cho**: kết quả đo thật (bảng CPU vs GPU V1), trạng thái từng version, bảng mục tiêu 75/100/125% chi tiết, và phân công công việc — các mục này chỉ có bản rút gọn ở Slide 6 "Roadmap" (1×→5-10×→15×→30-80×, khớp với bảng mục tiêu). Đã giữ nguyên toàn bộ nội dung cũ này ở cuối `OUTLINE_AND_CONTENT.md`, đánh dấu `[CHƯA CÓ TRONG PPTX]` — **2 bạn cần quyết định có thêm slide cho phần này vào pptx hay không** trước khi thuyết trình, vì đây là nội dung giám khảo hay hỏi (số liệu thật, trạng thái trung thực, phân công).
5. **⚠️ Xung đột số liệu cần thống nhất** — xem mục ngay bên dưới.

## Trạng thái số liệu — cái gì thật, cái gì đang chờ

| Version | Correctness | Speedup đo thật trên Colab T4 | Trạng thái |
|---|---|---|---|
| CPU baseline | Khớp `torchvision.ops.nms` (test tự động) | **2.49s @ N=10.000** (mốc 1×, đo cùng lần chạy với GPU V1 bên dưới — xem `src/gpu_v1.ipynb`) | ✅ Xong |
| GPU V1 | Khớp CPU baseline (test tự động, đã chạy Colab) | **9.7× @ N=10.000** (256ms), 10.3× @ N=1.000, 1.2× @ N=100 — số thật, xem `src/gpu_v1.ipynb` | ✅ Xong, đã đo thật |
| GPU V2 | Code xong; có test tự động (`test_correctness.py`) nhưng **chưa từng chạy trên máy có GPU** — `src/gpu_v2.ipynb` chưa có cell output nào | **CHƯA ĐO** — cần chạy `python src/gpu_v2.py --benchmark` (hoặc mở `gpu_v2.ipynb` trên Colab) | ⏳ Code xong, chờ verify |
| GPU V3 | Code xong; test sanity cơ bản, **chưa từng chạy trên máy có GPU** — `src/gpu_v3.ipynb` chưa có cell output | **CHƯA ĐO** — cần chạy `python src/gpu_v3.py --benchmark` | ⏳ Code xong, chờ verify |

**→ Việc cần làm trước khi hoàn thiện slide**: chạy `gpu_v2.ipynb` và `gpu_v3.ipynb` trên Colab (T4), lấy bảng benchmark thật, rồi thay các chỗ đánh dấu `[CHỜ COLAB]` trong `OUTLINE_AND_CONTENT.md` và `SCRIPT.md` bằng số thật. Cách chạy chi tiết: xem `docs/HOW_TO_RUN.md`.

**Lưu ý chữ "Batched" trong tiêu đề pptx Slide 10** ("GPU v2: Batched NMS & Hardware Optimization"): nghĩa là gom nhóm 64 box/khối để nén bitmask (`_nms_bitmask_kernel`), **không phải** batch size 32 theo catalog A4 (xử lý nhiều ảnh/tập box cùng lúc). Dòng "Batch size 32" trong bảng trạng thái vẫn đúng là **chưa implement** ở cả 3 version — đừng để tên slide gây hiểu lầm là đã làm xong.

Trong lúc soạn bộ tài liệu này, đã tìm và sửa 1 bug: `tests/test_correctness.py` có 4 test gọi `compute_iou_matrix_gpu_v2` nhưng hàm đó chưa tồn tại trong `src/gpu_v2.py` → sẽ lỗi `ImportError` khi chạy trên Colab. Đã thêm hàm này vào `gpu_v2.py` (xem commit liên quan) — cần chạy lại `pytest tests/ -v` trên Colab để xác nhận toàn bộ test V2 pass thật, không chỉ hết lỗi import.

**Đã đính chính 1 sai số liệu quan trọng**: bản trước của tài liệu proposal (`docs/TECHNICAL_DOCUMENTATION.md` và các file `.md` slide cũ đã xoá) ghi CPU baseline N=10.000 là **0.2846s / 0.289s**, và cProfile là **65% suppression / 34% IoU** — số này **chưa từng khớp với bất kỳ output thật nào đã lưu trong repo** (có vẻ là số dự kiến từ bản proposal ban đầu, chưa cập nhật lại sau khi chạy Colab thật). Đã đối chiếu lại với output thật trong `cpu_baseline.ipynb`/`gpu_v1.ipynb` và sửa toàn bộ: CPU N=10.000 thật là **~1.8-2.5s** (tuỳ lần chạy Colab), và tỉ lệ cProfile đúng là **~65% IoU / ~35% suppression loop** — **nhãn bị đảo ngược** trong bản cũ (may mắn là kết luận "IoU là phần nặng nhất và song song hoá được" vẫn đúng, chỉ sai con số/nhãn cụ thể). Đã sửa trong `OUTLINE_AND_CONTENT.md`, `SCRIPT.md`, và `docs/TECHNICAL_DOCUMENTATION.md`.

Đã chạy lại cProfile thật trên máy local để đối chiếu thêm 1 lần nữa — output lưu tại [`cprofile_N10000_local.txt`](cprofile_N10000_local.txt) (file thật đầu tiên trong repo, trước đây tài liệu cũ trích dẫn 1 file `profile_output/cprofile_N10000.txt` chưa từng tồn tại). Kết quả local: 0.449s tổng, ~59% suppression / ~40% IoU — tỉ lệ đảo nhẹ so với Colab (do khác phần cứng), nhưng kết luận không đổi. Xem thêm câu hỏi liên quan trong `QA_PREP.md` mục H.

**⚠️ Cần thống nhất trước khi thuyết trình — pptx Slide 4 (CPU Baseline thực tế) dùng lại đúng bộ số "cũ"**: pptx của Tân có 1 bảng ghi rõ nhãn "đo trong proposal" — N=100 → 0.0008s, N=1.000 → 0.0103s, N=10.000 → **0.2846s** — và text đi kèm nói **65% suppression loop / 34% IoU**. Đây chính xác là bộ số mà đoạn trên đã gọi là "chưa từng khớp với output thật nào đã lưu trong repo" và "nhãn bị đảo ngược". Vì pptx tự gắn nhãn rõ ràng "đo trong proposal" (không claim là số Colab đã verify), 2 cách xử lý hợp lý: (1) giữ nguyên bảng đó như số liệu lịch sử/motivation ban đầu (không xoá — pptx đã ghi rõ ngữ cảnh), và khi tới phần kết quả benchmark chính thức thì dùng số Colab verify (2.4918s, 65% IoU/35% suppression) — đã làm theo hướng này trong `OUTLINE_AND_CONTENT.md` Slide 4 + `SCRIPT.md`; hoặc (2) nếu 2 bạn xác nhận số "đo trong proposal" đó thực ra mới là số đúng (ví dụ do cProfile trên Colab từng chạy lại và ra kết quả này thật, chỉ là chưa lưu output), thì cần cập nhật ngược lại bảng "Trạng thái số liệu" ở trên. **Không tự ý chọn 1 trong 2 hướng — 2 bạn cần xác nhận nguồn gốc con số 0.2846s trước khi chốt slide cuối.**

## Nguyên tắc khi chỉnh sửa nội dung

1. **Không tự bịa số liệu.** Chỗ nào chưa đo thật, giữ nguyên nhãn `[CHỜ COLAB]` hoặc `[kỳ vọng, chưa verify]` — nói "chưa đo" vẫn tốt hơn thầy cô phát hiện số sai.
2. **Slide chỉ giữ ý chính, chi tiết kỹ thuật dồn vào `SCRIPT.md`/`QA_PREP.md`.** 2 nhóm khác trong lớp từng bị chê "quá chi tiết cho 1 buổi proposal" dù nội dung đúng — xem `CROSS_GROUP_LESSONS.md`.
3. **Mỗi version (V1/V2/V3) phải có 1 slide kiến trúc riêng, giải thích CÁCH nó song song hoá** — không chỉ nêu số speedup. 1 nhóm khác bị chê thẳng "chưa hiểu quy trình song song hóa" dù đã trình bày.
