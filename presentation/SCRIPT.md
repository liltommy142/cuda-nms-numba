# Kịch bản thuyết trình (nói theo ý, không học thuộc từng chữ)

> Đi kèm [`OUTLINE_AND_CONTENT.md`](OUTLINE_AND_CONTENT.md) — **15 slide**, khớp đúng thứ tự bản pptx thật `Slide_Proposal.pptx` (đã đối chiếu, xem ghi chú đầu file kia và `README.md`). Tổng thời lượng ước tính: ~8-9 phút (dài hơn bản cũ ~6-7 phút vì giờ thuyết trình đủ 15 slide thay vì 10, có thêm slide lưu đồ Greedy, roadmap, và 2 slide "hạn chế" của V1/V2). Mỗi slide có câu mở đầu tín hiệu chuyển ý rõ ràng — đây là điểm sửa trực tiếp từ bài học "Nhóm 9 — trình bày chưa rõ ràng" trong `CROSS_GROUP_LESSONS.md`.
>
> Nếu bản pptx dùng để trình bày **chưa được bổ sung** slide "Kết quả đo thật / Đang ở đâu / Mục tiêu / Phân công" (xem mục "Slide bổ sung" cuối `OUTLINE_AND_CONTENT.md`), thì phần lời cho các slide đó (đánh số 16-19 dưới đây) cần chèn thêm thủ công vào cuối bài nói, sau Slide 15, trước khi nhận câu hỏi — đừng bỏ sót vì đây là phần giám khảo hay hỏi nhất (số liệu thật + trạng thái + phân công).

---

## Slide 1 — Mở đầu (~20s)

"Xin chào thầy/cô và các bạn. Nhóm em là Nhóm 11, đề tài A4: tăng tốc thuật toán Non-Maximum Suppression bằng GPU, dùng CUDA thông qua Numba, kèm thêm Matrix NMS. Em là [tên], cùng làm với [tên bạn còn lại] — phần tên đầy đủ nhóm em để ở slide cuối cùng."

## Slide 2 — Vấn đề (~30s)

"Trước khi nói về giải pháp, nhóm em muốn nói rõ vấn đề đang giải quyết. Các mô hình object detection như YOLO sinh ra hàng nghìn box ứng viên cho mỗi ảnh — rất nhiều box trong đó trùng lặp lên cùng một vật thể. NMS có nhiệm vụ lọc bỏ các box trùng, chỉ giữ box tốt nhất. Vấn đề là NMS truyền thống chạy tuần tự, độ phức tạp O(n²)."

## Slide 3 — NMS Greedy: thuật toán hoạt động thế nào (~25s)

"Đây là lưu đồ của Greedy NMS truyền thống, để nhóm em và cả lớp cùng nhìn thống nhất trước khi vào phần kiến trúc GPU. Sắp xếp box theo điểm tin cậy giảm dần, chọn box điểm cao nhất, thêm vào keep list, rồi duyệt qua từng box còn lại — box nào chồng lấp (IoU) vượt ngưỡng với box vừa chọn thì bị loại, không thì giữ lại cho vòng sau. Lặp lại đến khi hết box. Vòng lặp 'duyệt từng box còn lại' này chính là phần tuần tự nhóm em sẽ nói kỹ ở 2 slide tiếp theo."

## Slide 4 — CPU Baseline thực tế (~30s)

"Nhóm em đã profile thật thuật toán này. Khi số box lên tới 1.000, bắt đầu thấy độ trễ rõ rệt. Khi lên 10.000, hệ thống nghẽn hẳn: khoảng 65% thời gian nằm ở vòng lặp khử trùng lặp tuần tự, 34% còn lại là hàm tính IoU. Đây là số đo ở giai đoạn viết proposal ban đầu, dùng để minh hoạ vì sao cần tăng tốc — phần kết quả benchmark chính thức trên Colab T4 sau này có bộ số khác (nhanh hơn về tỉ lệ tuyệt đối do khác điều kiện đo), nhóm em sẽ nói rõ ở phần kết quả để tránh nhầm lẫn nếu thầy/cô so 2 bộ số."

## Slide 5 — Thách Thức (~30s)

"Nhìn kỹ thuật toán, bước quyết định triệt tiêu mang tính tuần tự nghiêm ngặt: hộp B có được giữ lại hay không phụ thuộc trực tiếp vào việc hộp A điểm cao hơn đã bị xoá hay được giữ trước đó. Đây chính là thử thách xuyên suốt cả 3 phiên bản GPU nhóm em làm — mỗi phiên bản tấn công phần tuần tự này theo một cách khác nhau."

## Slide 6 — Roadmap (~15s)

"Đây là lộ trình tổng quan trước khi đi vào chi tiết: từ CPU baseline làm mốc 1 lần, GPU V1 tăng tốc ban đầu 5 đến 10 lần, GPU V2 tối ưu thêm lên khoảng 15 lần, và GPU V3 — phiên bản đổi hẳn thuật toán — đạt mức 30 đến 80 lần. Nhóm em sẽ đi qua từng bước một."

## Slide 7 — Vì sao GPU phù hợp (~30s)

"Bài toán có 2 nửa rất khác nhau. Tính IoU giữa các cặp box hoàn toàn độc lập với nhau — bài toán 'song song hoàn hảo', rất hợp để giao cho hàng nghìn thread GPU cùng lúc: với N box, có thể phân N² thread, mỗi thread lo đúng 1 cặp. Nhưng quyết định giữ hay loại box lại có tính tuần tự như slide trước đã nói. Đây chính là sợi chỉ xuyên suốt cả 3 phiên bản GPU."

## Slide 8 — Kiến trúc V1: Naive Parallel IoU Matrix Kernel (~40s)

"Bắt đầu với V1 — naive nhất. Ý tưởng: giao cho N nhân N thread GPU, mỗi thread tính đúng 1 cặp IoU giữa box i và box j — toàn bộ ma trận IoU N×N tính xong chỉ trong 1 lần gọi kernel, vì các cặp hoàn toàn độc lập, không cần đồng bộ hoá. Sau khi có ma trận này, phần suppression vẫn chạy tuần tự trên CPU — nhưng khác CPU baseline ở chỗ: giờ chỉ là tra bảng có sẵn, không phải tính lại IoU mỗi vòng lặp."

## Slide 9 — Hạn chế của V1 → chuyển sang V2 (~25s)

"V1 còn 2 nút thắt. Một, phải tải cả ma trận N×N về CPU qua đường truyền PCIe, dung lượng tăng theo N bình phương — ở N=10.000 là khoảng 400MB. Hai, vòng lặp loại bỏ tuần tự vẫn còn chạy trên CPU. Cần tiếp tục cải tiến với GPU v2: batched NMS và tối ưu phần cứng."

## Slide 10 — Kiến trúc V2: Batched NMS & Hardware Optimization (~50s)

"V2 tấn công đúng 2 điểm yếu đó của V1. Thứ nhất, về cách đọc bộ nhớ: V1 lưu box dạng mảng gộp 4 toạ độ, khiến các thread trong cùng một warp đọc bộ nhớ không liền mạch. V2 tách thành 4 mảng riêng x1, y1, x2, y2 — giờ các thread liền kề đọc đúng các ô nhớ liền kề nhau, gom lại thành 1 lần đọc hiệu quả thay vì nhiều lần. Thứ hai, về suppression: V2 gom box thành từng khối 64 và dùng parallel reduction để GPU tự nén luôn kết quả 'ai suppress ai' thành bitmask 64-bit — giảm dung lượng truyền đi khoảng 64 lần so với V1. Vòng lặp trên CPU vẫn còn một lần cho mỗi thứ hạng, nhưng giờ mỗi lần chỉ cần OR hai mảng rất ngắn thay vì so sánh cả một hàng dài như V1. Lưu ý: chữ 'batched' ở đây là gom nhóm 64 box để nén bitmask, khác với batch size 32 theo catalog đề tài — phần batch size 32 xử lý nhiều ảnh cùng lúc nhóm em chưa làm, sẽ nói rõ ở phần 'đang ở đâu'."

    *(Nếu đã có số Colab thật, chèn: "Đo thật trên Colab, V2 nhanh hơn CPU X lần tại N=10.000." Nếu chưa, nói: "Phần này code đã xong và có bộ test riêng, nhóm em đang chờ chạy benchmark thật trên Colab.")*

## Slide 11 — Hạn chế của V2 → lý do cần đổi thuật toán (~25s)

"Nhưng V2 vẫn chưa giải quyết xong bài toán gốc. Thứ nhất, nó vẫn mang bản chất Greedy NMS: dù đã gom cụm và tối ưu phần cứng, quyết định giữ hay xoá box B vẫn phụ thuộc vào việc box điểm cao hơn đã bị xoá hay chưa. Thứ hai, việc dựng mặt nạ triệt tiêu bằng parallel reduction chỉ giảm độ trễ, chứ chưa triệt tiêu được tư duy so sánh tuần tự của thuật toán gốc. Đây là lý do V3 phải đổi hẳn thuật toán, không chỉ tối ưu thêm phần cứng."

## Slide 12 — Kiến trúc V3: Matrix NMS (~50s)

"V3 đi xa hơn — không chỉ tối ưu cách làm, mà đổi hẳn thuật toán. NMS truyền thống là 'hard suppression': giữ hoặc loại dứt khoát, và phải làm theo đúng thứ tự điểm số nên buộc phải tuần tự. V3 dùng Matrix NMS của Wang và cộng sự năm 2020 — 'soft suppression': thay vì loại hẳn, mỗi box tự giảm dần điểm số theo mức độ chồng lấp. Ba ưu điểm chính: tốc độ vượt trội, độ chính xác cao hơn vì không xoá nhầm box, và không tốn thêm tài nguyên huấn luyện vì đây chỉ là bước hậu xử lý. Cách làm gồm 2 bước, cả hai đều chạy song song hoàn toàn cho mọi box cùng lúc, không còn vòng lặp nào chạy trên CPU nữa: bước một, mỗi box tự tính độ chồng lấp lớn nhất với các box điểm cao hơn nó; bước hai, mỗi box dùng con số đó để tự tính hệ số giảm điểm của chính mình."

    *(Nếu đã có số Colab thật, chèn tương tự Slide 10.)*

## Slide 13 — V3 sửa một lỗi thật của Hard NMS (~30s)

"Điều nhóm em muốn nói rõ ngay tại đây, không đợi bị hỏi: V3 không chỉ nhanh hơn mà còn giải quyết được một trường hợp mà hard NMS hay xử lý sai — khi 2 vật thể khác nhau đứng sát nhau, IoU giữa 2 box thật cao dù là 2 vật thể riêng biệt, V1/V2 có thể xoá nhầm 1 trong 2 box đúng. V3 chỉ giảm điểm chứ không xoá hẳn, nên cả 2 box vẫn có thể sống sót nếu điểm còn đủ cao. Đổi lại, vì đổi thuật toán, tập box V3 giữ lại sẽ không còn khớp y hệt như CPU baseline hay V1/V2 nữa — đây là một đánh đổi có chủ đích, không phải lỗi."

## Slide 14 — Vì sao Soft Decay là chìa khoá song song hoá 100% (~30s)

"Tóm lại tại sao Matrix NMS song song hoá được hoàn toàn: một, nó phá vỡ chuỗi phụ thuộc tuần tự — không cần biết box A có bị xoá hay không mới quyết định được box B. Hai, hệ số suy giảm của mỗi box được tính hoàn toàn độc lập, chỉ dựa trên mức chồng lấp lớn nhất với các box điểm cao hơn nó. Ba, về bản chất phép toán này quy về 1 lần duyệt ma trận IoU song song và 1 phép nhân ma trận điểm số — thực thi đồng loạt trên GPU, đạt tốc độ vượt trội 30 đến 80 lần."

## Slide 15 — Kết (~15s)

"Đó là toàn bộ phần trình bày của nhóm em — Lê Quang Tân và Phùng Quốc Tuấn. Cảm ơn thầy/cô và các bạn đã lắng nghe, nhóm em sẵn sàng nhận câu hỏi."

---

## Phần bổ sung (chỉ nói nếu pptx có thêm slide, hoặc chèn bằng lời sau Slide 15)

> Bản pptx 15-slide hiện tại **chưa có slide riêng** cho phần này — xem cảnh báo đầu file và mục "Slide bổ sung" trong `OUTLINE_AND_CONTENT.md`. Nếu chưa kịp thêm slide, nói phần này bằng lời ngay sau Slide 15, trước khi nhận câu hỏi.

### Slide 16 — Kết quả đo thật (~30s)

"Đây là số liệu thật, không phải lý thuyết. GPU V1 đã đo trên Colab T4: ở N=10.000, CPU mất khoảng 2.5 giây, GPU chỉ mất 256 mili giây — nhanh hơn 9.7 lần, khớp 100% với thư viện chuẩn torchvision. Ở N nhỏ như 100, chỉ nhanh 1.2 lần — vì lúc đó chi phí khởi động GPU và truyền dữ liệu lấn át lợi ích song song. GPU V2 và V3 code đã xong, đang trong quá trình đo benchmark thật, nhóm em sẽ cập nhật số liệu ngay khi có."

### Slide 17 — Đang ở đâu (~20s)

"Tổng kết trạng thái hiện tại một cách trung thực: CPU baseline và GPU V1 đã hoàn thành và đo tốc độ thật. GPU V2 và V3 đã viết xong code, có bộ test tự động đi kèm, đang chờ chạy trên Colab để lấy số liệu benchmark thật. Một điểm nhóm em muốn nói thẳng: mục tiêu batch size 32 theo đúng catalog đề tài A4 vẫn chưa được cài đặt ở cả 3 phiên bản — hiện mỗi lần chạy chỉ xử lý một tập box."

### Slide 18 — Mục tiêu (~20s)

"Nhóm đặt 3 mức mục tiêu theo đúng catalog, khớp với roadmap đã chiếu ở đầu bài: 75% nếu chỉ có GPU V1 đúng và đo tốc độ — mốc này đã đạt. 100% nếu thêm GPU V2 đạt từ 15 lần trở lên — code đã xong, đang chờ đo. 125% nếu thêm GPU V3 đạt 30 đến 80 lần, dưới 5 mili giây — cũng đã code xong, đang chờ đo."

### Slide 19 — Phân công (~15s)

"Tuấn phụ trách CPU baseline, bộ test và dựng repo. Tân phụ trách toàn bộ 3 kernel GPU và benchmark. Cả hai đều nắm được toàn bộ code, sẵn sàng trả lời bất kỳ phần nào."

---

## Lưu ý khi trình bày

- Nếu quên số liệu chính xác — nói ước lượng đúng bậc độ lớn ("khoảng 250 mili giây", "tầm 10 lần") vẫn tốt hơn đứng im. Xem thêm mẹo trong `QA_PREP.md`.
- Slide 10/12 có 2 câu thoại tuỳ trạng thái số liệu (đã đo/chưa đo) — chọn đúng câu trước khi trình bày, đừng đọc cả hai.
- Slide 4 có 2 bộ số CPU baseline khác nhau (proposal ban đầu vs Colab verify sau này, xem `OUTLINE_AND_CONTENT.md`) — nói đúng ngữ cảnh từng bộ số, đừng để bị hỏi dồn "vậy số nào đúng" mà không có câu trả lời sẵn.
- Nếu bị ngắt hỏi giữa chừng ở slide kiến trúc (8/10/12) — cứ trả lời trực tiếp rồi quay lại đúng slide, đừng cố nói hết kịch bản rồi mới nhận câu hỏi (bài học Nhóm 9: rõ ràng quan trọng hơn nói hết ý đã chuẩn bị).
