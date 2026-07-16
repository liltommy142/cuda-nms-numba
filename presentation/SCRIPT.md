# Kịch bản thuyết trình (nói theo ý, không học thuộc từng chữ)

> Đi kèm [`OUTLINE_AND_CONTENT.md`](OUTLINE_AND_CONTENT.md) — 10 slide. Tổng thời lượng ước tính: ~6-7 phút (dài hơn bản cũ ~4-5 phút vì giờ có 3 kiến trúc GPU thay vì 1). Mỗi slide có câu mở đầu tín hiệu chuyển ý rõ ràng — đây là điểm sửa trực tiếp từ bài học "Nhóm 9 — trình bày chưa rõ ràng" trong `CROSS_GROUP_LESSONS.md`.

---

## Slide 1 — Mở đầu (~20s)

"Xin chào thầy/cô và các bạn. Nhóm em là Nhóm 11, đề tài A4: tăng tốc thuật toán Non-Maximum Suppression bằng GPU, dùng CUDA thông qua Numba. Em là [tên], cùng làm với [tên bạn còn lại]."

## Slide 2 — Vấn đề (~30s)

"Trước khi nói về giải pháp, nhóm em muốn nói rõ vấn đề đang giải quyết. Các mô hình object detection như YOLO sinh ra hàng nghìn box ứng viên cho mỗi ảnh — rất nhiều box trong đó trùng lặp lên cùng một vật thể. NMS có nhiệm vụ lọc bỏ các box trùng, chỉ giữ box tốt nhất. Vấn đề là NMS truyền thống chạy tuần tự, độ phức tạp O(n²). Nhóm em đã profile thật bằng cProfile ở N=10.000 box: khoảng 65% thời gian là tính IoU giữa các cặp box, 35% còn lại là phần vòng lặp suppression tuần tự — tức là chính phần tính toán nặng nhất (IoU) lại đúng là phần có thể song song hoá được, không phải phần đọc dữ liệu mới là nút thắt cổ chai."

## Slide 3 — Vì sao GPU phù hợp (~30s)

"Nhìn kỹ thuật toán, có 2 phần rất khác nhau. Tính IoU giữa các cặp box hoàn toàn độc lập với nhau — bài toán 'song song hoàn hảo', rất hợp để giao cho hàng nghìn thread GPU cùng lúc. Nhưng quyết định giữ hay loại box lại có tính tuần tự — box sau phụ thuộc quyết định của box trước. Đây chính là sợi chỉ xuyên suốt cả 3 phiên bản GPU nhóm em làm: mỗi phiên bản tấn công phần tuần tự này theo một cách khác nhau, không phải chỉ tối ưu tốc độ đơn thuần."

## Slide 4 — Kiến trúc V1 (~40s)

"Bắt đầu với V1 — naive nhất. Ý tưởng: giao cho N nhân N thread GPU, mỗi thread tính đúng 1 cặp IoU giữa box i và box j — toàn bộ ma trận IoU N×N tính xong chỉ trong 1 lần gọi kernel, vì các cặp hoàn toàn độc lập, không cần đồng bộ hoá. Sau khi có ma trận này, phần suppression vẫn chạy tuần tự trên CPU — nhưng khác CPU baseline ở chỗ: giờ chỉ là tra bảng có sẵn, không phải tính lại IoU mỗi vòng lặp. Bottleneck còn lại của V1: phải tải cả ma trận N×N về CPU qua đường truyền PCIe, dung lượng tăng theo N bình phương — ở N=10.000 là khoảng 400MB."

## Slide 5 — Kiến trúc V2 (~50s)

"V2 tấn công đúng 2 điểm yếu đó của V1. Thứ nhất, về cách đọc bộ nhớ: V1 lưu box dạng mảng gộp 4 toạ độ, khiến các thread trong cùng một warp đọc bộ nhớ không liền mạch. V2 tách thành 4 mảng riêng x1, y1, x2, y2 — giờ các thread liền kề đọc đúng các ô nhớ liền kề nhau, gom lại thành 1 lần đọc hiệu quả thay vì nhiều lần. Thứ hai, về suppression: thay vì tải cả ma trận IoU dạng số thực về CPU, V2 để GPU tự nén luôn kết quả 'ai suppress ai' thành các số nguyên 64-bit dạng bitmask — giảm dung lượng truyền đi khoảng 64 lần so với V1. Vòng lặp trên CPU vẫn còn một lần cho mỗi thứ hạng, nhưng giờ mỗi lần chỉ cần OR hai mảng rất ngắn thay vì so sánh cả một hàng dài như V1."

    *(Nếu đã có số Colab thật, chèn: "Đo thật trên Colab, V2 nhanh hơn CPU X lần tại N=10.000." Nếu chưa, nói: "Phần này code đã xong và có bộ test riêng, nhóm em đang chờ chạy benchmark thật trên Colab.")*

## Slide 6 — Kiến trúc V3 (~50s)

"V3 đi xa hơn — không chỉ tối ưu cách làm, mà đổi hẳn thuật toán. NMS truyền thống là 'hard suppression': giữ hoặc loại dứt khoát, và phải làm theo đúng thứ tự điểm số nên buộc phải tuần tự. V3 dùng Matrix NMS của Wang và cộng sự năm 2020 — 'soft suppression': thay vì loại hẳn, mỗi box tự giảm dần điểm số theo mức độ chồng lấp. Cách làm gồm 2 bước, cả hai đều chạy song song hoàn toàn cho mọi box cùng lúc, không còn vòng lặp nào chạy trên CPU nữa: bước một, mỗi box tự tính độ chồng lấp lớn nhất với các box điểm cao hơn nó; bước hai, mỗi box dùng con số đó để tự tính hệ số giảm điểm của chính mình. Điều nhóm em muốn nói rõ ngay tại đây, không đợi bị hỏi: vì đổi thuật toán, tập box V3 giữ lại sẽ không còn khớp y hệt như CPU baseline hay V1/V2 nữa — đây là một đánh đổi có chủ đích của thuật toán, không phải lỗi."

    *(Nếu đã có số Colab thật, chèn tương tự Slide 5.)*

## Slide 7 — Kết quả đo thật (~30s)

"Đây là số liệu thật, không phải lý thuyết. GPU V1 đã đo trên Colab T4: ở N=10.000, CPU mất khoảng 2.5 giây, GPU chỉ mất 256 mili giây — nhanh hơn 9.7 lần, khớp 100% với thư viện chuẩn torchvision. Ở N nhỏ như 100, chỉ nhanh 1.2 lần — vì lúc đó chi phí khởi động GPU và truyền dữ liệu lấn át lợi ích song song. GPU V2 và V3 code đã xong, đang trong quá trình đo benchmark thật, nhóm em sẽ cập nhật số liệu ngay khi có."

## Slide 8 — Đang ở đâu (~20s)

"Tổng kết trạng thái hiện tại một cách trung thực: CPU baseline và GPU V1 đã hoàn thành và đo tốc độ thật. GPU V2 và V3 đã viết xong code, có bộ test tự động đi kèm, đang chờ chạy trên Colab để lấy số liệu benchmark thật. Một điểm nhóm em muốn nói thẳng: mục tiêu batch size 32 theo đúng catalog đề tài A4 vẫn chưa được cài đặt ở cả 3 phiên bản — hiện mỗi lần chạy chỉ xử lý một tập box."

## Slide 9 — Mục tiêu (~20s)

"Nhóm đặt 3 mức mục tiêu theo đúng catalog: 75% nếu chỉ có GPU V1 đúng và đo tốc độ — mốc này đã đạt. 100% nếu thêm GPU V2 đạt từ 15 lần trở lên — code đã xong, đang chờ đo. 125% nếu thêm GPU V3 đạt 30 đến 80 lần, dưới 5 mili giây — cũng đã code xong, đang chờ đo."

## Slide 10 — Phân công & Kết (~15s)

"Tuấn phụ trách CPU baseline, bộ test và dựng repo. Tân phụ trách toàn bộ 3 kernel GPU và benchmark. Cả hai đều nắm được toàn bộ code, sẵn sàng trả lời bất kỳ phần nào. Đó là toàn bộ phần trình bày của nhóm em, cảm ơn thầy/cô và các bạn đã lắng nghe, nhóm em sẵn sàng nhận câu hỏi."

---

## Lưu ý khi trình bày

- Nếu quên số liệu chính xác — nói ước lượng đúng bậc độ lớn ("khoảng 250 mili giây", "tầm 10 lần") vẫn tốt hơn đứng im. Xem thêm mẹo trong `QA_PREP.md`.
- Slide 5/6 có 2 câu thoại tuỳ trạng thái số liệu (đã đo/chưa đo) — chọn đúng câu trước khi trình bày, đừng đọc cả hai.
- Nếu bị ngắt hỏi giữa chừng ở slide kiến trúc (4/5/6) — cứ trả lời trực tiếp rồi quay lại đúng slide, đừng cố nói hết kịch bản rồi mới nhận câu hỏi (bài học Nhóm 9: rõ ràng quan trọng hơn nói hết ý đã chuẩn bị).
