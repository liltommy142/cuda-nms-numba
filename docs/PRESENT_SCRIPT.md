# Kịch bản thuyết trình (học thuộc rồi nói)

> Đi kèm [SLIDE_CONTENT.md](SLIDE_CONTENT.md) — 9 slide, mỗi slide chỉ có hình + vài chữ, **toàn bộ nội dung nói nằm ở đây**. Tổng thời lượng ước tính: ~4-5 phút.

---

### Slide 1 — Mở đầu (~20s)
"Xin chào thầy/cô và các bạn, nhóm em là nhóm 11, đề tài A4: tăng tốc thuật toán Non-Maximum Suppression bằng GPU, dùng CUDA thông qua Numba. Em là [tên], cùng làm với [tên bạn còn lại]."

### Slide 2 — Vấn đề (~30s)
"Các mô hình object detection như YOLO, SSD sinh ra hàng nghìn box ứng viên cho mỗi ảnh — rất nhiều trong số đó trùng lặp lên cùng một vật thể. NMS có nhiệm vụ lọc bỏ các box trùng, chỉ giữ lại box tốt nhất. Vấn đề là: NMS truyền thống chạy tuần tự, độ phức tạp O(n²) — càng nhiều box, thời gian xử lý càng tăng theo cấp số nhân, trở thành nút thắt cổ chai khi suy luận thời gian thực."

*(Số liệu dự phòng nếu cần dẫn chứng — không bắt buộc nói: đã profile bằng cProfile, N=10.000 box mất 0.289 giây, trong đó hơn 99% thời gian là ở chính bước NMS.)*

### Slide 3 — Vì sao dùng GPU (~30s)
"Nhìn kỹ thuật toán, có 2 phần rất khác nhau. Phần tính độ chồng lấp IoU giữa các cặp box thì hoàn toàn độc lập với nhau — đây là bài toán 'song song hoàn hảo', rất hợp để giao cho hàng nghìn thread GPU cùng lúc. Nhưng phần quyết định giữ hay loại box lại có tính tuần tự — box sau phụ thuộc quyết định của box trước. Đây chính là thử thách cốt lõi mà cả đồ án xoay quanh: làm sao biến phần tuần tự đó thành song song."

### Slide 4 — Lộ trình (~30s)
"Nhóm em chia lộ trình thành 4 bước. CPU baseline làm mốc so sánh. GPU V1 — song song hoá phần tính IoU, dự kiến nhanh 5 đến 10 lần. GPU V2 — thêm kỹ thuật parallel reduction để song song hoá luôn cả phần suppression, mục tiêu 15 lần trở lên. GPU V3, mục tiêu vượt trội, áp dụng thuật toán Matrix NMS để loại bỏ hoàn toàn phần tuần tự, kỳ vọng 30 đến 80 lần."

### Slide 5 — Kết quả thật đã đo (~30s)
"Đây không phải số liệu lý thuyết — nhóm em đã cài đặt xong GPU V1 và đo thật trên GPU T4 của Google Colab. Ở N=10.000 box, GPU nhanh hơn CPU **9.7 lần**, đúng như dự đoán ban đầu. Kết quả GPU cũng được đối chiếu và khớp 100% với thư viện chuẩn `torchvision`."

*(Nếu được hỏi thêm: ở N=100 chỉ nhanh 1.2 lần, vì lúc dữ liệu nhỏ thì chi phí khởi động GPU và truyền dữ liệu lấn át lợi ích song song.)*

### Slide 6 — Đang ở đâu (~20s)
"Hiện tại nhóm em đã hoàn thành CPU baseline và GPU V1, đúng tiến độ đề ra. GPU V2 và V3 đang trong kế hoạch, sẽ triển khai ở 2-3 tuần tới."

### Slide 7 — Mục tiêu (~20s)
"Nhóm đặt 3 mức mục tiêu: 75% nếu chỉ hoàn thành V1, 100% nếu đạt V2 với tốc độ từ 15 lần trở lên, và 125% nếu hoàn thành được V3, đạt 30 đến 80 lần, xử lý xong 10.000 box trong dưới 5 mili giây."

### Slide 8 — Phân công (~15s)
"[Tuấn] phụ trách dựng repo, CPU baseline và bộ test ban đầu. [Tân] phụ trách kernel GPU V1 và phần test cho GPU. Cả 2 đều nắm được toàn bộ code, sẵn sàng trả lời bất kỳ phần nào."

### Slide 9 — Kết (~10s)
"Đó là toàn bộ proposal của nhóm em. Cảm ơn thầy/cô và các bạn đã lắng nghe, nhóm em sẵn sàng nhận câu hỏi."

---

## Q&A dự phòng — chuẩn bị sẵn, không cần lên slide

**Hỏi: Đã đạt mục tiêu <5ms chưa?**
→ "Chưa ạ, mục tiêu đó thuộc về GPU V3 chưa triển khai. Hiện GPU V1 đo được 255.7 mili giây tại N=10.000 — đây là kết quả bình thường ở giai đoạn naive, cách mục tiêu cuối khoảng 51 lần, đúng như lộ trình 3 bước đã đề ra."

**Hỏi: Sao lại chọn batch size 32?**
→ "Đây là mốc do catalog đề tài A4 của môn quy định sẵn, nhóm em bám theo để so sánh được với chuẩn chấm điểm, không phải nhóm tự chọn."

**Hỏi: Vì sao không dùng CUDA C/C++ mà dùng Numba?**
→ "Đây là ràng buộc của môn học — Numba `@cuda.jit` là công cụ GPU chính thức được yêu cầu, giúp giữ code Python thuần, dễ đối chiếu trực tiếp với bản CPU NumPy."

**Hỏi: Làm sao biết GPU tính đúng, không sai số?**
→ "Nhóm so sánh kết quả GPU với `torchvision.ops.nms` — thư viện đã được kiểm chứng rộng rãi — trong dung sai 1e-4, và dùng thuật toán sắp xếp ổn định (stable sort) ở cả 2 bên để tránh lệch kết quả do trùng điểm số."

**Hỏi: Rủi ro lớn nhất của nhóm là gì?**
→ "GPU V3 là phần khó nhất và xếp lịch cuối cùng, nên nếu tiến độ V1/V2 chậm thì V3 có thể không kịp. Nhóm đã tính trước phương án dự phòng: chỉ cần hoàn thành V1/V2 là đã đạt mốc 100%, V3 chỉ là điểm cộng thêm."
