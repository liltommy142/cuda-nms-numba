# Thiết kế viết lại bộ nộp Seminar 3 theo giọng sinh viên

## Mục tiêu

Viết lại các phần dành cho người đọc trong bộ Seminar 3 bằng tiếng Việt đơn giản, thân thiện, xưng là “nhóm em”. Người chưa biết CUDA vẫn cần hiểu được nhóm đã làm gì, cách kiểm tra và kết quả thực tế. Việc viết lại chỉ thay đổi cách diễn đạt, không thay đổi sự thật kỹ thuật hay số liệu đã đo.

## Giọng văn và nguyên tắc

- Dùng “nhóm em”, câu ngắn, nói thẳng nhóm đã làm gì và quan sát được gì.
- Giữ những từ quen thuộc như CUDA, NMS, CPU, GPU, benchmark và notebook; giải thích ngắn ở lần đầu nếu người mới cần biết.
- Tránh giọng doanh nghiệp hoặc khẳng định quá mức, ví dụ “primary path”, “submission contract”, “provenance”, “đảm bảo tối ưu” và các cách nói tương tự.
- Phân biệt rõ việc đã chạy/đã kiểm tra với việc chưa làm hoặc chưa đạt. Đặc biệt, mục tiêu `<5 ms/batch` **chưa đạt** và phải được ghi trung thực ở mọi nơi có nhắc tới.

## Phạm vi

Chỉ viết lại nội dung con người đọc trong các nguồn sau:

- `README.md` ở thư mục gốc và README trong `src`.
- Các `explain.md` của baseline, common, v1 và v2.
- Markdown, tiêu đề và nhãn output trong bốn notebook.
- `submission/seminar_3/README.md`, `TEAM_PLAN.md` và `SUBMISSION_MANIFEST.txt`.
- Bản thiết kế này khi đóng gói cùng bài nộp.

Không thay đổi nội dung kỹ thuật của các phần sau:

- Mã nguồn và các câu lệnh chạy.
- Bằng chứng thô dạng JSON/TXT.
- Hash/checksum, cấu hình benchmark, số lần chạy, thông tin GPU và mọi số liệu benchmark.
- Kết luận thực tế, gồm cả việc chưa đạt `<5 ms/batch`.

Nếu một nhãn notebook được kiểm tra tự động, giữ nguyên phần máy đọc được hoặc cập nhật kiểm tra tương ứng; phần giải thích tiếng Việt vẫn phải rõ ràng cho người đọc.

## Cách thực hiện và kiểm tra

1. Sửa nguồn gốc trong repository, không chỉ sửa bản ZIP hoặc thư mục đã giải nén.
2. Viết lại từng tài liệu theo cùng giọng “nhóm em”; so sánh từng số liệu với JSON/TXT thô để không đổi hay tô hồng kết quả.
3. Chạy lại bốn notebook để Markdown, tiêu đề và output hiển thị khớp với nội dung nguồn.
4. Chạy test hiện có và bước kiểm tra dữ liệu; xác nhận notebook và dữ liệu bằng chứng vẫn hợp lệ.
5. Tạo lại hash/checksum và ZIP từ nguồn đã sửa.
6. Giải nén ZIP mới vào vị trí sạch, xác nhận nội dung bản giải nén giống gói ZIP, rồi đồng bộ thư mục giải nén mà người dùng đang dùng bằng bản đã kiểm tra.

## Tiêu chí hoàn thành

- Người không rành CUDA có thể đọc và hiểu nhóm làm gì, cách chạy lại và kết quả chính.
- Cách xưng “nhóm em” và giọng văn thân thiện, không-corporate nhất quán trong toàn bộ phạm vi.
- Không thay đổi mã, lệnh, bằng chứng JSON/TXT, hash, cấu hình benchmark hoặc số liệu; không có tuyên bố vượt quá bằng chứng, và trạng thái `<5 ms/batch` vẫn là chưa đạt.
- Bốn notebook chạy lại thành công; test và kiểm tra dữ liệu đều đạt.
- Checksum hợp lệ; ZIP mới, bản giải nén mới và thư mục giải nén của người dùng đồng nhất về nội dung.
