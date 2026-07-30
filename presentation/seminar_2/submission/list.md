# Submission list — Seminar 2

Đánh dấu từng mục chỉ khi file/link đã mở được từ một máy khác hoặc đã upload
lên đúng nơi giảng viên yêu cầu.

## 1. GitHub repository — bắt buộc xác nhận trước

- [ ] Repository public hoặc thầy đã được cấp quyền xem.
- [ ] Link đã gửi cho thầy/Moodle/form: <https://github.com/liltommy142/cuda-nms-numba>
- [ ] Ghi commit hash cuối cùng đã dùng để chạy test/benchmark: `________________`
- [ ] Mở link ở cửa sổ ẩn danh để chắc chắn thầy xem được.
- [ ] README có lệnh chạy, dependencies và link tới tài liệu Seminar 2.

## 2. Code và reproducibility

- [ ] Push toàn bộ code cuối: `src/`, `tests/`, `benchmarks/`, `requirements.txt`.
- [ ] Không commit token, notebook secret hay file cache lớn.
- [ ] `python -m pytest tests -v` chạy không có failed trên CUDA.
- [ ] Lưu full log CUDA ở `../evidence/pytest_<gpu>.txt`.
- [ ] Lưu benchmark JSON raw samples ở `../evidence/benchmark_<gpu>.json`.
- [ ] Lưu batch-32 report ở `../evidence/batch32_<gpu>.json`.

## 3. Slide và report

- [ ] PPTX final đã thay toàn bộ số `[CHỜ COLAB]` bằng evidence thật hoặc ghi rõ pending.
- [ ] PDF export từ PPTX final.
- [ ] Report/PDF có: problem, design V1/V2/V3, correctness, benchmark,
      environment, limitation và contribution.
- [ ] Mọi số tốc độ có GPU model, version và tên artifact nguồn.
- [ ] V3 được ghi là Matrix NMS, không claim khớp greedy torchvision NMS.

## 4. Demo và nộp bài

- [ ] Demo Colab chạy từ commit cuối, hoặc có video/screenshot fallback.
- [ ] Hai thành viên đã rehearsal và biết giải thích V1/V2/V3.
- [ ] Upload đúng nơi giảng viên yêu cầu: `________________`.
- [ ] Ghi thời hạn nộp: `________________`.
- [ ] Sau upload, mở lại file/link đã nộp để kiểm tra quyền truy cập.

## Kết luận kiểm tra Seminar 1

Remote hiện tại là <https://github.com/liltommy142/cuda-nms-numba>. Proposal
DOCX không chứa GitHub URL theo phần nội dung đã trích xuất, nên chưa có bằng
chứng rằng link này từng được gửi cho thầy. Hãy gửi lại link ở mục 1 ngay cả
nếu không chắc có gửi trước đó; gửi lại an toàn hơn là để thiếu quyền truy cập.
