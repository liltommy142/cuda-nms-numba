# 🧭 Mục lục tổng — cuda-nms-numba

> Trang này là "trang chủ" của toàn bộ tài liệu trong repo — vào đây trước, rồi đi tới đúng file cần đọc theo mục đích của bạn, thay vì lục tung từng thư mục. Coi cả `docs/` + `presentation/` như 1 vault ghi chú liên kết với nhau: mỗi file là 1 "note", các note trỏ qua lại lẫn nhau thay vì lặp lại nội dung.

## Bạn đang muốn làm gì?

| Mục đích | Đi tới đây |
|---|---|
| Mới vào repo, muốn hiểu tổng quan dự án là gì | [`../README.md`](../README.md) (giới thiệu ngắn) → [TECHNICAL_DOCUMENTATION §1](TECHNICAL_DOCUMENTATION.md#phần-1--phân-tích-dự-án) (phân tích đầy đủ) |
| Muốn tự chạy code/test trên máy mình (không cần AI hỗ trợ) | [`HOW_TO_RUN.md`](HOW_TO_RUN.md) |
| Muốn xem chỉ các kết quả đã chạy/xác minh được | [`VERIFIED_STATUS.md`](VERIFIED_STATUS.md) |
| Có CUDA GPU và muốn xác minh V1/V2/V3 đúng quy trình | [`GPU_VALIDATION_RUNBOOK.md`](GPU_VALIDATION_RUNBOOK.md) |
| Không rành CUDA/GPU, muốn hiểu khái niệm trước khi đọc code | [TECHNICAL_DOCUMENTATION §2](TECHNICAL_DOCUMENTATION.md#phần-2--giải-thích-kỹ-thuật-dành-cho-người-mới) (giải thích từ số 0) → [`GLOSSARY.md`](GLOSSARY.md) (tra thuật ngữ nhanh) |
| Muốn đọc kỹ từng hàm/kernel trong `src/*.py` | [TECHNICAL_DOCUMENTATION §2](TECHNICAL_DOCUMENTATION.md#phần-2--giải-thích-kỹ-thuật-dành-cho-người-mới) — có mục riêng cho từng file: [`cpu_baseline.py`](TECHNICAL_DOCUMENTATION.md#22-cpu_baselinepy--giải-thích-từng-hàm), [`gpu_v1.py`](TECHNICAL_DOCUMENTATION.md#23-gpu_v1py--giải-thích-từng-hàm-và-cuda-kernel), [`gpu_v2.py`](TECHNICAL_DOCUMENTATION.md#25-gpu_v2py--giải-thích-từng-hàm-và-cuda-kernel), [`gpu_v3.py`](TECHNICAL_DOCUMENTATION.md#26-gpu_v3py--giải-thích-từng-hàm-và-cuda-kernel) |
| Muốn xem sơ đồ luồng dữ liệu / độ phức tạp Big-O / các bottleneck đã biết | [TECHNICAL_DOCUMENTATION §3](TECHNICAL_DOCUMENTATION.md#phần-3--tài-liệu-kỹ-thuật-chi-tiết) |
| Tra 1 thuật ngữ cụ thể (IoU, warp, bitmask, decay factor...) | [`GLOSSARY.md`](GLOSSARY.md) |
| Đang chuẩn bị buổi thuyết trình seminar | [`../presentation/README.md`](../presentation/README.md) (điểm bắt đầu của bộ tài liệu thuyết trình) |
| Muốn biết số liệu nào là thật, số nào còn chờ CUDA | [`../presentation/seminar_2/README.md`](../presentation/seminar_2/README.md) |
| Chuẩn bị trả lời Q&A / lo bị hỏi khó | [`../presentation/seminar_2/QA_PREP.md`](../presentation/seminar_2/QA_PREP.md) |
| Muốn biết lớp/giảng viên hay hỏi kiểu gì (từ feedback nhóm khác) | [`../presentation/seminar_2/CROSS_GROUP_LESSONS.md`](../presentation/seminar_2/CROSS_GROUP_LESSONS.md) |

## Bản đồ toàn bộ tài liệu

```
cuda-nms-numba/
├── README.md                        → giới thiệu ngắn, lệnh chạy nhanh
├── docs/
│   ├── INDEX.md                     → bạn đang ở đây
│   ├── GLOSSARY.md                  → tra thuật ngữ, liên kết ngược vào TECHNICAL_DOCUMENTATION
│   ├── TECHNICAL_DOCUMENTATION.md   → tài liệu kỹ thuật đầy đủ (3 phần: phân tích, giải thích, chi tiết)
│   └── HOW_TO_RUN.md                → chạy code/test thật, không cần AI
├── presentation/
│   ├── README.md                    → điểm bắt đầu của bộ tài liệu thuyết trình + trạng thái số liệu
│   ├── seminar_1/                   → proposal đã nộp
│   └── seminar_2/                   → deck, outline, script, Q&A và evidence Seminar 2
├── src/*.py + src/*.ipynb           → code thật (giải thích chi tiết ở TECHNICAL_DOCUMENTATION §2)
└── tests/test_correctness.py        → đối chiếu CPU ↔ GPU V1 ↔ GPU V2 ↔ torchvision
```

## Nguyên tắc liên kết trong "vault" này

- Mỗi khái niệm chỉ định nghĩa đầy đủ **một lần** ở [`GLOSSARY.md`](GLOSSARY.md) — chỗ khác chỉ link tới, không định nghĩa lại để tránh 2 bản giải thích lệch nhau theo thời gian.
- Mỗi số liệu benchmark chỉ có **một nguồn thật** — artifact trong [`../presentation/seminar_2/evidence/`](../presentation/seminar_2/evidence/). Nơi khác trích số liệu nên ghi rõ đang trích từ đâu thay vì chép lại số có thể lệch.
- Link giữa các file `.md` dùng đường dẫn tương đối chuẩn (`[chữ](đường/dẫn.md#anchor)`), không dùng cú pháp `[[wikilink]]` — để đảm bảo hiển thị đúng trên GitHub lẫn khi mở cả repo bằng Obsidian.
