# 🧭 Mục lục tổng — cuda-nms-numba

> Trang này là "trang chủ" của toàn bộ tài liệu trong repo — vào đây trước, rồi đi tới đúng file cần đọc theo mục đích của bạn, thay vì lục tung từng thư mục. Coi cả `docs/` + `presentation/` như 1 vault ghi chú liên kết với nhau: mỗi file là 1 "note", các note trỏ qua lại lẫn nhau thay vì lặp lại nội dung.

## Bạn đang muốn làm gì?

| Mục đích | Đi tới đây |
|---|---|
| Mới vào repo, muốn hiểu tổng quan dự án là gì | [`../README.md`](../README.md) (giới thiệu ngắn) → [TECHNICAL_DOCUMENTATION §1](TECHNICAL_DOCUMENTATION.md#1-scope-and-current-status) (phạm vi và trạng thái hiện tại) |
| Muốn tự chạy code/test trên máy mình (không cần AI hỗ trợ) | [`HOW_TO_RUN.md`](HOW_TO_RUN.md) |
| Có CUDA GPU và muốn xác minh đúng quy trình | [mục Colab trong `HOW_TO_RUN.md`](HOW_TO_RUN.md#1-chạy-trên-google-colab-có-gpu-t4-thật) |
| Không rành CUDA/GPU, muốn hiểu khái niệm trước khi đọc code | [`GLOSSARY.md`](GLOSSARY.md) (tra thuật ngữ và khái niệm nền tảng) → [TECHNICAL_DOCUMENTATION §2](TECHNICAL_DOCUMENTATION.md#2-candidate-contract) (hợp đồng dữ liệu chung) |
| Muốn đọc kỹ từng module/kernel | [`TECHNICAL_DOCUMENTATION.md`](TECHNICAL_DOCUMENTATION.md) — contract chung, CPU, V1, V2, V3 và benchmark semantics |
| Muốn so sánh các phiên bản và bottleneck đã biết | [TECHNICAL_DOCUMENTATION §3](TECHNICAL_DOCUMENTATION.md#3-implementations) |
| Tra 1 thuật ngữ cụ thể (IoU, warp, bitmask, decay factor...) | [`GLOSSARY.md`](GLOSSARY.md) |
| Đang chuẩn bị buổi thuyết trình seminar | [`../presentation/README.md`](../presentation/README.md) (điểm bắt đầu của bộ tài liệu thuyết trình) |
| Muốn biết số liệu nào là thật, số nào còn chờ CUDA | [`../presentation/seminar_2/README.md`](../presentation/seminar_2/README.md) |
| Chuẩn bị trả lời Q&A / lo bị hỏi khó | [`../presentation/seminar_2/QA_PREP.md`](../presentation/seminar_2/QA_PREP.md) |
| Cần nộp Seminar 3 (notebook, evidence, team plan, hướng dẫn tái lập) | [`../submission/seminar_3/README.md`](../submission/seminar_3/README.md) |

## Bản đồ toàn bộ tài liệu

```
cuda-nms-numba/
├── README.md                        → giới thiệu ngắn, lệnh chạy nhanh
├── docs/
│   ├── INDEX.md                     → bạn đang ở đây
│   ├── GLOSSARY.md                  → tra thuật ngữ, liên kết ngược vào TECHNICAL_DOCUMENTATION
│   ├── TECHNICAL_DOCUMENTATION.md   → contract, implementation, benchmark và seminar-safe claims
│   └── HOW_TO_RUN.md                → chạy code/test thật, không cần AI
├── presentation/
│   ├── README.md                    → điểm bắt đầu của bộ tài liệu thuyết trình + trạng thái số liệu
│   ├── seminar_1/                   → proposal đã nộp
│   └── seminar_2/                   → deck, outline, script, Q&A và evidence Seminar 2
├── submission/
│   └── seminar_3/                   → bundle nộp: notebook, evidence, plan và checksums
├── src/*.py + src/*.ipynb           → code thật (tổng quan module ở TECHNICAL_DOCUMENTATION §3)
└── tests/                           → correctness + benchmark metadata contracts
```

## Nguyên tắc liên kết trong "vault" này

- Mỗi khái niệm chỉ định nghĩa đầy đủ **một lần** ở [`GLOSSARY.md`](GLOSSARY.md) — chỗ khác chỉ link tới, không định nghĩa lại để tránh 2 bản giải thích lệch nhau theo thời gian.
- Mỗi số liệu benchmark chỉ có **một nguồn thật** — artifact trong [`../presentation/seminar_2/evidence/`](../presentation/seminar_2/evidence/). Nơi khác trích số liệu nên ghi rõ đang trích từ đâu thay vì chép lại số có thể lệch.
- Link giữa các file `.md` dùng đường dẫn tương đối chuẩn (`[chữ](đường/dẫn.md#anchor)`), không dùng cú pháp `[[wikilink]]` — để đảm bảo hiển thị đúng trên GitHub lẫn khi mở cả repo bằng Obsidian.
