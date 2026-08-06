# Giải thích code Seminar 2

## Hợp đồng chung

CPU, V1 và V2 dùng cùng input `boxes (N,4)`, `scores (N,)`, `class_ids (N,)`.
`nms_common.py` kiểm tra shape/dtype, tạo synthetic input cố định và cung cấp
oracle `torchvision` chạy NMS độc lập theo từng class.

Điểm phải nói rõ: hai box chồng lấp nhưng thuộc hai class khác nhau **không**
được suppress lẫn nhau.

## CPU baseline

`run_cpu()` sắp xếp score giảm dần, greedy NMS trong từng class, rồi ghép lại
theo score. Nó cố ý tuần tự để là mốc đúng-đắn và mốc tốc độ cho V1/V2.

`load_raw_yolo_candidates()` chạy YOLOv5 ở tensor level, decode raw prediction
trước NMS và trả về contract chung. Không dùng AutoShape hay `Results.boxes`.

## V1 — full IoU matrix

Mỗi CUDA thread tính một cặp IoU. GPU dựng toàn bộ ma trận `N×N`, copy về
host, sau đó CPU chạy greedy resolver. Đây là bản đơn giản nhất để cho thấy
phần IoU pairwise có thể song song hóa, nhưng tốn bộ nhớ/transfer `O(N²)`.

## V2 — SoA + packed mask

V2 tách box thành bốn mảng `x1, y1, x2, y2` để coalesced read. Kernel ghi
quan hệ suppress vào word `uint64`, giảm dữ liệu so với full float IoU matrix.
CPU vẫn resolve greedy mask theo score rank; đây là giới hạn còn lại của hard
NMS. Batch một class giữ fused launch; batch đa class partition theo class rồi
dùng lại cùng kernel.

## Kiểm thử và số liệu

- CPU/V1/V2: compare với per-class torchvision hard NMS.
- Synthetic benchmark và detector-plus-NMS report là hai scope khác nhau.
- T4 evidence hiện có là trước tái cấu trúc; chỉ dùng số mới sau khi rerun
  current commit trên CUDA. Seminar 2 chỉ giải thích Baseline, V1 và V2.
