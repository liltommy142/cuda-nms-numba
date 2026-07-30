# Trạng thái đã xác minh

Tài liệu này chỉ ghi nhận kết quả đã chạy hoặc có thể đối chiếu trực tiếp với
mã nguồn. Nó không dùng các con số tốc độ dự kiến trong slide/ghi chú nháp.

## Môi trường local lần chạy gần nhất

| Thuộc tính | Giá trị |
|---|---|
| Python | 3.11.9 |
| NumPy | 1.26.4 |
| Numba | 0.66.0 |
| CUDA khả dụng | Không |
| Hệ điều hành | Windows 10 |

## Kết quả kiểm thử

Lệnh:

```bash
python -m pytest tests -v
```

Kết quả: **12 passed, 36 skipped**.

Các test CPU baseline, test đơn vị IoU và CPU oracle của Matrix NMS
(`matrix_nms_reference`) đã pass. Tất cả test GPU bị skip vì
`numba.cuda.is_available()` trả về `False`; điều này không phải bằng chứng
V1/V2/V3 chạy đúng trên GPU.

## Benchmark CPU local

Lệnh:

```bash
python benchmarks/run_all.py --versions cpu v1 v2 v3 --repeats 7 --warmup 2
```

| Số box | CPU median | Độ lệch chuẩn |
|---:|---:|---:|
| 100 | 1.134 ms | 0.039 ms |
| 1.000 | 16.502 ms | 1.246 ms |
| 10.000 | 382.775 ms | 3.804 ms |

V1/V2/V3 không chạy ở lần này vì không có CUDA. Không được suy ra speedup GPU
từ bảng trên.

## Phạm vi hiện có

- `run_cpu`, GPU V1 và GPU V2 là hard/greedy NMS; V1/V2 được thiết kế để trả về
  cùng tập index với CPU.
- GPU V3 là Matrix NMS (soft suppression), không được kỳ vọng có cùng output
  với greedy NMS. `matrix_nms_reference()` là CPU oracle để kiểm tra quy ước
  Matrix NMS mà implementation hiện dùng.
- Mọi implementation hiện xử lý một tập box mỗi lần gọi. Fused CUDA batch
  kernel chưa được cài đặt, vì vậy chưa đáp ứng mục tiêu catalog A4 về độ trễ
  batch-size 32.

## Điều kiện để thay trạng thái thành “GPU verified”

1. Chạy `pytest tests -v` trên máy có CUDA và lưu full output.
2. Chạy benchmark lặp lại bằng `benchmarks/run_all.py`, xuất JSON.
3. Ghi GPU, CUDA driver/runtime, Numba và cấu hình benchmark cùng kết quả.
4. Chỉ sau đó mới cập nhật speedup/latency vào README hoặc slide.
