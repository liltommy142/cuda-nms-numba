# GPU V1 — số đo Colab T4 (trích từ output notebook, commit 114df94)

Nguồn: output đã lưu trong `src/gpu_v1.ipynb` tại commit `114df94`, trước khi
notebook được viết lại thành dạng clone-and-run (không còn output).
Khôi phục bằng: `git show 114df94:src/gpu_v1.ipynb`.

Hạn chế đã biết của số này: chạy 1 lần/N, không có warmup lặp lại, không có
median/stddev — khác phương pháp của `benchmarks/run_all.py`. Dùng làm mốc
lịch sử, không dùng thay cho artefact JSON.

## Môi trường

```
(không có trong output notebook)
```

## Kết quả

```
       N |    CPU (s) |   GPU V1 (s) |  Speedup
-----------------------------------------------
     100 |     0.0069 |       0.0057 |     1.2x
    1000 |     0.1513 |       0.0146 |    10.3x
   10000 |     2.4918 |       0.2557 |     9.7x
```
