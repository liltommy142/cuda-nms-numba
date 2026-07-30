# Runbook xác minh GPU

Mục tiêu là tạo một phép đo có thể lặp lại cho V1, V2 và V3 trên CUDA GPU. Mọi
con số được đưa vào báo cáo/slide phải đi kèm file JSON từ bước 3.

## 1. Chuẩn bị runtime

Trên Google Colab, chọn **Runtime → Change runtime type → T4 GPU**. Clone repo
hoặc mở terminal trong repo, sau đó cài dependency tương thích với notebook.

```bash
pip install -r requirements.txt
python -c "from numba import cuda; print(cuda.is_available()); print(cuda.get_current_device().name)"
```

Chỉ đi tiếp nếu dòng đầu là `True`.

## 2. Xác minh correctness trước

```bash
pytest tests -v
```

Yêu cầu tối thiểu: không có `failed`. Lưu toàn bộ output, đặc biệt các test
V1/V2 ở `N=10_000` và test V3 đối chiếu `matrix_nms_reference`.

## 3. Đo lặp lại và lưu artefact

```bash
python benchmarks/run_all.py --versions cpu v1 v2 v3 \
  --n 100 1000 10000 --warmup 2 --repeats 7 \
  --json benchmarks/results/<gpu-name>.json
```

Runner đã warm-up trước khi đo và lưu raw samples, median, min, mean, stddev,
Python/NumPy/Numba/GPU/compute capability. Không thay JSON bằng bảng gõ tay.

## 4. Diễn giải đúng

- V1/V2: so tốc độ với CPU greedy NMS, đồng thời yêu cầu output đúng theo test.
- V3: so tốc độ với CPU greedy chỉ như một trade-off thuật toán; không tuyên bố
  “khớp greedy NMS”. V3 cần được kiểm tra với CPU Matrix-NMS oracle.
- Runner hiện đo một tập box mỗi lần gọi. Không dùng nó làm bằng chứng của
  fused batch kernel hay latency batch-32; tính năng đó chưa được cài đặt.

## 5. Nếu fail

- `cuda.is_available() == False`: kiểm tra GPU runtime/driver trước khi sửa code.
- V1/V2 fail correctness: lưu seed, N, threshold và output chênh lệch; không
  benchmark tiếp.
- V3 fail oracle: kiểm tra method (`linear`/`gaussian`), `sigma`, score
  threshold và sai số `fastmath` sát ranh giới threshold.
- Benchmark dao động mạnh: tăng repeats, đóng process nền và báo median/stddev
  thay vì chỉ chọn lần nhanh nhất.
