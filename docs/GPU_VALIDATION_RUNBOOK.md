# Runbook xác minh GPU

Mục tiêu là tạo phép đo lặp lại cho V1, V2 và V3 trên CUDA GPU. Mọi con số đưa
vào báo cáo/slide phải đi kèm file JSON. Với target catalog batch-size 32,
dùng thêm runner riêng ở bước 4.

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

## 4. Đo V2 batch-size 32

```bash
python benchmarks/run_v2_batch.py --batch-size 32 --n 10000 \
  --warmup 2 --repeats 7 --json benchmarks/results/<gpu-name>_v2_batch32.json
```

Runner này đo end-to-end: host sort, transfer, GPU bitmask kernel, tải mask và
CPU greedy resolution. Nó là phép đo đúng cho V2 batch, không thay thế bảng
single-image của `run_all.py`.

## 5. Diễn giải đúng

- V1/V2: so tốc độ với CPU greedy NMS, đồng thời yêu cầu output đúng theo test.
- V3: so tốc độ với CPU greedy chỉ như một trade-off thuật toán; không tuyên bố
  “khớp greedy NMS”. V3 cần được kiểm tra với CPU Matrix-NMS oracle.
- `run_all.py` đo một tập box mỗi lần gọi. V2 batch phải dùng
  `run_v2_batch.py`; V1/V3 chưa có fused batch implementation.

## 6. Nếu fail

- `cuda.is_available() == False`: kiểm tra GPU runtime/driver trước khi sửa code.
- V1/V2 fail correctness: lưu seed, N, threshold và output chênh lệch; không
  benchmark tiếp.
- V3 fail oracle: kiểm tra method (`linear`/`gaussian`), `sigma`, score
  threshold và sai số `fastmath` sát ranh giới threshold.
- Benchmark dao động mạnh: tăng repeats, đóng process nền và báo median/stddev
  thay vì chỉ chọn lần nhanh nhất.
