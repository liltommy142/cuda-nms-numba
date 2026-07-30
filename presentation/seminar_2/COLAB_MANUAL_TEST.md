# Test Colab thủ công — chạy theo thứ tự này

Mục tiêu là dừng ngay khi sai correctness, không benchmark một kernel chưa
đúng. Trước khi chạy, bảo đảm commit chứa V2 batch đã được push lên GitHub;
notebook clone repo không thấy thay đổi chỉ nằm trên máy local.

## 0. Tạo runtime và ghi môi trường

1. Colab → `Runtime` → `Change runtime type` → chọn **T4 GPU**.
2. Clone đúng commit, mở terminal/cell và chạy:

```bash
git clone --depth 1 https://github.com/liltommy142/cuda-nms-numba.git
cd cuda-nms-numba
git rev-parse HEAD
nvidia-smi
python - <<'PY'
import numpy, numba
from numba import cuda
print('numpy =', numpy.__version__)
print('numba =', numba.__version__)
print('cuda available =', cuda.is_available())
print('gpu =', cuda.get_current_device().name)
PY
```

Lưu output cùng commit hash. Nếu `cuda.is_available()` là `False`, dừng và
đổi runtime; không sửa code để che lỗi môi trường.

## 1. JIT smoke test — trước full test

Chạy lần lượt V1, V2 và V3 ở N nhỏ. Nếu Numba hiện cài không JIT được kernel,
thử fallback ghi trong `requirements.txt`: `numba==0.59.1` với `numpy<2`, rồi
restart runtime và clone lại repo.

```bash
python src/gpu_v1.py --n 64 --verify
python src/gpu_v2.py --n 64 --verify
python src/gpu_v3.py --n 64 --method gaussian
```

V2 phải chạy thêm boundary + batch:

```bash
python src/gpu_v2.py --n 50 --batch-size 3 --verify
python src/gpu_v2.py --n 100 --batch-size 32
```

`N=50` cố tình tạo block cuối không đủ 64 thread; đây là case bắt buộc cho bản
sửa `cuda.syncthreads()`.

## 2. Full correctness gate

```bash
python -m pytest tests -v | tee presentation/seminar_2/evidence/pytest_t4.txt
```

Chỉ đi tiếp khi không có `failed`. Kiểm tra riêng các nhóm sau trong output:

- CPU match `torchvision.ops.nms`;
- V1/V2 match CPU ở N=50, 200, 1,000, 10,000;
- V2 score ties, nhiều threshold và test batch partial block;
- V3 linear/gaussian match `matrix_nms_reference`.

Nếu fail, lưu full traceback, seed, N, threshold và dừng benchmark version đó.

## 3. Benchmark single-image có JSON

Đây là benchmark công bằng CPU/V1/V2/V3 cho một tập box mỗi lần gọi:

```bash
python benchmarks/run_all.py --versions cpu v1 v2 v3 \
  --n 100 1000 10000 --warmup 2 --repeats 7 \
  --json presentation/seminar_2/evidence/benchmark_t4_single.json \
  | tee presentation/seminar_2/evidence/benchmark_t4_single.txt
```

Trong slide dùng median và standard deviation từ JSON; không chọn lần nhanh
nhất. V3 chỉ là trade-off Matrix NMS, không ghi “V3 match torchvision”.

## 4. Benchmark V2 batch-size 32

CLI V2 hiện có đường batch riêng. Warm-up xảy ra trong process, nên chạy nhiều
lần trong **cùng một notebook cell** nếu muốn thống kê. Cell sau lưu raw sample
đơn vị giây và latency trung bình trên một ảnh:

```python
import json, statistics, time
import numpy as np
from cpu_baseline import load_data
from gpu_v2 import run_gpu_v2

B, N, REPEATS = 32, 10_000, 7
samples = [load_data(N, seed=i) for i in range(B)]
boxes = np.stack([x[0] for x in samples])
scores = np.stack([x[1] for x in samples])

run_gpu_v2(boxes[:, :64], scores[:, :64])  # JIT warm-up
run_gpu_v2(boxes, scores)                  # allocation/cache warm-up

times = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    run_gpu_v2(boxes, scores)
    times.append(time.perf_counter() - t0)

report = {
    "batch_size": B,
    "boxes_per_image": N,
    "samples_seconds": times,
    "median_batch_seconds": statistics.median(times),
    "stddev_batch_seconds": statistics.stdev(times),
    "median_per_image_seconds": statistics.median(times) / B,
}
print(json.dumps(report, indent=2))
with open("presentation/seminar_2/evidence/batch32_t4.json", "w") as f:
    json.dump(report, f, indent=2)
```

Con số này là **end-to-end V2 latency**: sort host, transfer, kernel, copy
mask về host và CPU greedy resolution. Ghi rõ điều đó trên slide.

## 5. Thu artifact và cập nhật trạng thái

Tải về hoặc commit các file trong `presentation/seminar_2/evidence/`. Điền số
vào slide chỉ khi tên file, GPU, runtime và command đã rõ. Nếu V2 không đạt
`<5 ms`, vẫn báo số thật và giải thích mask copy + CPU greedy là giới hạn.
