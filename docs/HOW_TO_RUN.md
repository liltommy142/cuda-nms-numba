# Hướng dẫn tự chạy code & test

> 🧭 Về [docs/INDEX.md](INDEX.md) · Giải thích khái niệm/thuật ngữ xem [docs/GLOSSARY.md](GLOSSARY.md), giải thích code chi tiết xem [docs/TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).
>
> Ghi chú nhanh để không phải hỏi lại — chạy trực tiếp trên máy, không cần AI hỗ trợ.

## 1. Chạy trên Google Colab (có GPU T4 thật)

Mở thẳng notebook từ GitHub qua link Colab (không cần tải file về máy):

- GPU V1: https://colab.research.google.com/github/liltommy142/cuda-nms-numba/blob/main/src/gpu_v1.ipynb
- GPU V2 (Bitmask NMS): https://colab.research.google.com/github/liltommy142/cuda-nms-numba/blob/main/src/gpu_v2.ipynb
- GPU V3 (Matrix NMS): https://colab.research.google.com/github/liltommy142/cuda-nms-numba/blob/main/src/gpu_v3.ipynb
- CPU baseline: https://colab.research.google.com/github/liltommy142/cuda-nms-numba/blob/main/src/cpu_baseline.ipynb

**Các bước:**

1. Mở link ở trên (đăng nhập Google trước).
2. **Chỉ cần cho file GPU**: bật GPU — menu `Runtime` → `Change runtime type` → chọn **T4 GPU** ở mục Hardware accelerator → `Save`.
3. Chạy toàn bộ: menu `Runtime` → `Run all` (hoặc `Ctrl+F9`). Mất khoảng 1-2 phút, phần lớn là thời gian cài `numba` / kết nối GPU lần đầu.
4. Đọc kết quả: cell benchmark cuối cùng in bảng `N | CPU(s) | GPU V1(s) | Speedup`. Không có traceback đỏ → thành công.

Lưu ý: T4 trên Colab **là GPU của NVIDIA** (Tesla T4, kiến trúc Turing, 16GB VRAM) — Google chỉ cho thuê hạ tầng, không phải hãng GPU riêng. Code CUDA/Numba chạy y hệt trên T4 hay bất kỳ GPU NVIDIA nào khác, chỉ khác tốc độ.

Notebook GPU clone repo rồi chạy trực tiếp `src/*.py` và `tests/`; đây là để
kernel, test và benchmark luôn dùng cùng một source of truth. Trước khi mở
Colab, hãy push commit cần kiểm tra lên GitHub hoặc upload đúng bản repo đó.

## 2. Chạy local (không cần GPU/Colab)

```bash
cd "Applied-Parallel-Programming/Project/cuda-nms-numba"
source .venv/bin/activate

# CPU baseline: class-aware hard NMS on deterministic synthetic candidates
python src/cpu_baseline.py --source synthetic --n 1000 --verify
python src/cpu_baseline.py --source synthetic --benchmark --verify

# Real detector integration: raw (pre-NMS) YOLO candidates + CPU NMS
python src/cpu_baseline.py --source yolo-live --image path/to/image.jpg --verify

# Chạy GPU scripts (Chỉ áp dụng nếu máy bạn CÓ GPU NVIDIA cài sẵn CUDA và Numba)
python src/gpu_v1.py --benchmark
python src/gpu_v2.py --benchmark
python src/gpu_v3.py --benchmark

# Toàn bộ test
pytest tests/ -v

# Benchmark lặp lại: median/stddev, raw samples và metadata máy/GPU
python benchmarks/run_all.py --versions cpu v1 v2 v3 --repeats 7 \
  --json benchmarks/results/measurement.json

# GPU V1 sẽ báo lỗi có chủ đích trên máy không có CUDA (vd. macOS) — bình thường:
python src/gpu_v1.py --n 100 --verify
# → "ERROR: No CUDA-capable GPU detected." — đúng thiết kế, không phải bug
```

### Cách tự đọc kết quả

- `--verify`: xem dòng cuối `Exact match: True` → đúng. `False` → có bug thật.
- `pytest tests/ -v`: đếm `passed` / `failed` / `skipped` ở dòng cuối.
  - `failed` → vấn đề thật, cần sửa.
  - `skipped` (test GPU trên máy không có CUDA) → bình thường, không phải lỗi.
  - Muốn xem log chi tiết khi fail: thêm `-vv` hoặc `--tb=long`.

V2 nhận input class-aware `(B, N, 4)` / `(B, N)` / `(B, N) class_ids`. Khi
tất cả candidate thuộc một class, V2 chạy một CUDA mask-kernel cho cả batch;
multi-class được partition theo class trước khi dùng cùng kernel. Để đo đúng batch-size 32,
không dùng `run_all.py` mà chạy:

```bash
python benchmarks/run_v2_batch.py --batch-size 32 --n 10000 \
  --warmup 2 --repeats 7 --json benchmarks/results/v2_batch32.json
```

Đây là end-to-end latency (host sort, transfer, GPU kernel, copy mask về và
CPU greedy resolution). Evidence T4 hiện có là 1.002 s/batch, nên không được
tuyên bố đạt mục tiêu catalog `<5 ms`/batch.

## 3. Chạy detector thật với raw candidate trước NMS

```bash
python src/cpu_baseline.py --source yolo-live --image path/to/image.jpg --verify

# Report tách thời gian raw detector candidate và NMS
python benchmarks/run_detector_pipeline.py --image path/to/image.jpg --runner cpu \
  --json benchmarks/results/detector_pipeline.json
```

Cần weight `yolov5s.pt` trong repo (đã có) và dependency `ultralytics`. Adapter
gọi network tensor trực tiếp rồi tự decode `xywh/objectness/class probability`;
không dùng AutoShape/`Results.boxes` đã NMS.

Nếu gặp lỗi SSL `CERTIFICATE_VERIFY_FAILED` (thường do venv Python thiếu chứng chỉ gốc, hay gặp trên macOS):

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
```

## Tóm tắt 2 lệnh cần nhớ nhất

```bash
pytest tests/ -v                       # kiểm tra đúng/sai
python src/cpu_baseline.py --source synthetic --benchmark # đo NMS synthetic
```
