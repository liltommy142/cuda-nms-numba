# Kế hoạch đóng lỗ hổng catalog A4 — cuda-nms-numba

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đưa repo từ trạng thái "code xong, chưa có bằng chứng" sang "mọi tuyên bố trong slide đều có artefact đo thật", và đóng 3 lỗ hổng nội dung so với catalog A4: batch size 32, verify GPU ↔ torchvision, profiling detector pipeline.

**Architecture:** Mọi thay đổi code làm trên máy local (không GPU) và chỉ được coi là "xong" sau khi pytest + benchmark chạy trên **Google Colab T4**. Vì notebook Colab `git clone` từ GitHub, mỗi lần muốn verify phải **commit + push trước**, rồi mới chạy Colab — ngược với vòng lặp TDD thông thường. Kế hoạch chia 4 phase, mỗi phase kết thúc bằng một **Colab Gate** bắt buộc.

**Tech Stack:** Python 3.11, NumPy 1.26, Numba CUDA (`@cuda.jit`), pytest, torch/torchvision (chỉ để verify), Google Colab T4 (Turing, 16GB VRAM).

## Global Constraints

- **Không có GPU local.** Mọi test có marker `@requires_gpu` chỉ chạy được trên Colab. Local chỉ chạy `pytest tests -q` (hiện: 12 passed, 35 skipped) để bắt lỗi cú pháp/CPU.
- **Notebook Colab clone từ `https://github.com/liltommy142/cuda-nms-numba.git` nhánh mặc định.** Code chưa push = code Colab không thấy. Push trước, verify sau.
- **Không viết số đo vào README/slide nếu chưa có file JSON trong `benchmarks/results/`.** Quy tắc này đã có ở `docs/GPU_VALIDATION_RUNBOOK.md:3-4`; kế hoạch này tuân thủ tuyệt đối.
- **Ngôn ngữ tài liệu:** tiếng Việt (theo `docs/*.md` hiện có). Docstring/comment trong code: tiếng Anh.
- **Không dùng CUDA C/C++.** Chỉ Numba `@cuda.jit` — ràng buộc của môn học (proposal, mục Resources).
- **Dung sai đối chiếu:** IoU `1e-4`; tập index giữ lại phải khớp **tuyệt đối** với `run_cpu` cho V1/V2 (hard NMS). V3 là soft suppression — chỉ đối chiếu với `matrix_nms_reference`, **không** với greedy NMS.
- **Batch dimension:** `boxes` shape `(B, N, 4)` float32, `scores` shape `(B, N)` float32. Hàm batch trả về **list gồm B mảng int64**, mỗi mảng là index trong tập box của ảnh đó (không phải index phẳng) — vì mỗi ảnh giữ số box khác nhau.
- **Grid limits (T4, compute 7.5):** `gridDim.x` ≤ 2³¹−1, `gridDim.y`/`gridDim.z` ≤ 65535. Với N=10.000, B=32 mọi cấu hình dưới đây đều nằm trong giới hạn.

---

## Phân vai theo catalog A4 (nguồn cho mọi quyết định phạm vi bên dưới)

| Version | Catalog A4 yêu cầu (nguyên văn) | Optimization ladder (Project Description) | Batch 32? | Trạng thái sau kế hoạch này |
|---|---|---|---|---|
| CPU baseline | *"Full object detection inference using a pretrained YOLO or SSD model in PyTorch, followed by greedy NMS in NumPy. Profile to confirm NMS is the post-processing bottleneck at large batch sizes"* | Level 0 | Chỉ để so sánh (wrapper lặp) | Task 6, 7, 8 |
| GPU V1 | *"Parallel IoU matrix kernel — one thread per (box_i, box_j) pair"* | Level 1: *"Correct, but possibly slower than CPU. Goal: correctness, not speed"* | **Không** — catalog không yêu cầu; và 32×400MB = OOM ở N=10.000 | Giữ nguyên thuật toán; chỉ thêm test + cứu bằng chứng (Task 1, 5) |
| GPU V2 | *"**Batched** NMS using parallel reduction to build the suppression mask; coalesced box coordinate reads"* | Level 2: memory optimization (coalesced, shared memory, giảm round-trip) | **Có — fused** (Task 11) | Task 9, 10, 11 |
| GPU V3 | *"Matrix NMS (Wang et al. 2020) — replace serial greedy NMS with a fully parallel soft-suppression pass"* | Level 3: compute optimization (*warp-level intrinsics, kernel fusion*) | **Có — fused** (Task 12) | Task 10, 12, 14 |
| Benchmark | *"NMS latency for N ∈ {100, 1000, 10,000}; verify final detections match torchvision NMS within 1e-4"* | — | Có (Task 13) | Task 5, 13 |
| Target | *"Process 10,000 boxes at batch size 32 in under 5 ms. Target 30–80× speedup"* | — | — | Đo thật ở Colab Gate #3 |

## File Structure

**Tạo mới:**
- `benchmarks/profile_cpu.py` — cProfile CPU baseline, ghi ra `profile_output/cprofile_N10000.txt` (đúng đường dẫn proposal đã trích dẫn).
- `benchmarks/profile_detector_pipeline.py` — chạy YOLOv5s trên batch ảnh, tách thời gian forward pass vs NMS, chứng minh NMS là bottleneck của post-processing (catalog mục 1). Chỉ chạy được ở Colab.
- `benchmarks/results/colab_t4_v1_legacy.md` — output V1 cứu từ `git show HEAD:src/gpu_v1.ipynb` (bằng chứng cho con số 9.7×).
- `benchmarks/results/*.json` — artefact từ mỗi Colab Gate.
- `tests/test_batch.py` — test riêng cho đường batch (V2/V3 fused vs gọi từng ảnh).

**Sửa:**
- `src/gpu_v2.py` — sửa docstring cho khớp code thật; sửa barrier phân kỳ trong `_nms_bitmask_kernel`; thêm `_nms_bitmask_batch_kernel` + `run_gpu_v2_batch`.
- `src/gpu_v3.py` — thêm guard `cuda.is_available()`; thêm `_iou_max_batch_kernel`/`_decay_scores_batch_kernel` + `run_gpu_v3_batch`; warp-shuffle reduction.
- `src/cpu_baseline.py` — thêm `run_cpu_batch()` (wrapper lặp, dùng làm mốc so sánh batch).
- `src/cpu_baseline.ipynb` — sửa bản sao `load_real_boxes` đã lệch.
- `src/gpu_v*.ipynb` — cell setup tự chẩn đoán numba thay vì pin cứng.
- `tests/test_correctness.py` — sửa 4 test gọi `.copy_to_host()` sai; thêm test torchvision cho GPU; thêm test N=0 cho V2/V3.
- `benchmarks/run_all.py` — thêm `--batch`.
- `requirements.txt` — sửa comment sai về Colab.
- `docs/VERIFIED_STATUS.md`, `docs/TECHNICAL_DOCUMENTATION.md`, `presentation/*.md` — đồng bộ với số thật.

---

# PHASE A — Cứu bằng chứng & mở khoá Colab

> Không làm gì khác trước khi xong Phase A. Hiện tại runbook xác minh GPU **không chạy được** (3 blocker), và một `git commit -a` lúc này sẽ xoá mất bằng chứng duy nhất của con số 9.7×.

### Task 1: Cứu output benchmark V1 khỏi bản rewrite notebook

`presentation/README.md:33` và `presentation/SCRIPT.md:81` viện dẫn "9.7× @ N=10.000, xem `src/gpu_v1.ipynb`". Bản `src/gpu_v1.ipynb` trong working tree (chưa commit) có **0 cell output** — commit nguyên trạng là mất bằng chứng.

**Files:**
- Create: `benchmarks/results/colab_t4_v1_legacy.md`
- Read-only: `git show HEAD:src/gpu_v1.ipynb`

**Interfaces:**
- Produces: file bằng chứng mà `presentation/README.md` và `docs/VERIFIED_STATUS.md` sẽ trỏ tới thay cho notebook (Task 15).

- [ ] **Step 1: Trích toàn bộ output của notebook bản HEAD ra file tạm**

```bash
cd "D:/Study/HCMUS-APP/cuda-nms-numba"
git show HEAD:src/gpu_v1.ipynb > /tmp/gpu_v1_head.ipynb
python - <<'PY'
import json
nb = json.load(open('/tmp/gpu_v1_head.ipynb', encoding='utf8'))
for i, c in enumerate(nb['cells']):
    for o in c.get('outputs', []):
        txt = ''.join(o.get('text', [])) or o.get('data', {}).get('text/plain', '')
        if txt:
            print(f'--- cell {i} ---')
            print(''.join(txt) if isinstance(txt, list) else txt)
PY
```

Kỳ vọng: thấy bảng `N | CPU (s) | GPU V1 (s) | Speedup` chứa `2.4918`, `0.2557`, `9.7x`, và output `nvidia-smi` (tên GPU, driver).

- [ ] **Step 2: Viết file bằng chứng**

Tạo `benchmarks/results/colab_t4_v1_legacy.md` với đúng cấu trúc sau, dán output thật vào các khối code (không gõ lại số bằng tay — copy nguyên văn):

```markdown
# GPU V1 — số đo Colab T4 (trích từ output notebook, commit 114df94)

Nguồn: output đã lưu trong `src/gpu_v1.ipynb` tại commit `114df94`, trước khi
notebook được viết lại thành dạng clone-and-run (không còn output).
Khôi phục bằng: `git show 114df94:src/gpu_v1.ipynb`.

Hạn chế đã biết của số này: chạy 1 lần/N, không có warmup lặp lại, không có
median/stddev — khác phương pháp của `benchmarks/run_all.py`. Dùng làm mốc
lịch sử, không dùng thay cho artefact JSON.

## Môi trường

​```
<dán output nvidia-smi + numba version tại đây>
​```

## Kết quả

​```
<dán nguyên văn bảng benchmark tại đây>
​```
```

- [ ] **Step 3: Kiểm tra file không rỗng và chứa số**

```bash
grep -c "9.7\|0.2557" benchmarks/results/colab_t4_v1_legacy.md
```
Kỳ vọng: ≥ 1.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/results/colab_t4_v1_legacy.md
git commit -m "docs: preserve GPU V1 Colab T4 measurements before notebook rewrite"
```

---

### Task 2: Sửa 4 test V2 gọi `.copy_to_host()` trên ndarray

`src/gpu_v2.py:284` trả về **host ndarray** (`return d_iou.copy_to_host()`), nhưng test gọi tiếp `.copy_to_host()` lần nữa → `AttributeError` trên bất kỳ máy có CUDA. Lỗi có sẵn từ HEAD, ẩn được vì local skip hết test GPU. Đây là 4 trong số các test mà `docs/GPU_VALIDATION_RUNBOOK.md` bước 2 yêu cầu phải pass.

**Files:**
- Modify: `tests/test_correctness.py:230-281`
- Read-only: `src/gpu_v1.py:102-126`, `src/gpu_v2.py:266-284`

**Interfaces:**
- Consumes: `compute_iou_matrix_gpu_v2(boxes: np.ndarray) -> np.ndarray` (host array, cùng chữ ký với `compute_iou_matrix_gpu` của V1).
- Produces: bộ test V2 chạy được trên Colab Gate #1.

- [ ] **Step 1: Sửa cả 4 chỗ**

Trong `tests/test_correctness.py`, thay 4 cặp dòng sau:

```python
# test_gpu_v2_iou_matrix_diagonal_is_one (dòng ~232)
    d_iou = compute_iou_matrix_gpu_v2(boxes)
    iou_mat = d_iou.copy_to_host()
```
thành:
```python
    iou_mat = compute_iou_matrix_gpu_v2(boxes)
```

```python
# test_gpu_v2_iou_matrix_is_symmetric (dòng ~244)
    d_iou = compute_iou_matrix_gpu_v2(boxes)
    iou_mat = d_iou.copy_to_host()
```
thành:
```python
    iou_mat = compute_iou_matrix_gpu_v2(boxes)
```

```python
# test_gpu_v2_iou_matrix_matches_cpu (dòng ~256)
    d_iou = compute_iou_matrix_gpu_v2(boxes)
    iou_mat_gpu = d_iou.copy_to_host()
```
thành:
```python
    iou_mat_gpu = compute_iou_matrix_gpu_v2(boxes)
```

```python
# test_gpu_v2_iou_matrix_matches_v1 (dòng ~276)
    d_iou_v2 = compute_iou_matrix_gpu_v2(boxes)
    iou_v2 = d_iou_v2.copy_to_host()
```
thành:
```python
    iou_v2 = compute_iou_matrix_gpu_v2(boxes)
```

- [ ] **Step 2: Thêm test chặn tái phát kiểu trả về**

Thêm vào `tests/test_correctness.py`, ngay trước `test_gpu_v2_iou_matrix_diagonal_is_one`:

```python
@requires_gpu
def test_gpu_v2_iou_helper_returns_host_array():
    """compute_iou_matrix_gpu_v2 must return a host ndarray, same as the V1
    helper. Four tests previously called .copy_to_host() on its result and
    would have crashed on the first GPU machine that ran them."""
    from gpu_v2 import compute_iou_matrix_gpu_v2

    boxes, _ = load_data(8, seed=0)
    out = compute_iou_matrix_gpu_v2(boxes)
    assert isinstance(out, np.ndarray), f"expected host ndarray, got {type(out)}"
    assert not hasattr(out, "copy_to_host")
```

- [ ] **Step 3: Chạy local để chắc không vỡ collection**

```bash
.venv/Scripts/python.exe -m pytest tests -q
```
Kỳ vọng: `12 passed, 36 skipped` (thêm 1 skip do test mới cần GPU). Không có `error`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_correctness.py
git commit -m "fix(tests): stop calling copy_to_host on host ndarray in V2 IoU tests"
```

---

### Task 3: Cell setup Colab tự chẩn đoán numba thay vì pin cứng

Notebook hiện chạy `!pip install -q numba==0.59.1`. Numba 0.59.1 yêu cầu `numpy<1.27`, trong khi Colab hiện ship numpy 2.x → pip sẽ downgrade numpy và bắt restart runtime giữa chừng, làm hỏng luôn `%cd`. Đồng thời `requirements.txt:9-10` ghi "notebook dùng numba có sẵn của Colab, không dùng file này" — mâu thuẫn với chính notebook.

**Files:**
- Modify: `src/gpu_v1.ipynb` (cell 2), `src/gpu_v2.ipynb` (cell 2), `src/gpu_v3.ipynb` (cell 2)
- Modify: `requirements.txt:9-10`

**Interfaces:**
- Produces: cell setup dùng chung cho cả 3 notebook, fail-fast với thông báo rõ nếu numba của Colab không compile được kernel.

- [ ] **Step 1: Thay nội dung cell setup ở cả 3 notebook**

Nội dung cell mới (giống hệt nhau ở cả 3 file, chỉ khác dòng comment tiêu đề):

```python
!git clone --depth 1 https://github.com/liltommy142/cuda-nms-numba.git
%cd cuda-nms-numba
!nvidia-smi

import sys
import numba, numpy
import numpy as np
print('numba', numba.__version__, '| numpy', numpy.__version__)

from numba import cuda
assert cuda.is_available(), 'Enable a CUDA GPU runtime in Colab, then reconnect.'
print('GPU:', cuda.get_current_device().name)

# Smoke test before anything else. This project's kernels use loops plus several
# scalar arguments -- the exact pattern that failed to compile on numba 0.66
# (see requirements.txt). Fail here with a clear message instead of halfway
# through a benchmark.
sys.path.insert(0, 'src')
from cpu_baseline import load_data
from gpu_v1 import compute_iou_matrix_gpu

_boxes, _ = load_data(8, seed=0)
try:
    _m = compute_iou_matrix_gpu(_boxes)
    assert np.allclose(np.diag(_m), 1.0, atol=1e-4)
    print('JIT smoke test OK on numba', numba.__version__)
except Exception as exc:
    print('Preinstalled numba failed to compile the kernels:', type(exc).__name__, exc)
    print('Fallback: run  !pip install -q "numba==0.59.1" "numpy<2"  then')
    print('Runtime > Restart session, and re-run this cell.')
    raise
```

- [ ] **Step 2: Sửa comment sai trong `requirements.txt`**

Thay 2 dòng `requirements.txt:9-10`:
```
               # Colab's notebooks (gpu_v1/v2/v3.ipynb) use Colab's preinstalled numba, not this file,
               # so this pin mainly matters for local `pip install -r requirements.txt`.
```
bằng:
```
               # The Colab notebooks do NOT install this pin by default: they run a JIT smoke
               # test against Colab's preinstalled numba first and only fall back to
               # `numba==0.59.1 numpy<2` if that smoke test fails (numba 0.59.1 is incompatible
               # with the numpy 2.x Colab ships, so installing it eagerly forces a runtime restart).
```

- [ ] **Step 3: Kiểm tra notebook vẫn là JSON hợp lệ**

```bash
.venv/Scripts/python.exe -c "
import json
for f in ['src/gpu_v1.ipynb','src/gpu_v2.ipynb','src/gpu_v3.ipynb']:
    nb = json.load(open(f, encoding='utf8'))
    print(f, len(nb['cells']), 'cells OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/gpu_v1.ipynb src/gpu_v2.ipynb src/gpu_v3.ipynb requirements.txt
git commit -m "fix(colab): smoke-test preinstalled numba instead of forcing an incompatible pin"
```

---

### Task 4: Commit + push mọi thứ Colab cần thấy

`benchmarks/` đang **untracked** — nhưng cả 3 notebook chạy `!python benchmarks/run_all.py` sau khi clone, và `docs/GPU_VALIDATION_RUNBOOK.md:30` cũng dựa vào nó. Chạy hôm nay → `No such file or directory`. Tương tự `docs/VERIFIED_STATUS.md`, `docs/GPU_VALIDATION_RUNBOOK.md`, và các sửa ở `src/gpu_v2.py`/`src/gpu_v3.py`/`tests/`.

**Files:**
- Add: `benchmarks/run_all.py`, `docs/GPU_VALIDATION_RUNBOOK.md`, `docs/VERIFIED_STATUS.md`
- Commit: các thay đổi đang dirty ở `src/gpu_v2.py`, `src/gpu_v3.py`, `README.md`, `docs/HOW_TO_RUN.md`, `docs/INDEX.md`, `src/*.ipynb`

- [ ] **Step 1: Đảm bảo không commit rác**

Kiểm tra `.gitignore` đã loại `__pycache__` và `.venv`:
```bash
cat .gitignore
git status --short
```
Nếu `benchmarks/__pycache__` hoặc `src/__pycache__` xuất hiện trong `git status`, thêm vào `.gitignore` trước:
```
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: Add + commit theo nhóm**

```bash
git add benchmarks/run_all.py docs/GPU_VALIDATION_RUNBOOK.md docs/VERIFIED_STATUS.md
git commit -m "feat(benchmarks): add reproducible runner and GPU validation runbook"

git add src/gpu_v2.py src/gpu_v3.py
git commit -m "docs(kernels): correct V2 PCIe complexity claims, document V3 kernels"

git add src/cpu_baseline.ipynb src/gpu_v1.ipynb src/gpu_v2.ipynb src/gpu_v3.ipynb
git commit -m "refactor(notebooks): run repo source instead of duplicating kernel code"

git add README.md docs/HOW_TO_RUN.md docs/INDEX.md
git commit -m "docs: state batch-32 gap and repeated-measurement workflow"
```

- [ ] **Step 3: Push**

```bash
git push origin main
```

- [ ] **Step 4: Xác nhận GitHub đã có file**

```bash
git ls-remote --heads origin main
gh api repos/liltommy142/cuda-nms-numba/contents/benchmarks/run_all.py --jq .name
```
Kỳ vọng: in ra `run_all.py`. Nếu lỗi 404 → chưa push xong, không được sang Colab Gate.

---

### 🔶 COLAB GATE #1 — lần chạy GPU thật đầu tiên của V2/V3

**Người thực hiện: bạn (thủ công trên trình duyệt).** Đây là lần đầu tiên V2 và V3 chạy trên GPU thật — khả năng cao sẽ lộ lỗi.

- [ ] **Step 1: Mở lần lượt 3 notebook, bật T4, Run all**

- https://colab.research.google.com/github/liltommy142/cuda-nms-numba/blob/main/src/gpu_v1.ipynb
- https://colab.research.google.com/github/liltommy142/cuda-nms-numba/blob/main/src/gpu_v2.ipynb
- https://colab.research.google.com/github/liltommy142/cuda-nms-numba/blob/main/src/gpu_v3.ipynb

`Runtime → Change runtime type → T4 GPU` → `Runtime → Run all`.

- [ ] **Step 2: Ghi lại đúng 3 thứ cho mỗi notebook**

1. Dòng cuối của pytest (`N passed, M skipped, K failed`).
2. Toàn bộ traceback nếu có `failed`.
3. File JSON tải về từ cell cuối.

- [ ] **Step 3: Đưa kết quả về repo**

Copy 3 file JSON vào `benchmarks/results/`, rồi:
```bash
git add benchmarks/results/colab_t4_v1.json benchmarks/results/colab_t4_v2.json benchmarks/results/colab_t4_v3.json
git commit -m "test(colab): first real T4 measurements for V1/V2/V3"
git push origin main
```

- [ ] **Step 4: Quyết định**

- Nếu **có `failed`**: dừng Phase B, dán traceback vào phiên làm việc — sửa lỗi trước, chạy lại Gate #1.
- Nếu **all pass**: đây là lần đầu tiên V2/V3 có số thật. Sang Phase B.

---

# PHASE B — Đúng yêu cầu catalog cho từng version

### Task 5: Test GPU ↔ torchvision trong dung sai 1e-4

Catalog mục 6: *"verify final detections match torchvision NMS within 1e-4"*. Hiện chỉ **CPU baseline** so trực tiếp với torchvision (`test_cpu_matches_torchvision_reference`). V1/V2 chỉ so với `run_cpu` — đúng về mặt bắc cầu nhưng không phải điều catalog viết, và không bắt được trường hợp cả `run_cpu` lẫn GPU cùng sai.

**Files:**
- Modify: `tests/test_correctness.py` (thêm vào cuối phần V2)

**Interfaces:**
- Consumes: `run_gpu_v1`, `run_gpu_v2`, `requires_gpu`, `requires_torch`, `load_data`.

- [ ] **Step 1: Viết test**

```python
@requires_gpu
@requires_torch
@pytest.mark.parametrize("n", [200, 1_000, 10_000])
def test_gpu_versions_match_torchvision_reference(n):
    """Catalog A4 benchmark item: 'verify final detections match torchvision
    NMS within 1e-4'. Checking V1/V2 only against run_cpu would not catch a
    bug shared by the baseline and the kernels."""
    import torch
    from torchvision.ops import nms as torch_nms
    from gpu_v1 import run_gpu_v1
    from gpu_v2 import run_gpu_v2

    boxes, scores = load_data(n, seed=13)
    iou_threshold = 0.5

    ref = set(
        torch_nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold)
        .numpy()
        .tolist()
    )
    v1 = set(run_gpu_v1(boxes, scores, iou_threshold).tolist())
    v2 = set(run_gpu_v2(boxes, scores, iou_threshold).tolist())

    assert v1 == ref, f"V1 vs torchvision: only V1={sorted(v1 - ref)[:5]}, only ref={sorted(ref - v1)[:5]}"
    assert v2 == ref, f"V2 vs torchvision: only V2={sorted(v2 - ref)[:5]}, only ref={sorted(ref - v2)[:5]}"
```

- [ ] **Step 2: Chạy local (sẽ skip, chỉ để chắc không lỗi import)**

```bash
.venv/Scripts/python.exe -m pytest tests -q -k torchvision
```
Kỳ vọng: 3 passed (test CPU cũ) + 3 skipped (test GPU mới), không error.

- [ ] **Step 3: Commit**

```bash
git add tests/test_correctness.py
git commit -m "test: verify GPU V1/V2 detections directly against torchvision.ops.nms"
```

---

### Task 6: `benchmarks/profile_cpu.py` + tái tạo artefact cProfile proposal đã trích

Project Description Part 5.1 nêu đích danh file `benchmarks/profile_cpu.py`. Proposal (dòng 49) trích `profile_output/cprofile_N10000.txt` làm bằng chứng "65% trong run_cpu, 34% trong iou_one_to_many" — **file đó không tồn tại trong repo**.

**Files:**
- Create: `benchmarks/profile_cpu.py`
- Create (do script sinh ra): `profile_output/cprofile_N10000.txt`

**Interfaces:**
- Consumes: `cpu_baseline.load_data`, `cpu_baseline.run_cpu`.
- Produces: file text mà proposal + `presentation/OUTLINE_AND_CONTENT.md:62` trỏ tới.

- [ ] **Step 1: Viết script**

```python
"""cProfile the CPU baseline -- Project Description Part 5.1.

Writes profile_output/cprofile_N<N>.txt, the artefact the proposal cites as
evidence that the suppression loop, not data loading, dominates runtime.

Usage:
    python benchmarks/profile_cpu.py            # N=10000
    python benchmarks/profile_cpu.py --n 1000
"""

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpu_baseline import load_data, run_cpu  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the CPU NMS baseline")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    boxes, scores = load_data(args.n, seed=args.seed)
    print(f"Input shape: {boxes.shape}, dtype: {boxes.dtype}")
    print(f"Input size: {boxes.nbytes / 1e6:.1f} MB")

    profiler = cProfile.Profile()
    profiler.enable()
    keep = run_cpu(boxes, scores, args.iou_threshold)
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("cumulative").print_stats(args.top)
    report = buf.getvalue()
    print(report)
    print(f"kept {len(keep)}/{len(boxes)} boxes")

    out_dir = ROOT / "profile_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"cprofile_N{args.n}.txt"
    header = (
        f"cProfile of cpu_baseline.run_cpu\n"
        f"N={args.n}  seed={args.seed}  iou_threshold={args.iou_threshold}\n"
        f"kept {len(keep)}/{len(boxes)} boxes\n"
        f"{'=' * 72}\n"
    )
    out_path.write_text(header + report, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Chạy và đọc kết quả**

```bash
.venv/Scripts/python.exe benchmarks/profile_cpu.py --n 10000
```
Kỳ vọng: file `profile_output/cprofile_N10000.txt` tồn tại; trong bảng, `run_cpu` và `iou_one_to_many` chiếm phần lớn `cumtime`. **Ghi lại tỉ lệ thật** — nhiều khả năng khác 65/34% trong proposal (máy khác, NumPy khác); tỉ lệ mới sẽ dùng ở Task 15.

- [ ] **Step 3: Đảm bảo `profile_output/` được commit chứ không bị ignore**

```bash
git check-ignore -v profile_output/cprofile_N10000.txt || echo "not ignored - OK"
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/profile_cpu.py profile_output/cprofile_N10000.txt
git commit -m "feat(benchmarks): add profile_cpu.py and regenerate the cProfile artefact the proposal cites"
```

---

### Task 7: Profiling detector pipeline — chứng minh NMS là bottleneck post-processing

Catalog mục 1 yêu cầu *"Full object detection inference using a pretrained YOLO or SSD model in PyTorch, followed by greedy NMS in NumPy. **Profile to confirm NMS is the post-processing bottleneck at large batch sizes**"*. Hiện repo mới chứng minh "NMS chiếm 99% thời gian của script NMS" — chưa chứng minh NMS là nút thắt **trong pipeline detector**.

**Files:**
- Create: `benchmarks/profile_detector_pipeline.py`
- Read-only: `src/cpu_baseline.py:36-54` (`load_real_boxes`)

**Interfaces:**
- Produces: `benchmarks/results/detector_pipeline_<device>.json` với các khoá `forward_seconds`, `nms_seconds`, `nms_share`, `boxes_per_image`.

- [ ] **Step 1: Viết script**

```python
"""Profile a real detector pipeline: YOLOv5s forward pass vs our NumPy NMS.

Catalog A4 item 1 asks us to confirm NMS is the post-processing bottleneck at
large batch sizes. This script times the two stages separately on the same
batch and reports the split.

Needs internet (torch.hub) and is meant to run on Colab, not the offline dev box.

Usage:
    python benchmarks/profile_detector_pipeline.py --batch 32 --json benchmarks/results/detector_pipeline_t4.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpu_baseline import run_cpu  # noqa: E402

IMAGE_URL = "https://ultralytics.com/images/zidane.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv5s forward vs NumPy NMS")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf-threshold", type=float, default=0.001,
                        help="low on purpose: we want the raw pre-NMS candidates")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import torch

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, trust_repo=True)
    model.conf = args.conf_threshold
    model.iou = 1.0          # keep AutoShape from doing the NMS we want to measure
    model.max_det = 30_000   # do not truncate the candidate list

    images = [IMAGE_URL] * args.batch

    # Warm up: first call downloads weights and builds cuDNN plans.
    _ = model(images[:1])

    forward_samples, nms_samples, box_counts = [], [], []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        results = model(images)
        forward_samples.append(time.perf_counter() - t0)

        per_image = [
            (pred[:, :4].cpu().numpy().astype(np.float32),
             pred[:, 4].cpu().numpy().astype(np.float32))
            for pred in results.xyxy
        ]
        box_counts.append(int(np.mean([len(b) for b, _ in per_image])))

        t0 = time.perf_counter()
        for boxes, scores in per_image:
            run_cpu(boxes, scores, args.iou_threshold)
        nms_samples.append(time.perf_counter() - t0)

    forward = float(np.median(forward_samples))
    nms = float(np.median(nms_samples))
    report = {
        "batch": args.batch,
        "repeats": args.repeats,
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "mean_boxes_per_image": int(np.mean(box_counts)),
        "forward_seconds": forward,
        "nms_seconds": nms,
        "nms_share_of_total": nms / (forward + nms),
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    print(json.dumps(report, indent=2))
    print(f"\nNMS is {report['nms_share_of_total'] * 100:.1f}% of forward+NMS at batch={args.batch}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Thêm cell chạy script này vào `src/cpu_baseline.ipynb`**

Thêm cell markdown + code ở cuối notebook:

```python
# Catalog A4 item 1: is NMS really the post-processing bottleneck at batch 32?
!python benchmarks/profile_detector_pipeline.py --batch 32 --repeats 3 \
    --json benchmarks/results/detector_pipeline_t4.json
```

- [ ] **Step 3: Commit (chưa chạy được ở local — không có torch hub/internet ổn định)**

```bash
git add benchmarks/profile_detector_pipeline.py src/cpu_baseline.ipynb
git commit -m "feat(benchmarks): profile YOLOv5s forward pass vs NumPy NMS at batch 32"
```

> ⚠️ Script này **chỉ chạy được ở Colab Gate #2**. Nếu `mean_boxes_per_image` ra rất nhỏ (< 100), tăng `--conf-threshold` xuống 0.0001 hoặc kiểm tra `model.max_det` — nghĩa là AutoShape vẫn đang cắt bớt candidate.

---

### Task 8: Sửa bản sao `load_real_boxes` đã lệch trong `cpu_baseline.ipynb`

Notebook chứa bản sao của `load_real_boxes` **thiếu `model.iou = 1.0` và `trust_repo=True`** so với `src/cpu_baseline.py:40-45`. Hệ quả thật: notebook lấy box **đã bị AutoShape NMS sẵn** rồi NMS tiếp → demo "real boxes" sai bản chất, và người xem sẽ thấy `run_cpu` gần như không loại được box nào.

**Files:**
- Modify: `src/cpu_baseline.ipynb` (cell `load_real_boxes`)

- [ ] **Step 1: Thay thân hàm trong notebook cho khớp `.py`**

```python
def load_real_boxes(image_paths=None, conf_threshold=0.25):
    """Run pretrained YOLOv5s on a handful of images to get real (boxes, scores)."""
    import torch

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, trust_repo=True)
    model.conf = conf_threshold
    # AutoShape runs its own NMS internally before returning results; disable it
    # (iou=1.0 keeps virtually all overlapping candidates) so load_real_boxes returns
    # raw pre-NMS boxes for this project's own run_cpu/run_gpu_v1 to suppress.
    model.iou = 1.0
    if not image_paths:
        image_paths = ["https://ultralytics.com/images/zidane.jpg"]

    results = model(image_paths)
    boxes_list = [pred[:, :4].cpu().numpy() for pred in results.xyxy]
    scores_list = [pred[:, 4].cpu().numpy() for pred in results.xyxy]
    boxes = np.concatenate(boxes_list, axis=0).astype(np.float32)
    scores = np.concatenate(scores_list, axis=0).astype(np.float32)
    return boxes, scores
```

- [ ] **Step 2: Thêm cảnh báo trùng lặp vào cell markdown ngay trên nó**

```markdown
> ⚠️ Hàm này là bản sao của `src/cpu_baseline.py`. Nếu sửa một bên, sửa cả hai —
> bản notebook đã từng lệch (thiếu `model.iou = 1.0`) khiến "real boxes" thực ra
> là box đã qua NMS nội bộ của AutoShape.
```

- [ ] **Step 3: Kiểm tra JSON hợp lệ + commit**

```bash
.venv/Scripts/python.exe -c "import json; json.load(open('src/cpu_baseline.ipynb', encoding='utf8')); print('OK')"
git add src/cpu_baseline.ipynb
git commit -m "fix(notebook): keep load_real_boxes in sync with cpu_baseline.py"
```

---

### Task 9: Đồng bộ mô tả V2 với code thật (kernel IoU coalesced không nằm trong hot path)

`_iou_matrix_coalesced_kernel` **không hề được gọi** trong `run_gpu_v2` — chỉ `compute_iou_matrix_gpu_v2` (dùng cho test) gọi nó. Nhưng docstring `src/gpu_v2.py:50-57` tính "IoU kernel (coalesced): ~10-20ms" vào tổng thời gian V2, và sơ đồ `docs/TECHNICAL_DOCUMENTATION.md:383-384` vẽ nó nằm trong pipeline. Giữ nguyên code (gọi nó sẽ cấp phát lại ma trận 400MB — đúng thứ V2 sinh ra để tránh), sửa mô tả.

Lưu ý: tuyên bố "coalesced reads" **vẫn đúng** — `_nms_bitmask_kernel` đọc chính 4 mảng SoA đó (`src/gpu_v2.py:193, 203-207`). Chỉ có chuyện "V2 = V1 + kernel IoU coalesced" là sai.

**Files:**
- Modify: `src/gpu_v2.py:1-64` (module docstring), `src/gpu_v2.py:266-272` (docstring helper)
- Modify: `docs/TECHNICAL_DOCUMENTATION.md:377-396` (sơ đồ 3.5)
- Modify: `presentation/OUTLINE_AND_CONTENT.md` (slide GPU V2), `presentation/SCRIPT.md:47`

- [ ] **Step 1: Sửa phần "Bottleneck analysis" trong module docstring**

Thay khối `src/gpu_v2.py:48-57`:
```
Expected speedup at N=10,000 -- theoretical, NOT yet measured on real GPU
...
      IoU kernel (coalesced)     : ~10-20ms
```
bằng:
```
Kernel inventory -- what actually runs in run_gpu_v2:
    _nms_bitmask_kernel        : the only kernel on the hot path. It reads the
                                 same four SoA arrays and caches a 64-box target
                                 block in shared memory, so the coalesced-read
                                 optimisation applies here.
    _iou_matrix_coalesced_kernel: NOT called by run_gpu_v2. It exists so tests
                                 (and the report) can inspect the coalesced SoA
                                 IoU values in isolation. Calling it from the
                                 pipeline would re-introduce the N x N matrix
                                 allocation V2 exists to avoid (~400MB at N=10000).

Measured numbers live in benchmarks/results/*.json -- do not quote estimates here.
```

- [ ] **Step 2: Sửa docstring của helper**

`src/gpu_v2.py:267-272`, thêm câu đầu:
```python
    """Run the coalesced SoA IoU kernel alone and return the (N, N) matrix on host.

    NOT part of run_gpu_v2's pipeline -- this is an inspection helper. It lets
    tests check the coalesced kernel's numerical output (diagonal, symmetry,
    match vs CPU/V1) without allocating the N x N matrix during a real run.
    """
```

- [ ] **Step 3: Sửa sơ đồ mermaid 3.5**

Trong `docs/TECHNICAL_DOCUMENTATION.md`, xoá node `D["_iou_matrix_coalesced_kernel..."]` khỏi chuỗi và nối thẳng `C --> E`. Thêm ngay dưới sơ đồ:

```markdown
> **Lưu ý**: `_iou_matrix_coalesced_kernel` **không** nằm trong pipeline này — nó
> là helper để test/kiểm tra giá trị IoU của layout SoA. Gọi nó trong `run_gpu_v2`
> sẽ cấp phát lại ma trận N×N 400MB, đúng thứ V2 sinh ra để tránh. Tối ưu
> "coalesced reads" của V2 nằm ở chính `_nms_bitmask_kernel`.
```

- [ ] **Step 4: Sửa 1 câu trong SCRIPT.md**

`presentation/SCRIPT.md:47`, sửa cụm "V2 tách thành 4 mảng riêng x1, y1, x2, y2 — giờ các thread liền kề đọc đúng các ô nhớ liền kề nhau" thành có nói rõ nó áp dụng cho kernel bitmask:

> "V2 tách box thành 4 mảng riêng x1, y1, x2, y2, và chính kernel dựng bitmask đọc trực tiếp 4 mảng này — nên các thread liền kề đọc đúng các ô nhớ liền kề nhau, gom lại thành một lần đọc hiệu quả thay vì nhiều lần."

- [ ] **Step 5: Commit**

```bash
git add src/gpu_v2.py docs/TECHNICAL_DOCUMENTATION.md presentation/SCRIPT.md presentation/OUTLINE_AND_CONTENT.md
git commit -m "docs(v2): describe the kernel that actually runs, not the inspection helper"
```

---

### Task 10: Vá rủi ro kernel + phủ test còn thiếu

**Files:**
- Modify: `src/gpu_v2.py:158-240` (`_nms_bitmask_kernel`)
- Modify: `src/gpu_v3.py:379-383` (entry point)
- Modify: `tests/test_correctness.py`

- [ ] **Step 1: Sửa barrier phân kỳ trong `_nms_bitmask_kernel`**

`if i >= n: return` hiện nằm **trước** `cuda.syncthreads()` → thread thoát sớm không tới barrier, là hành vi không xác định theo CUDA Programming Guide khi N không chia hết 64. Thay phần đầu kernel:

```python
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    tx = cuda.threadIdx.x

    # Block-uniform early exit: every thread in this block shares bx/by, so
    # returning here does not split the block across the barrier below.
    # Box i can only suppress higher-index (lower-score) boxes, so any column
    # block entirely before i's own row block has nothing to record.
    if by < bx:
        return

    i = bx * 64 + tx
    active = i < n

    # Load the 64 target boxes for this column block into shared memory.
    # Every thread must reach the barrier below, including inactive ones --
    # hence `active` as a flag instead of an early return.
    sx1 = cuda.shared.array(shape=(64,), dtype=nb_float32)
    sy1 = cuda.shared.array(shape=(64,), dtype=nb_float32)
    sx2 = cuda.shared.array(shape=(64,), dtype=nb_float32)
    sy2 = cuda.shared.array(shape=(64,), dtype=nb_float32)

    j_load = by * 64 + tx
    if j_load < n:
        sx1[tx] = x1[j_load]
        sy1[tx] = y1[j_load]
        sx2[tx] = x2[j_load]
        sy2[tx] = y2[j_load]
    cuda.syncthreads()

    if not active:
        return

    xi1 = x1[i]; yi1 = y1[i]; xi2 = x2[i]; yi2 = y2[i]
    area_i = (xi2 - xi1) * (yi2 - yi1)

    mask_val = nb_uint64(0)
```
(phần vòng `for k in range(64)` và dòng ghi `mask_out[by, i] = mask_val` giữ nguyên)

- [ ] **Step 2: Thêm guard `cuda.is_available()` cho V3**

`src/gpu_v3.py`, thay khối `if __name__ == "__main__":`:
```python
if __name__ == "__main__":
    if not _NUMBA_AVAILABLE:
        print("ERROR: numba is not installed.  Run: pip install numba")
        sys.exit(1)
    if not cuda.is_available():
        print("ERROR: No CUDA-capable GPU detected.")
        sys.exit(1)
    main()
```
và thêm `is_available` vào `CudaDummy` (`src/gpu_v3.py:42-54`) để dòng trên không ném `AttributeError` khi numba vắng mặt:
```python
        def is_available(self): return False
```

- [ ] **Step 3: Thêm test còn thiếu**

```python
@requires_gpu
def test_gpu_v2_handles_empty_input():
    """N=0 must not crash (0-block kernel launch)."""
    from gpu_v2 import run_gpu_v2

    keep = run_gpu_v2(np.zeros((0, 4), dtype=np.float32),
                      np.zeros((0,), dtype=np.float32), iou_threshold=0.5)
    assert len(keep) == 0


@requires_gpu
def test_gpu_v3_handles_empty_input():
    from gpu_v3 import run_gpu_v3_matrix_nms

    keep = run_gpu_v3_matrix_nms(np.zeros((0, 4), dtype=np.float32),
                                 np.zeros((0,), dtype=np.float32))
    assert len(keep) == 0


@requires_gpu
@pytest.mark.parametrize("n", [63, 64, 65, 127, 128, 129])
def test_gpu_v2_block_boundary_sizes(n):
    """N around multiples of 64 exercises the partially-filled shared-memory
    block and the `active` guard in _nms_bitmask_kernel."""
    from gpu_v2 import run_gpu_v2

    boxes, scores = load_data(n, seed=n)
    assert set(run_gpu_v2(boxes, scores, 0.5).tolist()) == set(run_cpu(boxes, scores, 0.5).tolist())


@requires_gpu
def test_gpu_v3_matches_reference_scores_within_tolerance():
    """Compare decayed scores with a tolerance instead of asserting index-set
    equality: the kernel runs fastmath float32 while the oracle runs float64,
    so a box sitting exactly on score_threshold can legitimately flip."""
    from gpu_v3 import run_gpu_v3_matrix_nms

    boxes, scores = load_data(200, seed=17)
    expected_keep, expected_scores = matrix_nms_reference(
        boxes, scores, score_threshold=0.05, method="gaussian", sigma=2.0
    )
    actual_keep = run_gpu_v3_matrix_nms(
        boxes, scores, score_threshold=0.05, method="gaussian", sigma=2.0
    )

    symmetric_diff = set(actual_keep.tolist()) ^ set(expected_keep.tolist())
    for idx in symmetric_diff:
        assert abs(expected_scores[idx] - 0.05) < 1e-3, (
            f"box {idx} differs but its reference score {expected_scores[idx]:.6f} "
            f"is not near the 0.05 threshold -- this is a real mismatch, not fastmath noise"
        )
```

- [ ] **Step 4: Chạy local + commit**

```bash
.venv/Scripts/python.exe -m pytest tests -q
```
Kỳ vọng: số `passed` không đổi, số `skipped` tăng.

```bash
git add src/gpu_v2.py src/gpu_v3.py tests/test_correctness.py
git commit -m "fix(kernels): avoid divergent barrier in V2, guard V3 entry point, cover block boundaries"
git push origin main
```

---

### 🔶 COLAB GATE #2

- [ ] **Step 1:** Chạy lại 3 notebook GPU (như Gate #1) → toàn bộ test mới (torchvision, block boundary, empty input, V3 tolerance) phải pass.
- [ ] **Step 2:** Trong `cpu_baseline.ipynb` trên Colab, chạy cell mới của Task 7. Ghi lại `nms_share_of_total`.
- [ ] **Step 3:** Commit `benchmarks/results/detector_pipeline_t4.json` + JSON benchmark mới, push.
- [ ] **Step 4:** Nếu `nms_share_of_total` < 20%, **đó là kết quả hợp lệ và phải báo cáo trung thực** — nghĩa là ở batch 32 trên GPU, forward pass mới là phần nặng. Ghi vào `docs/VERIFIED_STATUS.md` ở Task 15 thay vì tìm cách chỉnh cho ra số đẹp.

---

# PHASE C — Batch size 32 (mục tiêu chính thức của catalog A4)

> Phân vai theo catalog: chữ "**Batched** NMS" nằm ở dòng GPU V2. V1 giữ nguyên (catalog chỉ yêu cầu "one thread per pair", và 32 × 400MB ma trận IoU sẽ OOM trên T4 16GB).

### Task 11: V2 fused batch — bitmask 3D

**Files:**
- Modify: `src/gpu_v2.py` (thêm kernel + host function)
- Create: `tests/test_batch.py`

**Interfaces:**
- Produces:
  - `_nms_bitmask_batch_kernel(x1, y1, x2, y2, mask_out, n, iou_threshold)` — `x1..y2`: `(B, N)` float32 device; `mask_out`: `(B, M, N)` uint64 device, `M = ceil(N/64)`; grid `(M, M, B)` × 64 threads.
  - `run_gpu_v2_batch(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[np.ndarray]` — `boxes`: `(B, N, 4)` float32, `scores`: `(B, N)` float32; trả về list B mảng int64.

- [ ] **Step 1: Viết test trước (chạy được ở Colab, skip ở local)**

Tạo `tests/test_batch.py`:

```python
"""Batch-dimension tests -- catalog A4 target: 10,000 boxes at batch size 32.

Run with:
    pytest tests/test_batch.py -v
"""

import os
import sys

import numpy as np
import pytest

_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, _SRC)

from cpu_baseline import load_data, run_cpu  # noqa: E402


def _gpu_available() -> bool:
    try:
        from numba import cuda
        return cuda.is_available()
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="No CUDA GPU available (or numba not installed) — skipping GPU tests",
)


def make_batch(batch: int, n: int, seed: int = 0):
    """Build (B, N, 4) boxes and (B, N) scores from B different random draws."""
    boxes, scores = [], []
    for b in range(batch):
        bx, sc = load_data(n, seed=seed + b)
        boxes.append(bx)
        scores.append(sc)
    return np.stack(boxes).astype(np.float32), np.stack(scores).astype(np.float32)


@requires_gpu
@pytest.mark.parametrize("batch,n", [(4, 200), (32, 1_000)])
def test_gpu_v2_batch_matches_per_image_calls(batch, n):
    """The fused batch kernel must produce exactly what B separate calls produce."""
    from gpu_v2 import run_gpu_v2, run_gpu_v2_batch

    boxes, scores = make_batch(batch, n, seed=100)
    batched = run_gpu_v2_batch(boxes, scores, iou_threshold=0.5)

    assert len(batched) == batch
    for b in range(batch):
        single = run_gpu_v2(boxes[b], scores[b], iou_threshold=0.5)
        assert set(batched[b].tolist()) == set(single.tolist()), f"image {b} differs"


@requires_gpu
def test_gpu_v2_batch_matches_cpu_baseline():
    from gpu_v2 import run_gpu_v2_batch

    boxes, scores = make_batch(8, 300, seed=200)
    batched = run_gpu_v2_batch(boxes, scores, iou_threshold=0.5)
    for b in range(8):
        assert set(batched[b].tolist()) == set(run_cpu(boxes[b], scores[b], 0.5).tolist())
```

- [ ] **Step 2: Chạy test — phải fail vì hàm chưa tồn tại**

```bash
.venv/Scripts/python.exe -m pytest tests/test_batch.py -q
```
Kỳ vọng ở local: `3 skipped` (không có GPU). Để thấy fail thật, chạy import trực tiếp:
```bash
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); import gpu_v2; gpu_v2.run_gpu_v2_batch"
```
Kỳ vọng: `AttributeError: module 'gpu_v2' has no attribute 'run_gpu_v2_batch'`.

- [ ] **Step 3: Viết kernel batch**

Thêm vào `src/gpu_v2.py` ngay sau `_nms_bitmask_kernel`:

```python
@cuda.jit
def _nms_bitmask_batch_kernel(x1, y1, x2, y2, mask_out, n, iou_threshold):
    """Fused batch version of _nms_bitmask_kernel -- catalog A4 batch size 32.

    Same per-image algorithm, one extra grid dimension for the image index.
    All B images are processed in a single kernel launch instead of B launches,
    so short per-image kernels no longer pay B times the launch latency.

    Parameters
    ----------
    x1, y1, x2, y2 : (B, N) float32 SoA device arrays
    mask_out       : (B, ceil(N/64), N) uint64 device array
    n              : int   boxes per image
    iou_threshold  : float32

    Grid  : (M, M, B) blocks with M = ceil(N/64)
    Block : 64 threads
    """
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z   # image index within the batch
    tx = cuda.threadIdx.x

    if by < bx:
        return

    i = bx * 64 + tx
    active = i < n

    sx1 = cuda.shared.array(shape=(64,), dtype=nb_float32)
    sy1 = cuda.shared.array(shape=(64,), dtype=nb_float32)
    sx2 = cuda.shared.array(shape=(64,), dtype=nb_float32)
    sy2 = cuda.shared.array(shape=(64,), dtype=nb_float32)

    j_load = by * 64 + tx
    if j_load < n:
        sx1[tx] = x1[bz, j_load]
        sy1[tx] = y1[bz, j_load]
        sx2[tx] = x2[bz, j_load]
        sy2[tx] = y2[bz, j_load]
    cuda.syncthreads()

    if not active:
        return

    xi1 = x1[bz, i]; yi1 = y1[bz, i]; xi2 = x2[bz, i]; yi2 = y2[bz, i]
    area_i = (xi2 - xi1) * (yi2 - yi1)

    mask_val = nb_uint64(0)
    for k in range(64):
        j = by * 64 + k
        if j >= n:
            break
        if j > i:
            xj1 = sx1[k]; yj1 = sy1[k]; xj2 = sx2[k]; yj2 = sy2[k]

            ix1 = max(xi1, xj1); iy1 = max(yi1, yj1)
            ix2 = min(xi2, xj2); iy2 = min(yi2, yj2)

            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            inter = inter_w * inter_h

            if inter > 0.0:
                area_j = (xj2 - xj1) * (yj2 - yj1)
                union = area_i + area_j - inter
                if (inter / union) > iou_threshold:
                    mask_val |= (nb_uint64(1) << nb_uint64(k))

    mask_out[bz, by, i] = mask_val
```

- [ ] **Step 4: Viết host function**

Thêm vào `src/gpu_v2.py` ngay sau `run_gpu_v2`:

```python
def run_gpu_v2_batch(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list:
    """Batched GPU V2 NMS -- one kernel launch for the whole batch.

    Parameters
    ----------
    boxes  : (B, N, 4) float32  [x1, y1, x2, y2] per image
    scores : (B, N)    float32
    iou_threshold : float

    Returns
    -------
    list of B int64 arrays. Element b holds indices into image b's own box
    list (not a flattened index), because each image keeps a different count.
    """
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be (B, N, 4), got {boxes.shape}")
    if scores.shape != boxes.shape[:2]:
        raise ValueError(f"scores must be (B, N) matching boxes, got {scores.shape}")

    b, n = scores.shape
    if n == 0:
        return [np.array([], dtype=np.int64) for _ in range(b)]

    order = np.argsort(-scores, axis=1, kind="stable")          # (B, N)
    rows = np.arange(b)[:, None]
    boxes_sorted = np.ascontiguousarray(boxes[rows, order], dtype=np.float32)

    d_x1 = cuda.to_device(np.ascontiguousarray(boxes_sorted[:, :, 0]))
    d_y1 = cuda.to_device(np.ascontiguousarray(boxes_sorted[:, :, 1]))
    d_x2 = cuda.to_device(np.ascontiguousarray(boxes_sorted[:, :, 2]))
    d_y2 = cuda.to_device(np.ascontiguousarray(boxes_sorted[:, :, 3]))

    m = (n + 63) // 64
    d_mask = cuda.device_array((b, m, n), dtype=np.uint64)

    _nms_bitmask_batch_kernel[(m, m, b), 64](
        d_x1, d_y1, d_x2, d_y2, d_mask, n, np.float32(iou_threshold)
    )
    cuda.synchronize()

    mask_cpu = d_mask.copy_to_host()

    keep = []
    for img in range(b):
        suppressed = np.zeros(m, dtype=np.uint64)
        keep_ranks = []
        img_mask = mask_cpu[img]
        for i in range(n):
            block_idx = i // 64
            bit_idx = i % 64
            if (suppressed[block_idx] & (np.uint64(1) << np.uint64(bit_idx))) != 0:
                continue
            keep_ranks.append(i)
            suppressed[block_idx:] |= img_mask[block_idx:, i]
        keep.append(order[img][np.array(keep_ranks, dtype=np.int64)])
    return keep
```

- [ ] **Step 5: Chạy local (chỉ để chắc import + validation hoạt động)**

```bash
.venv/Scripts/python.exe -c "
import sys, numpy as np; sys.path.insert(0,'src')
import gpu_v2
try:
    gpu_v2.run_gpu_v2_batch(np.zeros((2,3,4), np.float32), np.zeros((3,), np.float32))
except ValueError as e:
    print('validation OK:', e)
"
```
Kỳ vọng: in `validation OK: scores must be (B, N) matching boxes, got (3,)`.

- [ ] **Step 6: Commit + push**

```bash
git add src/gpu_v2.py tests/test_batch.py
git commit -m "feat(v2): fused batched bitmask kernel for catalog A4 batch size 32"
git push origin main
```

---

### Task 12: V3 fused batch — grid (N, B)

**Files:**
- Modify: `src/gpu_v3.py`
- Modify: `tests/test_batch.py`

**Interfaces:**
- Produces:
  - `_iou_max_batch_kernel(x1, y1, x2, y2, iou_max_out, n)` — `x1..y2`, `iou_max_out`: `(B, N)` float32; grid `(N, B)` × 256 threads.
  - `_decay_scores_batch_kernel(x1, y1, x2, y2, scores, iou_max, n, method, sigma)` — cùng shape; sửa `scores` tại chỗ.
  - `run_gpu_v3_batch(boxes, scores, score_threshold=0.05, method="gaussian", sigma=2.0) -> list[np.ndarray]`.

- [ ] **Step 1: Thêm test vào `tests/test_batch.py`**

```python
@requires_gpu
@pytest.mark.parametrize("batch,n", [(4, 200), (32, 1_000)])
def test_gpu_v3_batch_matches_per_image_calls(batch, n):
    """Fused batch Matrix NMS must reproduce per-image calls exactly: same
    kernels, same math, only the grid gained a dimension."""
    from gpu_v3 import run_gpu_v3_batch, run_gpu_v3_matrix_nms

    boxes, scores = make_batch(batch, n, seed=300)
    batched = run_gpu_v3_batch(boxes, scores, score_threshold=0.05, method="gaussian")

    assert len(batched) == batch
    for b in range(batch):
        single = run_gpu_v3_matrix_nms(
            boxes[b], scores[b], score_threshold=0.05, method="gaussian"
        )
        assert np.array_equal(np.sort(batched[b]), np.sort(single)), f"image {b} differs"


@requires_gpu
def test_gpu_v3_batch_matches_cpu_oracle():
    from gpu_v3 import matrix_nms_reference, run_gpu_v3_batch

    boxes, scores = make_batch(4, 100, seed=400)
    batched = run_gpu_v3_batch(boxes, scores, score_threshold=0.05, method="linear")
    for b in range(4):
        expected, _ = matrix_nms_reference(
            boxes[b], scores[b], score_threshold=0.05, method="linear"
        )
        assert np.array_equal(np.sort(batched[b]), np.sort(expected)), f"image {b} differs"
```

- [ ] **Step 2: Viết 2 kernel batch**

Thêm vào `src/gpu_v3.py` sau `_decay_scores_kernel`:

```python
@cuda.jit(fastmath=True)
def _iou_max_batch_kernel(x1, y1, x2, y2, iou_max_out, n):
    """Batched _iou_max_kernel. Block (i, b) owns iou_max_out[b, i].

    Grid  : (N, B) blocks   Block : 256 threads
    x1..y2, iou_max_out : (B, N) float32 device arrays
    """
    i = cuda.blockIdx.x    # box index within the image
    b = cuda.blockIdx.y    # image index within the batch
    tx = cuda.threadIdx.x

    if i >= n:
        return

    xi = x1[b, i]; yi = y1[b, i]; xi2 = x2[b, i]; yi2 = y2[b, i]
    area_i = (xi2 - xi) * (yi2 - yi)

    local_max = 0.0
    for k in range(tx, i, cuda.blockDim.x):
        xk = x1[b, k]; yk = y1[b, k]; xk2 = x2[b, k]; yk2 = y2[b, k]

        ix1 = max(xi, xk); iy1 = max(yi, yk)
        ix2 = min(xi2, xk2); iy2 = min(yi2, yk2)
        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw > 0.0 and ih > 0.0:
            inter = iw * ih
            area_k = (xk2 - xk) * (yk2 - yk)
            iou = inter / (area_i + area_k - inter)
            if iou > local_max:
                local_max = iou

    s_max = cuda.shared.array(shape=(256,), dtype=nb_float32)
    s_max[tx] = local_max
    cuda.syncthreads()

    stride = 128
    while stride > 0:
        if tx < stride:
            if s_max[tx + stride] > s_max[tx]:
                s_max[tx] = s_max[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0:
        iou_max_out[b, i] = s_max[0]


@cuda.jit(fastmath=True)
def _decay_scores_batch_kernel(x1, y1, x2, y2, scores, iou_max, n, method, sigma):
    """Batched _decay_scores_kernel. Block (j, b) scales scores[b, j] in place.

    Grid  : (N, B) blocks   Block : 256 threads
    """
    j = cuda.blockIdx.x
    b = cuda.blockIdx.y
    tx = cuda.threadIdx.x

    if j >= n:
        return

    xj = x1[b, j]; yj = y1[b, j]; xj2 = x2[b, j]; yj2 = y2[b, j]
    area_j = (xj2 - xj) * (yj2 - yj)

    local_min_decay = 1.0
    for i in range(tx, j, cuda.blockDim.x):
        xi = x1[b, i]; yi = y1[b, i]; xi2 = x2[b, i]; yi2 = y2[b, i]

        ix1 = max(xj, xi); iy1 = max(yj, yi)
        ix2 = min(xj2, xi2); iy2 = min(yj2, yi2)
        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw > 0.0 and ih > 0.0:
            inter = iw * ih
            area_i = (xi2 - xi) * (yi2 - yi)
            iou = inter / (area_j + area_i - inter)

            if iou > iou_max[b, i]:
                if method == 0:      # linear
                    den = 1.0 - iou_max[b, i]
                    if den < 1e-9:
                        den = 1e-9
                    decay = (1.0 - iou) / den
                else:                # gaussian
                    decay = math.exp((iou_max[b, i] * iou_max[b, i] - iou * iou) / sigma)
                if decay < local_min_decay:
                    local_min_decay = decay

    s_min = cuda.shared.array(shape=(256,), dtype=nb_float32)
    s_min[tx] = local_min_decay
    cuda.syncthreads()

    stride = 128
    while stride > 0:
        if tx < stride:
            if s_min[tx + stride] < s_min[tx]:
                s_min[tx] = s_min[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0:
        scores[b, j] = scores[b, j] * s_min[0]
```

- [ ] **Step 3: Viết host function**

Thêm vào `src/gpu_v3.py` sau `run_gpu_v3_matrix_nms`:

```python
def run_gpu_v3_batch(
    boxes: np.ndarray,
    scores: np.ndarray,
    score_threshold: float = 0.05,
    method: str = "gaussian",
    sigma: float = 2.0,
) -> list:
    """Batched Matrix NMS -- two kernel launches for the whole batch.

    Parameters
    ----------
    boxes  : (B, N, 4) float32
    scores : (B, N)    float32
    Returns a list of B int64 arrays of indices into each image's box list.
    """
    if method not in {"linear", "gaussian"}:
        raise ValueError("method must be 'linear' or 'gaussian'")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be (B, N, 4), got {boxes.shape}")
    if scores.shape != boxes.shape[:2]:
        raise ValueError(f"scores must be (B, N) matching boxes, got {scores.shape}")

    b, n = scores.shape
    if n == 0:
        return [np.array([], dtype=np.int64) for _ in range(b)]

    order = np.argsort(-scores, axis=1, kind="stable")
    rows = np.arange(b)[:, None]
    boxes_sorted = np.ascontiguousarray(boxes[rows, order], dtype=np.float32)
    scores_sorted = np.ascontiguousarray(scores[rows, order], dtype=np.float32)

    d_x1 = cuda.to_device(np.ascontiguousarray(boxes_sorted[:, :, 0]))
    d_y1 = cuda.to_device(np.ascontiguousarray(boxes_sorted[:, :, 1]))
    d_x2 = cuda.to_device(np.ascontiguousarray(boxes_sorted[:, :, 2]))
    d_y2 = cuda.to_device(np.ascontiguousarray(boxes_sorted[:, :, 3]))
    d_scores = cuda.to_device(scores_sorted)
    d_iou_max = cuda.device_array((b, n), dtype=np.float32)

    grid = (n, b)
    _iou_max_batch_kernel[grid, _TPB](d_x1, d_y1, d_x2, d_y2, d_iou_max, n)
    # Same default stream, so the second launch is ordered after the first;
    # the explicit synchronize below is what guarantees the host copy is safe.
    method_id = 0 if method == "linear" else 1
    _decay_scores_batch_kernel[grid, _TPB](
        d_x1, d_y1, d_x2, d_y2, d_scores, d_iou_max, n, method_id, np.float32(sigma)
    )
    cuda.synchronize()

    final_scores = d_scores.copy_to_host()
    return [
        order[img][np.flatnonzero(final_scores[img] > score_threshold)].astype(np.int64)
        for img in range(b)
    ]
```

- [ ] **Step 4: Chạy local để chắc import/validation OK**

```bash
.venv/Scripts/python.exe -c "
import sys, numpy as np; sys.path.insert(0,'src')
import gpu_v3
try:
    gpu_v3.run_gpu_v3_batch(np.zeros((2,3,4), np.float32), np.zeros((2,3), np.float32), method='bad')
except ValueError as e:
    print('validation OK:', e)
"
```

- [ ] **Step 5: Commit + push**

```bash
git add src/gpu_v3.py tests/test_batch.py
git commit -m "feat(v3): fused batched Matrix NMS kernels for catalog A4 batch size 32"
git push origin main
```

---

### Task 13: `--batch` trong `run_all.py` + wrapper CPU/V1 để so sánh

**Files:**
- Modify: `src/cpu_baseline.py` (thêm `run_cpu_batch`)
- Modify: `src/gpu_v1.py` (thêm `run_gpu_v1_batch`)
- Modify: `benchmarks/run_all.py`

**Interfaces:**
- Consumes: `run_gpu_v2_batch` (Task 11), `run_gpu_v3_batch` (Task 12).
- Produces: `run_cpu_batch(boxes, scores, iou_threshold=0.5) -> list[np.ndarray]`, `run_gpu_v1_batch(...)` cùng chữ ký; cờ `--batch` cho runner.

- [ ] **Step 1: Wrapper lặp cho CPU và V1**

Thêm vào `src/cpu_baseline.py`:
```python
def run_cpu_batch(boxes, scores, iou_threshold=0.5):
    """Sequential batch wrapper -- NOT a fused kernel.

    Exists so batch-size-32 charts have an honest CPU reference. The GPU
    versions that actually gained a batch dimension are V2 and V3, which is
    what the A4 catalog asks for ('GPU V2: Batched NMS ...').
    """
    return [run_cpu(boxes[b], scores[b], iou_threshold) for b in range(len(scores))]
```

Thêm vào `src/gpu_v1.py`:
```python
def run_gpu_v1_batch(boxes, scores, iou_threshold=0.5):
    """Sequential batch wrapper for V1 -- NOT a fused kernel.

    V1 allocates an N x N IoU matrix per image (~400MB at N=10,000), so a fused
    batch would need ~12.8GB at batch 32 and would not fit a T4. The A4 catalog
    only asks V1 for 'one thread per (box_i, box_j) pair'; batching is V2's job.
    """
    return [run_gpu_v1(boxes[b], scores[b], iou_threshold) for b in range(len(scores))]
```

- [ ] **Step 2: Thêm `--batch` vào runner**

Trong `benchmarks/run_all.py`:

Thêm arg:
```python
    parser.add_argument("--batch", type=int, default=1,
                        help="images per call; >1 uses the batch entry points "
                             "(fused for v2/v3, sequential wrappers for cpu/v1)")
```

Thêm hàm chọn runner batch, đặt cạnh `select_runner`:
```python
def select_batch_runner(version: str):
    if version == "cpu":
        from cpu_baseline import run_cpu_batch
        return run_cpu_batch
    if version == "v1":
        from gpu_v1 import run_gpu_v1_batch
        return run_gpu_v1_batch
    if version == "v2":
        from gpu_v2 import run_gpu_v2_batch
        return run_gpu_v2_batch
    if version == "v3":
        from gpu_v3 import run_gpu_v3_batch
        return lambda boxes, scores: run_gpu_v3_batch(boxes, scores)
    raise ValueError(version)


def load_batch(batch: int, n: int, seed: int):
    """Stack `batch` independently seeded box sets into (B, N, 4) / (B, N)."""
    boxes, scores = [], []
    for b in range(batch):
        bx, sc = load_data(n, seed=seed + b)
        boxes.append(bx)
        scores.append(sc)
    return np.stack(boxes).astype(np.float32), np.stack(scores).astype(np.float32)
```

Sửa vòng đo trong `main()`:
```python
    report["input_semantics"] = (
        "one image / one box set" if args.batch == 1
        else f"batch of {args.batch} images per call; v2/v3 fused, cpu/v1 sequential wrappers"
    )
    for n in args.n:
        if args.batch == 1:
            boxes, scores = load_data(n, seed=args.seed)
        else:
            boxes, scores = load_batch(args.batch, n, args.seed)
        report["results"][str(n)] = {}
        for version in versions:
            runner = select_runner(version) if args.batch == 1 else select_batch_runner(version)
            for _ in range(args.warmup):
                timed_run(runner, boxes, scores)
            samples = [timed_run(runner, boxes, scores) for _ in range(args.repeats)]
            result = summarize(samples)
            report["results"][str(n)][version] = result
            print(
                f"N={n:>6} batch={args.batch:>3} {version:>3}: "
                f"median={result['median_seconds'] * 1e3:9.3f} ms  "
                f"std={result['stddev_seconds'] * 1e3:7.3f} ms"
            )
```

- [ ] **Step 3: Chạy local với CPU để chắc đường batch hoạt động**

```bash
.venv/Scripts/python.exe benchmarks/run_all.py --versions cpu --n 100 --batch 4 --repeats 2 --warmup 1
```
Kỳ vọng: in `N=   100 batch=  4 cpu: median=... ms`, không traceback.

- [ ] **Step 4: Thêm cell benchmark batch-32 vào `src/gpu_v3.ipynb`**

```python
# Catalog A4 performance target: 10,000 boxes at batch size 32 in under 5 ms.
!python benchmarks/run_all.py --versions cpu v2 v3 --n 10000 --batch 32 \
    --warmup 2 --repeats 7 --json benchmarks/results/colab_t4_batch32.json
from google.colab import files
files.download('benchmarks/results/colab_t4_batch32.json')
```

- [ ] **Step 5: Commit + push**

```bash
git add src/cpu_baseline.py src/gpu_v1.py benchmarks/run_all.py src/gpu_v3.ipynb
git commit -m "feat(benchmarks): measure batch-32 latency against the A4 5ms target"
git push origin main
```

---

### 🔶 COLAB GATE #3 — đo mục tiêu chính của catalog

- [ ] **Step 1:** Chạy `pytest tests/test_batch.py -v` trên Colab (thêm cell vào notebook V2 và V3, hoặc chạy trực tiếp). **Mọi test batch phải pass trước khi đo.**
- [ ] **Step 2:** Chạy cell benchmark batch-32 ở `gpu_v3.ipynb`. Tải `colab_t4_batch32.json`.
- [ ] **Step 3:** Đọc kết quả theo đúng 2 câu hỏi của catalog:
  - `median_seconds` của v3 ở N=10.000, batch=32 có < 0.005s không?
  - `cpu_median / v3_median` có nằm trong 30–80× không?
- [ ] **Step 4:** Commit JSON, push. **Dù kết quả không đạt cũng commit** — số không đạt vẫn là dữ liệu, và `presentation/CROSS_GROUP_LESSONS.md:21` đã ghi rõ trung thực về phần chưa đạt tốt hơn né tránh.

---

# PHASE D — Level 3 compute optimization & đồng bộ tài liệu

### Task 14: Warp-level intrinsics cho reduction của V3

Project Description, Level 3 (Week 9-10) yêu cầu *"Warp-level intrinsics, kernel fusion"*. V3 hiện dùng tree reduction qua shared memory với `cuda.syncthreads()` ở cả 8 bước — 5 bước cuối (stride ≤ 32) nằm gọn trong 1 warp nên không cần barrier, thay bằng `cuda.shfl_down_sync` sẽ bỏ được 5 lần đồng bộ toàn block.

**Files:**
- Modify: `src/gpu_v3.py` (`_iou_max_kernel`, `_decay_scores_kernel`, và 2 kernel batch của Task 12)

**Interfaces:**
- Không đổi chữ ký kernel nào — thuần tối ưu nội bộ. Test hiện có (Task 10, 12) là lưới an toàn.

- [ ] **Step 1: Chạy Gate #3 trước để có số "trước khi tối ưu"**

Không có số trước thì không chứng minh được tối ưu có tác dụng. Lấy `median_seconds` của v3 từ `colab_t4_batch32.json` và `colab_t4_v3.json`, ghi vào phần mô tả commit sau.

- [ ] **Step 2: Thay 5 bước cuối của reduction trong `_iou_max_kernel`**

```python
    s_max = cuda.shared.array(shape=(256,), dtype=nb_float32)
    s_max[tx] = local_max
    cuda.syncthreads()

    # Shared-memory tree down to one warp's worth of partials...
    stride = 128
    while stride > 32:
        if tx < stride:
            if s_max[tx + stride] > s_max[tx]:
                s_max[tx] = s_max[tx + stride]
        cuda.syncthreads()
        stride //= 2

    # ...then finish inside a single warp with shuffle intrinsics: lanes of one
    # warp advance in lockstep, so the five remaining steps need no block-wide
    # barrier and no further shared-memory round-trips.
    if tx < 32:
        val = s_max[tx]
        other = s_max[tx + 32]
        if other > val:
            val = other
        for delta in (16, 8, 4, 2, 1):
            peer = cuda.shfl_down_sync(0xFFFFFFFF, val, delta)
            if peer > val:
                val = peer
        if tx == 0:
            iou_max_out[i] = val
```

- [ ] **Step 3: Cùng biến đổi cho `_decay_scores_kernel` (đổi max thành min)**

```python
    if tx < 32:
        val = s_min[tx]
        other = s_min[tx + 32]
        if other < val:
            val = other
        for delta in (16, 8, 4, 2, 1):
            peer = cuda.shfl_down_sync(0xFFFFFFFF, val, delta)
            if peer < val:
                val = peer
        if tx == 0:
            scores[j] = scores[j] * val
```

- [ ] **Step 4: Áp dụng y hệt cho `_iou_max_batch_kernel` và `_decay_scores_batch_kernel`**

Cùng khối code, chỉ khác dòng ghi kết quả (`iou_max_out[b, i] = val` và `scores[b, j] = scores[b, j] * val`).

- [ ] **Step 5: Commit + push, rồi chạy lại Colab Gate #3**

```bash
git add src/gpu_v3.py
git commit -m "perf(v3): finish block reduction with warp shuffle intrinsics"
git push origin main
```

Chạy lại Gate #3, so `median_seconds` trước/sau. **Nếu không nhanh hơn, giữ nguyên code và ghi lại kết quả âm** — đó vẫn là một phát hiện hợp lệ (reduction không phải bottleneck của V3; vòng grid-stride mới là). Ghi vào Task 15.

---

### Task 15: Đồng bộ toàn bộ tài liệu với số đo thật

Sau 3 Colab Gate, mọi con số đã có artefact. Giờ mới được sửa README/slide.

**Files:**
- Modify: `docs/VERIFIED_STATUS.md` (viết lại)
- Modify: `docs/TECHNICAL_DOCUMENTATION.md:7, 342`
- Modify: `presentation/README.md:28-39`, `presentation/OUTLINE_AND_CONTENT.md:226-251`, `presentation/SCRIPT.md:81-85`, `presentation/QA_PREP.md:166-170`
- Modify: `README.md:20-21`

- [ ] **Step 1: Viết lại `docs/VERIFIED_STATUS.md`**

Sửa 3 lỗi hiện có: (a) ghi "15 passed, 35 skipped" trong khi chạy thật là 12 passed; (b) nhắc "test batch orchestration" mà repo không có; (c) ghi Numba 0.66.0/Windows 10 trong khi requirements pin 0.59.1 và máy là Windows 11.

Cấu trúc mới:
```markdown
# Trạng thái đã xác minh

Tài liệu này chỉ ghi kết quả có artefact kèm theo. Mỗi con số phải trỏ được
tới một file trong `benchmarks/results/`.

## Bảng trạng thái

| Hạng mục | Trạng thái | Artefact |
|---|---|---|
| CPU baseline vs torchvision | ... | `benchmarks/results/colab_t4_v1.json` |
| GPU V1 vs CPU (N=10.000) | ... | ... |
| GPU V2 vs CPU/V1/torchvision | ... | ... |
| GPU V3 vs matrix_nms_reference | ... | ... |
| Batch 32 @ N=10.000 < 5ms | ... | `benchmarks/results/colab_t4_batch32.json` |
| NMS share of detector pipeline | ... | `benchmarks/results/detector_pipeline_t4.json` |

## Môi trường đo
<dán khối "environment" từ JSON>

## Kết quả pytest
<dán dòng cuối của pytest ở Colab, kèm ngày chạy>

## Còn thiếu
<liệt kê thẳng những gì vẫn chưa đo được>
```

- [ ] **Step 2: Gỡ mâu thuẫn "V1 đã đo hay chưa"**

`docs/TECHNICAL_DOCUMENTATION.md:7` và `:342` nói "CPU baseline và GPU V1 đã đo thật trên Colab T4"; `docs/VERIFIED_STATUS.md` (bản cũ) nói chưa có số GPU nào. Sau Gate #1 cả hai đều có artefact — sửa cả 2 chỗ để trỏ tới `benchmarks/results/*.json` thay vì trỏ tới notebook.

- [ ] **Step 3: Sửa `presentation/README.md` bảng "Trạng thái số liệu"**

Cột "Speedup đo thật trên Colab T4" hiện trỏ "xem `src/gpu_v1.ipynb`" — notebook đó giờ không còn output. Đổi thành trỏ `benchmarks/results/colab_t4_v1_legacy.md` (số lịch sử) và `benchmarks/results/colab_t4_v1.json` (số mới, có median/stddev). Điền V2/V3/batch-32 bằng số thật.

- [ ] **Step 4: Cập nhật dòng batch-32 ở mọi nơi**

Các chỗ hiện ghi "chưa implement": `README.md:20-21`, `docs/HOW_TO_RUN.md:64-66`, `docs/GPU_VALIDATION_RUNBOOK.md:43-44`, `docs/TECHNICAL_DOCUMENTATION.md:361-362`, `presentation/QA_PREP.md:166-167`, `presentation/SCRIPT.md:85`, `presentation/CROSS_GROUP_LESSONS.md:21`, `presentation/OUTLINE_AND_CONTENT.md:139, 251`.

Đổi sang trạng thái thật sau Gate #3, và **giữ nguyên phần phân biệt** "batched 64 box/khối" vs "batch size 32" — phần đó vẫn đúng và vẫn cần thiết.

- [ ] **Step 5: Cập nhật tỉ lệ cProfile**

`presentation/SCRIPT.md:23` và `presentation/OUTLINE_AND_CONTENT.md:62` đang dùng 65%/34% từ proposal. Giờ đã có `profile_output/cprofile_N10000.txt` tái tạo được (Task 6) — thêm số mới bên cạnh số proposal, giữ cả hai và nói rõ ngữ cảnh đo, đúng như cách `OUTLINE_AND_CONTENT.md:62` đang xử lý cặp số 0.28s/2.49s.

- [ ] **Step 6: Chạy full test lần cuối + commit**

```bash
.venv/Scripts/python.exe -m pytest tests -q
git add docs/ presentation/ README.md
git commit -m "docs: sync every claimed number with a benchmarks/results artefact"
git push origin main
```

---

## Phụ lục — 2 việc cố ý KHÔNG làm

1. **Không nâng N của CPU baseline lên 30.000+.** Project Description Part 4.2 muốn baseline mất 2–30s; hiện N=10.000 chỉ 0.38s local (2.49s trên Colab — vừa chạm đáy). Nâng N sẽ làm V1 OOM (N=30.000 → 3.6GB ma trận IoU) và phá vỡ bộ N ∈ {100, 1000, 10000} mà catalog quy định cho benchmark. Nếu hội đồng hỏi, câu trả lời là: batch 32 × N=10.000 = 320.000 box/lần gọi mới là workload thật của dự án này, và nó vượt xa mốc 2s.

2. **Không làm NMS theo từng class (`batched_nms` kiểu torchvision).** Đã được ghi nhận là ngoài phạm vi ở `presentation/QA_PREP.md:197` và `docs/GLOSSARY.md:18`. Catalog A4 đặt bài toán class-agnostic.
