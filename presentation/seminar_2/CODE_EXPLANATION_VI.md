# Giải thích code trước khi chạy Colab

Tài liệu này giải thích code hiện tại, không thay cho kết quả chạy GPU. Trạng
thái pre-flight local: toàn bộ file Python compile được; `pytest` có **9 pass,
40 skip**. Các skip gồm mọi test CUDA, nên correctness của kernel chỉ được xác
nhận sau khi chạy Colab.

## 1. Luồng chung

Mỗi implementation nhận box dạng `[x1, y1, x2, y2]`, score và ngưỡng IoU.

```text
boxes + scores
      │
      ├── CPU / V1 / V2: hard greedy NMS → index box được giữ
      └── V3: Matrix NMS → score bị decay → index có score vượt ngưỡng
```

`load_data()` tạo box hợp lệ (`x2 > x1`, `y2 > y1`) và score `float32` với
seed cố định. Nhờ vậy các version có thể dùng cùng input khi đối chiếu.

## 2. CPU baseline — `src/cpu_baseline.py`

`run_cpu()` là oracle nội bộ của greedy NMS.

1. `np.argsort(-scores, kind="stable")` sắp xếp score giảm dần. Stable sort
   làm cách xử lý score bằng nhau xác định được.
2. Vòng lặp lấy box chưa bị suppress có score cao nhất tiếp theo.
3. `iou_one_to_many()` tính IoU của box đó với mọi box còn lại.
4. Box có `IoU > threshold` bị đánh dấu suppress.

IoU là `diện tích giao / diện tích hợp`. Không giao nhau thì IoU bằng 0;
hai box trùng nhau thì bằng 1. CPU baseline phải khớp `torchvision.ops.nms`
trước khi dùng làm oracle cho V1/V2.

## 3. GPU V1 — ma trận IoU đầy đủ

`_iou_matrix_kernel` dùng grid 2D. Thread `(i, j)` tính đúng một phần tử
`IoU(box_i, box_j)`, nên phần O(N²) độc lập được chạy song song. Sau đó
`run_gpu_v1()` tải toàn bộ ma trận về host và thực hiện đúng vòng greedy như
CPU bằng các row slice NumPy.

Điểm mạnh là dễ chứng minh đúng: ma trận phải có diagonal bằng 1, đối xứng và
khớp CPU trong sai số `1e-4`. Điểm yếu là ở `N=10,000`, ma trận float32 chiếm
khoảng 400 MB và vẫn phải copy GPU → CPU; vòng greedy cũng chưa biến mất.

## 4. GPU V2 — SoA, bitmask và batch

### 4.1 Input/output

`run_gpu_v2()` vẫn hỗ trợ input cũ:

```python
keep = run_gpu_v2(boxes_N4, scores_N)
```

Với batch, gọi:

```python
keeps = run_gpu_v2(boxes_BN4, scores_BN)
```

Trong đó `boxes_BN4.shape == (B, N, 4)`, `scores_BN.shape == (B, N)`, và
`keeps` là list gồm B mảng index. Mỗi ảnh có số box được giữ khác nhau nên
không thể trả một ma trận K cố định.

### 4.2 Vì sao dùng SoA

AoS lưu box như `(N, 4)`. V2 tách thành bốn mảng `(B, N)` là `x1`, `y1`,
`x2`, `y2`. Khi các lane liên tiếp đọc anchor/target liên tiếp, GPU có thể
gộp các truy cập bộ nhớ tốt hơn. `_batched_boxes_to_soa_device()` thực hiện
chuyển layout trước khi upload.

### 4.3 Bitmask có nghĩa gì

Sau khi sort score, rank nhỏ hơn nghĩa là score cao hơn. Kernel tạo:

```text
mask[batch, target_word, anchor]
```

Mỗi `uint64` có 64 bit. Bit `k = 1` nghĩa là anchor suppress box có rank
`target_word * 64 + k`. Kernel chỉ xét `target > anchor`, đúng với greedy NMS:
box score cao chỉ suppress box score thấp hơn.

Thay vì tải `N × N` float32 IoU, V2 tải `ceil(N/64) × N` uint64 mask. Với
một ảnh N=10,000, mask khoảng 12.5 MB; batch 32 khoảng 400 MB. Đây là giảm
so với V1 theo từng ảnh, nhưng không phải phép màu để đảm bảo `<5 ms`.

### 4.4 Grid và shared memory

Kernel `_nms_bitmask_kernel` launch grid `(num_words, num_words, B)`:

- `blockIdx.x`: word của 64 anchor;
- `blockIdx.y`: word của 64 target;
- `blockIdx.z`: ảnh trong batch;
- `threadIdx.x`: một anchor trong word.

Mỗi block tải 64 target box vào shared memory một lần. Sau barrier, 64 thread
dùng lại target tile đó để tính IoU. Nhánh `target_word < anchor_word` là
đồng nhất trong toàn block nên có thể return an toàn. Với block cuối không đủ
64 box, lane inactive vẫn phải đi qua `cuda.syncthreads()` rồi mới return;
đây là bản sửa quan trọng để tránh divergent barrier.

### 4.5 Vì sao vẫn có CPU loop

`_resolve_greedy_mask()` duyệt anchor theo score. Chỉ khi anchor chưa bị
suppress thì mới OR mask của nó vào `suppressed`. Phụ thuộc này là bản chất
greedy NMS; V2 chỉ đưa phần tính quan hệ pairwise lên GPU. Vì vậy V2 không
được mô tả là “fully on-device greedy NMS”.

## 5. GPU V3 — Matrix NMS

V3 đổi thuật toán thay vì cố song song hoá hoàn toàn greedy NMS.

1. `_iou_max_kernel`: mỗi block xử lý một box `i`; 256 thread chia nhau quét
   các box có score cao hơn và reduction lấy IoU lớn nhất.
2. `_decay_scores_kernel`: mỗi block xử lý một box `j`; tính decay nghiêm
   khắc nhất từ các box score cao hơn, rồi nhân vào score của `j`.
3. Host tải final score về và lọc theo `score_threshold`.

V3 là soft suppression. Nó **không phải** và không cần khớp index với
`torchvision.ops.nms`; oracle đúng là `matrix_nms_reference()` trong cùng
file. `fastmath=True` có thể làm khác số rất nhỏ gần ngưỡng, nên test Colab
phải là nguồn phán quyết.

## 6. Kết luận correctness trước Colab

| Thành phần | Đã có bằng chứng local | Còn phải chứng minh trên CUDA |
|---|---|---|
| CPU IoU + greedy NMS | Unit test pass | Match torchvision ops.nms |
| V1 | Static review, test đã viết | JIT, ma trận IoU và kept set |
| V2 | Static review, input/batch/mask mapping, syntax | JIT, barrier, B=32 và kept set |
| V3 | CPU Matrix-NMS oracle pass | JIT, decay kernel và match oracle |

Không có lỗi logic hiển nhiên trong luồng V1/V2/V3 sau review này. Tuy nhiên,
không có CUDA thì không thể xác nhận Numba compile, launch configuration,
shared-memory behavior hoặc performance. Đó là lý do phải chạy đúng thứ tự
trong [`COLAB_MANUAL_TEST.md`](COLAB_MANUAL_TEST.md).
