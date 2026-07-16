# Bảng thuật ngữ (Glossary)

> 🧭 Về [docs/INDEX.md](INDEX.md) · Giải thích kỹ thuật đầy đủ hơn ở [docs/TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
>
> Trang này gom mọi thuật ngữ kỹ thuật xuất hiện trong repo (code, docs, tài liệu thuyết trình) vào một chỗ duy nhất, để không phải giải thích lại rải rác ở nhiều file. Mỗi thuật ngữ có cột "Xem thêm" trỏ tới nơi giải thích sâu hơn/dùng thật trong code.

---

## A. Khái niệm NMS cơ bản

| Thuật ngữ | Giải thích | Xem thêm |
|---|---|---|
| **NMS (Non-Maximum Suppression)** | Thuật toán hậu xử lý sau object detection: loại bỏ các box dự đoán trùng lặp, chỉ giữ lại box có độ tin cậy (score) cao nhất tại mỗi vị trí. | [TECHNICAL_DOCUMENTATION §2.1](TECHNICAL_DOCUMENTATION.md#21-nms-là-gì) |
| **IoU (Intersection over Union)** | Tỉ lệ diện tích vùng giao trên diện tích vùng hợp của 2 box — đo mức độ "chồng lấp"; 0 = không chạm, 1 = trùng khít. | [TECHNICAL_DOCUMENTATION §2.1](TECHNICAL_DOCUMENTATION.md#21-nms-là-gì), `cpu_baseline.py:iou_one_to_many` |
| **Greedy algorithm (thuật toán tham lam)** | Chiến lược "tốt nhất tại từng bước, không xét lại" — ở đây là luôn giữ box điểm cao nhất còn lại rồi loại các box chồng lấp nó. Đây là cách CPU baseline, V1, V2 đều làm (khác V3). | `cpu_baseline.py:run_cpu` |
| **Hard suppression (triệt tiêu cứng)** | Cách Greedy NMS loại box: quyết định nhị phân giữ/xoá dứt khoát khi IoU vượt ngưỡng — cách CPU baseline, V1, V2 dùng. Đối lập với **soft suppression** (mục D bên dưới). | [mục D](#d-matrix-nms--soft-suppression-v3) |
| **score_threshold** | Ngưỡng điểm số dùng ở V3: sau khi mọi box đã bị giảm điểm (decay), chỉ giữ box có điểm còn lại lớn hơn ngưỡng này (mặc định 0.05) — đây là tiêu chí "giữ/loại" cuối cùng của V3, khác V1/V2 (dựa trên suppress trực tiếp theo IoU, không qua ngưỡng điểm). | `gpu_v3.py:run_gpu_v3_matrix_nms` |
| **Class-agnostic NMS** | NMS chạy trên 1 tập box/score chung, không phân biệt object thuộc class nào — cách cả 3 version trong repo này làm. Khác với NMS theo từng class (`batched_nms` trong torchvision, offset toạ độ theo class để 2 object khác loại đứng gần nhau không bị suppress nhầm) — đây là hướng mở rộng ngoài phạm vi đồ án, xem `presentation/QA_PREP.md` mục K. | `presentation/QA_PREP.md` mục K |
| **Stable sort (sắp xếp ổn định)** | Thuật toán sắp xếp giữ nguyên thứ tự tương đối của các phần tử có giá trị bằng nhau — đảm bảo mọi cài đặt (`run_cpu`, `run_gpu_v1`, `run_gpu_v2`, `run_gpu_v3_matrix_nms`) cho kết quả tất định (deterministic) khi nhiều box cùng score, vì tất cả đều dùng chung `np.argsort(-scores, kind="stable")`. | Tất cả 4 file trong `src/` |
| **Big-O / độ phức tạp thuật toán** | Ký hiệu mô tả tốc độ tăng của thời gian chạy theo kích thước input N — ví dụ O(n²) nghĩa là thời gian tăng theo bình phương N. Dùng để so sánh CPU baseline (O(n²) tuần tự) với các bản GPU (cùng O(n²) công việc nhưng chia cho p thread chạy song song). | [TECHNICAL_DOCUMENTATION §3.2](TECHNICAL_DOCUMENTATION.md#32-phân-tích-độ-phức-tạp-thuật-toán-big-o) |

## B. CUDA và phần cứng GPU — khái niệm nền tảng

| Thuật ngữ | Giải thích | Xem thêm |
|---|---|---|
| **CUDA** | Nền tảng lập trình song song của NVIDIA, cho phép chạy code trực tiếp trên GPU. | [TECHNICAL_DOCUMENTATION §2.3](TECHNICAL_DOCUMENTATION.md#23-gpu_v1py--giải-thích-từng-hàm-và-cuda-kernel) |
| **Kernel** | Hàm viết để chạy song song trên GPU — hàng nghìn thread cùng thực thi, mỗi thread một bản sao, xử lý dữ liệu khác nhau. Trong repo: `_iou_matrix_kernel` (V1), `_iou_matrix_coalesced_kernel`/`_nms_bitmask_kernel` (V2), `_iou_max_kernel`/`_decay_scores_kernel` (V3). | [mục 2.3](TECHNICAL_DOCUMENTATION.md#23-gpu_v1py--giải-thích-từng-hàm-và-cuda-kernel), [2.5](TECHNICAL_DOCUMENTATION.md#25-gpu_v2py--giải-thích-từng-hàm-và-cuda-kernel), [2.6](TECHNICAL_DOCUMENTATION.md#26-gpu_v3py--giải-thích-từng-hàm-và-cuda-kernel) |
| **Thread** | Đơn vị thực thi nhỏ nhất trên GPU — biết "tôi là ai" qua toạ độ riêng (`cuda.grid(...)`, `cuda.threadIdx`). | như trên |
| **Block** | Một nhóm thread (16×16=256 ở kernel 2D của V1/V2; 256 ở kernel 1D của V3). Thread trong cùng block có thể chia sẻ **shared memory** và đồng bộ hoá (`cuda.syncthreads()`). | như trên |
| **Grid** | Toàn bộ tập block cần để phủ hết dữ liệu — `ceil(N/16) × ceil(N/16)` block ở V1/V2 (2D), hoặc đúng N block ở V3 (1D). | như trên |
| **Warp** | Đơn vị lập lịch cơ bản của GPU NVIDIA — 32 thread thực thi đồng thời theo cùng 1 lệnh. Block size 256 = bội số của 32 → không lãng phí thread nào trong warp. | `presentation/QA_PREP.md` mục C |
| **Host / Device** | "Host" = CPU và RAM hệ thống; "Device" = GPU và VRAM. Dữ liệu phải copy tường minh giữa 2 bên (`cuda.to_device`, `copy_to_host`). | [mục 2.3](TECHNICAL_DOCUMENTATION.md#23-gpu_v1py--giải-thích-từng-hàm-và-cuda-kernel) |
| **PCIe** | Đường truyền vật lý giữa Host và Device — băng thông thấp hơn nhiều so với bộ nhớ GPU nội bộ, thường là điểm nghẽn khi chuyển dữ liệu lớn (như ma trận IoU N×N ở V1). | [TECHNICAL_DOCUMENTATION §3.3](TECHNICAL_DOCUMENTATION.md#33-ghi-chú-hiệu-năng-performance-bottlenecks-và-lưu-ý-khi-mở-rộng) |
| **JIT (Just-In-Time compilation)** | Biên dịch code thành mã máy **ngay khi cần dùng lần đầu** — Numba dùng cơ chế này cho `@cuda.jit`, nên lần gọi đầu luôn chậm hơn (cần "warm-up" trước khi benchmark). | như trên |
| **Shared memory** | Bộ nhớ nhanh, dùng chung giữa các thread trong **cùng 1 block** — dùng để cache dữ liệu đọc nhiều lần (V2's `_nms_bitmask_kernel` cache 64 box của block cột) hoặc làm vùng trung gian cho **parallel reduction** (V3). | [mục C](#c-kỹ-thuật-tối-ưu-bộ-nhớ--song-song-hoá-v2v3) |
| **fastmath** | Cờ bật cho `@cuda.jit(fastmath=True)` (dùng ở V3) — cho phép trình biên dịch dùng phép toán gần đúng/nhanh hơn (vd. reciprocal xấp xỉ), đổi lại độ chính xác/determinism tuyệt đối. Chấp nhận được vì dự án đã dùng dung sai 1e-4 khi so khớp kết quả. | `gpu_v3.py`, `presentation/QA_PREP.md` mục K |
| **Bounds guard** | Câu lệnh kiểm tra chỉ số trong giới hạn hợp lệ trước khi truy cập mảng — bắt buộc vì grid luôn làm tròn **lên** (`ceil`), có thể dư thread ở rìa nếu N không chia hết cho kích thước block. | `if i >= n or j >= n: return` trong mọi kernel |
| **Grid-stride loop** | Vòng lặp nội bộ trong 1 kernel để chia đều công việc cho số thread cố định của block, bất kể phần việc lớn hay nhỏ hơn số thread — dùng ở `_iou_max_kernel`/`_decay_scores_kernel` (V3): `for k in range(tx, i, cuda.blockDim.x)`. | [mục 2.6](TECHNICAL_DOCUMENTATION.md#26-gpu_v3py--giải-thích-từng-hàm-và-cuda-kernel) |

## C. Kỹ thuật tối ưu bộ nhớ & song song hoá (V2/V3)

| Thuật ngữ | Giải thích | Xem thêm |
|---|---|---|
| **Embarrassingly parallel** | Bài toán mà các đơn vị công việc hoàn toàn độc lập, không cần giao tiếp/đồng bộ giữa chúng. Tính IoU giữa mọi cặp box thuộc loại này. | [TECHNICAL_DOCUMENTATION §1.2](TECHNICAL_DOCUMENTATION.md#12-vấn-đề-cần-giải-quyết) |
| **Vectorization (vector hoá)** | Thay vòng lặp Python từng phần tử bằng 1 phép toán trên toàn bộ mảng (NumPy) — chạy ở tốc độ mã C thay vì tốc độ thông dịch Python. Dùng ở suppression loop của V1 (`suppressed[i+1:] |= iou_matrix[i, i+1:] > iou_threshold`). | [mục 2.3](TECHNICAL_DOCUMENTATION.md#23-gpu_v1py--giải-thích-từng-hàm-và-cuda-kernel) |
| **SoA / AoS (Structure of Arrays / Array of Structures)** | AoS: 1 mảng `(N,4)` gộp cả 4 toạ độ mỗi box (V1). SoA: 4 mảng `x1[N],y1[N],x2[N],y2[N]` riêng (V2). SoA giúp các thread liền kề trong 1 warp đọc đúng ô nhớ liền kề (xem **coalesced memory access** ngay dưới). | [mục 2.5](TECHNICAL_DOCUMENTATION.md#25-gpu_v2py--giải-thích-từng-hàm-và-cuda-kernel), [3.5](TECHNICAL_DOCUMENTATION.md#35-sơ-đồ-luồng-dữ-liệu--gpu-v2) |
| **Coalesced memory access (truy cập bộ nhớ liền mạch)** | Các thread cạnh nhau trong 1 warp đọc dữ liệu ở vị trí bộ nhớ gần nhau → GPU gộp lại thành 1 giao dịch bộ nhớ thay vì nhiều giao dịch rời rạc. V2 đạt được điều này nhờ đổi sang layout SoA. | như trên |
| **Bitmask suppression** | Kỹ thuật nén kết quả "ai suppress ai" thành số nguyên 64-bit (mỗi bit = 1 quyết định suppress) thay vì ma trận IoU số thực đầy đủ — giảm dữ liệu tải Host↔Device ~64 lần. Dùng ở V2's `_nms_bitmask_kernel`. | [mục 2.5](TECHNICAL_DOCUMENTATION.md#25-gpu_v2py--giải-thích-từng-hàm-và-cuda-kernel) |
| **Parallel reduction (rút gọn song song) / Tree reduction** | Kỹ thuật gộp N giá trị (vd. tìm max/min) bằng cách chia đôi số phần tử đang so sánh sau mỗi bước (256→128→64→...→1), thay vì so sánh tuần tự từng cặp — dùng ở V3's `_iou_max_kernel` (tìm max) và `_decay_scores_kernel` (tìm min), mỗi bước cần `cuda.syncthreads()` để đảm bảo mọi thread thấy được kết quả bước trước. | [mục 2.6](TECHNICAL_DOCUMENTATION.md#26-gpu_v3py--giải-thích-từng-hàm-và-cuda-kernel) |

## D. Matrix NMS & soft suppression (V3)

| Thuật ngữ | Giải thích | Xem thêm |
|---|---|---|
| **Soft suppression (triệt tiêu mềm)** | Thay vì xoá hẳn box khi IoU vượt ngưỡng (hard suppression), giảm dần điểm số (score) của box theo mức độ chồng lấp — box vẫn có thể "sống sót" nếu điểm ban đầu đủ cao. Nền tảng của cả Soft-NMS và Matrix NMS. | [mục 2.6](TECHNICAL_DOCUMENTATION.md#26-gpu_v3py--giải-thích-từng-hàm-và-cuda-kernel) |
| **Decay factor (hệ số suy giảm)** | Con số nhân vào score gốc của 1 box để "làm mờ dần" điểm số — càng chồng lấp nhiều với box điểm cao hơn, decay factor càng nhỏ (gần 0); không chồng lấp thì decay = 1 (không đổi). Công thức: `Score_new = Score_old × Decay_Factor`. | `gpu_v3.py:_decay_scores_kernel` |
| **Matrix NMS** (Wang et al., 2020) | Biến thể NMS mà repo này cài đặt ở V3: thay hard suppression bằng soft suppression, tính decay factor của mỗi box hoàn toàn độc lập dựa trên `iou_max` (IoU lớn nhất với box điểm cao hơn) — cho phép song song hoá 100%, không còn vòng lặp CPU nào. | [mục 2.6](TECHNICAL_DOCUMENTATION.md#26-gpu_v3py--giải-thích-từng-hàm-và-cuda-kernel) |
| **Soft-NMS** (Bodla et al., 2017) | Một hướng tiếp cận khác cũng dùng soft suppression, được liệt kê trong tài liệu tham khảo của proposal — **không phải** thuật toán V3 cài đặt (V3 dùng Matrix NMS), nhưng cùng ý tưởng nền tảng "giảm điểm thay vì xoá hẳn". Dễ nhầm lẫn tên gọi với Matrix NMS nên tách riêng ở đây. | `CSC14116 - Proposal.docx` |
| **method: linear / gaussian** | 2 công thức decay khác nhau mà `run_gpu_v3_matrix_nms` hỗ trợ — `linear`: giảm tuyến tính theo tỉ lệ IoU; `gaussian`: giảm theo hàm mũ (mượt hơn), tham số độ mượt là **sigma**. | `gpu_v3.py:_decay_scores_kernel` |
| **iou_max[i]** | Giá trị IoU lớn nhất giữa box `i` và bất kỳ box nào có điểm cao hơn nó — tính ở kernel đầu tiên (`_iou_max_kernel`), dùng làm input cho kernel thứ hai (`_decay_scores_kernel`) tính decay factor. | `gpu_v3.py:_iou_max_kernel` |

## E. Đo lường & kiểm thử

| Thuật ngữ | Giải thích | Xem thêm |
|---|---|---|
| **Warm-up (làm nóng)** | Một lần gọi kernel nhỏ trước khi đo thời gian thật, để loại bỏ chi phí biên dịch JIT lần đầu ra khỏi phép đo — mọi hàm `benchmark()` trong repo đều làm bước này. | [mục 2.3](TECHNICAL_DOCUMENTATION.md#23-gpu_v1py--giải-thích-từng-hàm-và-cuda-kernel) |
| **Ground truth (bên ngoài)** | Kết quả tham chiếu để kiểm chứng đúng-sai — repo dùng `torchvision.ops.nms` (thư viện đã được kiểm chứng rộng rãi) làm ground truth cho V1/V2. | `tests/test_correctness.py` |
| **Dung sai (tolerance / atol)** | Sai số chấp nhận được khi so khớp 2 giá trị số thực — repo dùng `1e-4` cho IoU, vì CPU (NumPy vector hoá) và GPU (scalar operations) có thể lệch nhau vài ULP do khác thứ tự phép cộng/trừ, không phải lỗi logic. | `tests/test_correctness.py` |
| **Determinism (tính tất định)** | Chạy nhiều lần cho cùng 1 kết quả — đảm bảo bằng **stable sort** (mục A) khi có nhiều box cùng score. | mục A |
| **[CHỜ COLAB] / [kỳ vọng, chưa verify]** | Nhãn dùng xuyên suốt tài liệu thuyết trình để đánh dấu số liệu **chưa đo thật** trên GPU — nguyên tắc của nhóm là không tự bịa số, thà nói "chưa đo" còn hơn số sai. | `presentation/README.md` mục "Trạng thái số liệu" |
