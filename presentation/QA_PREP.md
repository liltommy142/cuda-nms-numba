# Chuẩn bị Hỏi-Đáp chi tiết

> Mục tiêu: **hiểu** để trả lời tự nhiên, không phải học thuộc lòng. Mỗi câu có phần "Vì sao" để nắm được lý do đằng sau — nếu bị hỏi biến thể khác vẫn trả lời được. Viết lại và mở rộng từ `QA_PREP.md` + `PRESENTATION_NOTES.md` cũ (đã xoá), cập nhật cho đúng trạng thái V1+V2+V3 đã có code, và bổ sung câu hỏi suy ra từ `CROSS_GROUP_LESSONS.md`.

---

## A. Khái niệm cơ bản

**Q: NMS là gì, giải thích đơn giản?**
NMS (Non-Maximum Suppression) là bước "dọn dẹp" sau khi mô hình phát hiện vật thể. Model như YOLO vẽ ra hàng nghìn khung ứng viên quanh cùng 1 vật thể. NMS sắp xếp theo độ tự tin, giữ khung tự tin nhất, xoá các khung chồng lấp nhiều lên nó, rồi lặp lại với khung tự tin tiếp theo còn sót — cho đến khi mỗi vật thể chỉ còn đúng 1 khung.

**Q: IoU là gì, tính như thế nào?**
IoU = Intersection over Union = diện tích giao / diện tích hợp của 2 khung. 0 (không chạm) → 1 (trùng khít). IoU cao → khả năng cao đang trùng lặp → loại bớt 1 khung.

**Q: Vì sao NMS truyền thống là O(n²)?**
Với mỗi khung được giữ, thuật toán phải so với tất cả khung còn lại chưa bị loại. Trường hợp xấu nhất (không khung nào bị loại sớm): n khung × so sánh ~n khung = n².

**Q: CUDA / kernel / thread / block / grid là gì?**
- CUDA: nền tảng NVIDIA cho code chạy trực tiếp trên GPU.
- Kernel: 1 hàm chạy song song — hàng nghìn bản sao chạy đồng thời, mỗi bản xử lý 1 phần dữ liệu.
- Thread: đơn vị thực thi nhỏ nhất, biết "tôi là ai" qua toạ độ riêng (`cuda.grid(...)`).
- Block: nhóm thread (16×16=256 ở kernel 2D của V1/V2; 256 threads/block ở kernel 1D của V3).
- Grid: toàn bộ tập block cần để phủ hết dữ liệu.

**Q: Vì sao dùng Numba thay vì CUDA C/C++?**
Ràng buộc của môn học — `@cuda.jit` là công cụ chính thức được yêu cầu, không phải nhóm tự chọn. Cho phép viết kernel CUDA bằng cú pháp gần NumPy, dễ đối chiếu trực tiếp với bản CPU.

---

## B. Vì sao bài toán hợp GPU

**Q: "Embarrassingly parallel" nghĩa là gì, và vì sao tính IoU thuộc loại này?**
Các phần việc hoàn toàn độc lập, không cần chờ đợi/trao đổi kết quả. `IoU(box_i, box_j)` không phụ thuộc bất kỳ cặp nào khác — giao N² thread tính N² cặp cùng lúc mà không cần `cuda.syncthreads()` giữa các cặp khác nhau.

**Q: Vậy tại sao không song song hoá luôn suppression ngay từ đầu (V1)?**
Vì có phụ thuộc dữ liệu tuần tự thật sự: "box B có bị loại không" phụ thuộc "box A (điểm cao hơn) đã được giữ hay chưa" — mà quyết định về A lại phụ thuộc các box cao điểm hơn A. Đây là lý do cả dự án cần 3 phiên bản thay vì làm xong ngay 1 bước — mỗi bước xử lý phần phụ thuộc này theo cách khác nhau (xem câu hỏi ở mục E).

---

## C. Thiết kế GPU V1 (IoU song song, suppression tuần tự trên host)

**Q: V1 chỉ song song hoá phần nào?**
Chỉ ma trận IoU N×N. Suppression vẫn chạy tuần tự trên CPU — khác CPU baseline ở chỗ chỉ tra bảng O(1) mỗi lần thay vì tính lại IoU.

**Q: Vì sao block size 16×16 = 256 thread?**
256 là bội số của warp size = 32 (đơn vị lập lịch cơ bản của GPU NVIDIA) → không lãng phí thread nào trong warp. 16×16 cũng khớp tự nhiên với bài toán 2 chiều (ma trận N×N).

**Q: "Bounds guard" (`if i >= n or j >= n: return`) để làm gì?**
Số block GPU cấp luôn làm tròn lên (ceil) — N không chia hết cho 16 sẽ dư thread ở rìa. Không chặn thì các thread thừa ghi/đọc ra ngoài mảng.

**Q: Vì sao sort box theo score TRƯỚC KHI đưa lên GPU?**
Suppression cần duyệt theo thứ tự điểm giảm dần. Sort trước giúp hàng thứ `i` của ma trận IoU tự động tương ứng thứ hạng `i` — vòng lặp chỉ cần chỉ số liên tiếp `i+1:`.

**Q: Vì sao tính TOÀN BỘ ma trận IoU N×N?**
Đánh đổi có chủ đích: tốn thêm bộ nhớ O(n²) để đổi lấy tốc độ — tận dụng tối đa GPU cho phần song song hoàn hảo, suppression trên CPU chỉ còn tra bảng.

**Q: Đánh đổi đó có giới hạn gì?**
Có — N=10.000 → ma trận ~400MB (chấp nhận được, T4 có 16GB VRAM). N=100.000 → ~40GB, vượt VRAM hầu hết GPU miễn phí. Đây chính là động lực cho V2 (không truyền cả ma trận về).

**Q: `cuda.synchronize()` để làm gì, bỏ được không?**
Lệnh phát kernel không chặn (non-blocking) — CPU chạy tiếp ngay. Không `synchronize()` trước khi đọc kết quả → race condition (đọc dữ liệu GPU chưa ghi xong). Không thể bỏ.

---

## D. Thiết kế GPU V2 (coalesced SoA + bitmask parallel reduction)

**Q: V2 khác V1 ở đâu, cụ thể?**
2 điểm: (1) đổi layout box từ AoS (mảng gộp 4 toạ độ/box) sang **SoA** — 4 mảng riêng `x1,y1,x2,y2` — giúp các thread liền kề trong 1 warp đọc đúng ô nhớ liền kề (coalesced access), gom thành 1 giao dịch bộ nhớ thay vì nhiều giao dịch rời rạc. (2) Suppression: GPU tự tính và nén sẵn "ai suppress ai" thành **bitmask uint64** (`_nms_bitmask_kernel`) — chỉ cần tải về ~N²/64 phần tử thay vì cả ma trận N² số thực như V1.

**Q: Coalesced memory access nghĩa là gì, ví dụ dễ hiểu?**
Các thread cạnh nhau đọc dữ liệu ở vị trí bộ nhớ gần nhau → GPU gom lại đọc 1 lần cho cả nhóm, thay vì đọc rải rác nhiều lần. Giống đi chợ mua đồ ở các sạp liền kề nhau thay vì chạy khắp chợ.

**Q: Bitmask hoạt động cụ thể ra sao?**
Kernel `_nms_bitmask_kernel` chia N box thành các khối 64 (do dùng số nguyên 64-bit). Mỗi thread phụ trách 1 box `i` (anchor), so với 64 box trong 1 khối cột `by` — nếu box `i` suppress box `j` (IoU > threshold và `j` có điểm thấp hơn `i`), bật đúng bit thứ `k` tương ứng trong số nguyên 64-bit đó. Kết quả: `mask_out[by, i]` — bit thứ `k` bật nghĩa là "box `i` suppress box thứ `k` trong khối `by`". Có `by < bx: return` để bỏ qua sớm các khối chắc chắn không cần tính (tối ưu, không phải lỗi).

**Q: V2 có loại bỏ hoàn toàn vòng lặp tuần tự trên CPU không?**
**Không hoàn toàn** — đây là điểm cần trả lời trung thực nếu bị hỏi sâu. Việc **tính ra** bitmask (ai-suppress-ai) đã 100% song song trên GPU. Nhưng bước cuối — quyết định thứ hạng nào thực sự được giữ theo đúng thứ tự điểm số — vẫn là 1 vòng `for i in range(n)` chạy trên CPU (`run_gpu_v2`), vì đây vẫn là chuỗi phụ thuộc tuần tự thật (không thể biết box thứ i có bị giữ hay không mà không biết trạng thái các box điểm cao hơn nó trước đó). Khác biệt so với V1: mỗi vòng lặp giờ chỉ OR 2 mảng ngắn (~N/64 phần tử, kiểu uint64) thay vì so sánh 1 hàng dài N phần tử số thực như V1 — nhẹ hơn nhiều, không phải nhanh hơn về mặt Big-O của vòng lặp ngoài (vẫn N lần lặp Python).

**Q: Parallel reduction ở đây là gì?**
Chỗ dùng đúng nghĩa "reduction": các bit suppress được TÍNH song song trên toàn bộ N² cặp cùng lúc rồi "gộp" lại thành 1 bitmask cô đọng — khác cách làm tuần tự "suppress từng box một" của baseline. Ví dụ dễ hiểu: gộp nhiều giá trị (ở đây là các quyết định suppress) bằng chia nhỏ tính song song rồi gộp, thay vì tính tuần tự từng cái.

---

## E. Thiết kế GPU V3 (Matrix NMS — soft suppression, không còn vòng lặp CPU)

**Q: Matrix NMS khác NMS truyền thống thế nào?**
NMS truyền thống (V1/V2) là **hard suppression** — loại bỏ hẳn (0/1) box IoU vượt ngưỡng, quyết định phải làm tuần tự theo thứ tự điểm số. Matrix NMS (Wang et al. 2020) dùng **soft suppression** — không loại bỏ, mà **giảm dần điểm số** (decay factor) theo mức chồng lấp. Vì công thức decay của mỗi box chỉ phụ thuộc thông tin đã biết trước (điểm cao hơn nó), toàn bộ phép tính có thể làm song song hoàn toàn, không cần biết box nào "đã được xử lý trước" theo nghĩa tuần tự.

**Q: Cụ thể V3 chạy 2 kernel để làm gì, sao không gộp 1 kernel?**
- Kernel 1 (`_iou_max_kernel`): mỗi box `i` (1 block/box, 256 thread) tính `iou_max[i]` = IoU lớn nhất giữa box `i` và **mọi box điểm cao hơn nó**. Các thread trong block chia nhau quét, dùng shared memory + parallel reduction (tree, stride giảm dần 128→1) để gộp ra max cuối.
- Kernel 2 (`_decay_scores_kernel`): mỗi box `j` tính hệ số suy giảm điểm số của chính nó, dựa trên `iou_max[i]` của các box điểm cao hơn nó — **đã tính sẵn ở kernel 1**.

Phải tách 2 kernel vì kernel 2 cần `iou_max` của TẤT CẢ box đã tính xong trước khi bắt đầu — đây là điểm đồng bộ bắt buộc giữa 2 giai đoạn (giống `cuda.synchronize()` ở V1 nhưng ở quy mô toàn bộ kernel, không phải trong 1 kernel). Bên trong mỗi kernel thì N block chạy hoàn toàn song song, không có box nào phải "đợi" box khác trong cùng kernel.

**Q: "Mathematical pruning" trong code nghĩa là gì?**
Theo công thức Matrix NMS, hệ số decay của box `j` với box `i` (điểm cao hơn) **chỉ nhỏ hơn 1** khi `IoU(i,j) > iou_max[i]`. Nếu không thoả điều kiện này, decay = 1 (không đổi gì) — code kiểm tra điều kiện đó TRƯỚC, chỉ tính `exp()`/phép chia (đắt) khi thật sự cần, bỏ qua ~99% cặp không ảnh hưởng. Đây là tối ưu hiệu năng, không phải rút gọn thuật toán.

**Q: `score_threshold`, `method` (linear/gaussian), `sigma` trong V3 dùng để làm gì?**
- `score_threshold`: sau khi mọi box đã bị giảm điểm, chỉ giữ box có điểm còn lại > ngưỡng này (mặc định 0.05) — đây là tiêu chí "giữ/loại" cuối cùng của V3, khác hẳn V1/V2 (dựa trên index/suppress trực tiếp).
- `method`: `linear` hay `gaussian` — 2 công thức decay khác nhau (Gaussian mượt hơn, giảm điểm theo hàm mũ; linear giảm tuyến tính).
- `sigma`: tham số độ "mượt" của decay kiểu Gaussian.

---

## F. Trade-off chất lượng của V3 — chuẩn bị kỹ, dễ bị hỏi

**Q: V3 có lợi ích nào ngoài tốc độ không?**
Có — và đây là điểm nên chủ động nêu ra, không chỉ nói về tốc độ: hard suppression (V1/V2) có 1 lỗi thật khi 2 vật thể **khác nhau** đứng sát nhau (IoU giữa 2 box thật cao dù là 2 vật thể riêng biệt) — thuật toán có thể xoá nhầm 1 trong 2 box đúng vì chỉ nhìn IoU vượt ngưỡng là loại thẳng. V3 (soft suppression) chỉ giảm điểm theo mức chồng lấp thay vì xoá hẳn, nên nếu điểm ban đầu đủ cao, cả 2 box vẫn có thể sống sót sau khi giảm điểm — giải quyết đúng trường hợp lỗi này của hard NMS. Nói cách khác, V3 không chỉ là "phiên bản nhanh hơn" mà còn "đúng hơn" trong tình huống vật thể chồng lấp tự nhiên (không phải box trùng lặp giả).

**Q: V3 nhanh hơn nhưng có đánh đổi gì không?**
Có, và nhóm em chủ động nói rõ: vì đổi từ hard suppression sang soft suppression, **tập box mà V3 giữ lại không còn khớp y hệt như CPU baseline/V1/V2** khi so theo index như cách nhóm em vẫn kiểm chứng V1/V2 (khớp 100% với `torchvision.ops.nms`). V3 dùng tiêu chí khác — ngưỡng điểm số sau khi đã giảm dần — nên về bản chất đang trả lời một câu hỏi hơi khác: "điểm tin cậy còn lại của mỗi box sau khi trừ hao phần chồng lấp là bao nhiêu", không phải "giữ hay loại". Test hiện tại của V3 (`test_gpu_v3_sanity`) chỉ kiểm tra tính hợp lý cơ bản (không lỗi, suppress đúng case rõ ràng), chưa so khớp trực tiếp với CPU baseline như V1/V2 — vì về mặt thiết kế, không nên kỳ vọng khớp.

**Q: Vậy làm sao biết V3 "đúng"?**
Đúng theo đúng công thức Matrix NMS gốc (Wang et al. 2020), không đúng theo nghĩa "giống hệt NMS cứng". Cách đánh giá đúng đắn hơn cho hướng này (nếu mở rộng ngoài phạm vi đồ án) là đo mAP (mean Average Precision) trên tập object-detection thật — nhóm em chưa làm phần này (ngoài phạm vi catalog A4, vốn chỉ yêu cầu tốc độ + so khớp cơ bản), nhưng hiểu đây là hướng đánh giá đúng nếu được hỏi sâu.

**Q: Vì sao không làm V3 luôn từ đầu, đỡ qua V1/V2?**
Đây là lộ trình học có chủ đích của môn: đi từ naive (V1, dễ làm nhưng vẫn còn phần tuần tự) → tối ưu bộ nhớ/nén dữ liệu (V2, vẫn hard suppression, vẫn còn 1 vòng lặp CPU nhẹ) → giải pháp triệt để nhất (V3, đổi hẳn thuật toán để loại bỏ hoàn toàn vòng lặp CPU). Ngoài ra V3 khó nhất và rủi ro tiến độ cao nhất nên xếp làm mục tiêu stretch (125%), không đặt cược toàn bộ điểm vào nó.

---

## G. Đúng đắn / kiểm thử

**Q: Làm sao biết GPU tính đúng?**
So với `torchvision.ops.nms` (ground truth ngoài, kiểm chứng rộng rãi). 2 lớp: (1) từng giá trị IoU của kernel khớp công thức CPU trong dung sai 1e-4; (2) **tập box giữ lại** khớp giữa CPU và GPU (áp dụng cho V1/V2 — V3 xem mục F).

**Q: Vì sao dung sai 1e-4 thay vì khớp tuyệt đối?**
Số thực máy tính không tuyệt đối chính xác — CPU (NumPy vector hoá) và GPU (scalar operations) có thể lệch nhau vài ULP do khác thứ tự phép cộng/trừ. Hiện tượng bình thường, không phải lỗi logic.

**Q: Vì sao cần stable sort?**
Khi 2 box điểm bằng nhau tuyệt đối, sort không ổn định có thể xếp thứ tự khác nhau giữa các lần chạy → kết quả không tất định. Stable sort giữ nguyên thứ tự gốc khi bằng điểm, đảm bảo CPU/GPU cho kết quả khớp nhau khi so sánh.

**Q: Bộ test hiện tại có đáng tin hoàn toàn chưa?**
Chưa 100% — nhóm em tự rà lại code trước buổi seminar và phát hiện 1 bug: 4 test của V2 (`test_gpu_v2_iou_matrix_*`) gọi hàm `compute_iou_matrix_gpu_v2` nhưng hàm này lúc đó chưa được viết trong `gpu_v2.py`, nên sẽ lỗi `ImportError` nếu chạy trên máy có GPU thật. Nhóm đã bổ sung hàm này (wrapper gọi kernel coalesced, cùng kiểu với `compute_iou_matrix_gpu` của V1). Đây là ví dụ thực tế cho việc bộ test chưa từng chạy trên GPU thật (`gpu_v2.ipynb`/`gpu_v3.ipynb` chưa có cell output nào) — cần chạy lại trên Colab để xác nhận PASS thật, không chỉ hết lỗi cú pháp.

---

## H. Hiệu năng / benchmark

**Q: Vì sao ở N=100, GPU chỉ nhanh 1.2 lần?**
Mỗi lần gọi GPU tốn "phí cố định" (overhead): khởi động kernel, truyền dữ liệu qua PCIe. N nhỏ → phần việc thật sự ít, phí cố định chiếm tỷ trọng lớn, ăn hết lợi ích song song.

**Q: GPU V1 có giảm độ phức tạp O(n²) xuống không?**
Không. Về Big-O, GPU V1 vẫn O(n²) — chỉ chia cho p thread chạy cùng lúc, thời gian thực tế ≈ O(n²/p). Cải thiện nằm ở hằng số nhân, không đổi bậc phức tạp (khác V3 — V3 thay đổi bản chất bài toán, không chỉ chia việc).

**Q: Đo thời gian có tính JIT compile không?**
Không — đo bằng `time.perf_counter()` bao quanh hàm, nhưng có bước "warm-up" (gọi thử 1 lần nhỏ trước) để loại phần biên dịch JIT lần đầu ra khỏi phép đo cho công bằng.

**Q: Nhóm đo cProfile 2 lần (Colab và máy local) mà tỉ lệ suppression/IoU khác nhau — vậy số nào đúng?**
Cả 2 đều đúng — chênh lệch đến từ khác phần cứng, không phải sai số đo. Trên Colab: ~65% IoU / ~35% suppression. Trên máy local (CPU nhanh hơn): ~59% suppression / ~40% IoU (xem `presentation/cprofile_N10000_local.txt`). CPU nhanh hơn giúp NumPy vector hoá (`iou_one_to_many`) chạy nhanh hơn tương đối, khiến phần vòng lặp Python thuần (`run_cpu`) nổi lên chiếm tỉ trọng lớn hơn. Điều quan trọng không đổi ở cả 2 lần đo: **cả 2 phần đều chiếm tỉ trọng đáng kể**, và **bản thân thuật toán NMS, không phải I/O, là bottleneck** — đây mới là kết luận nhóm dùng để biện minh cho việc đưa NMS lên GPU, không phải con số phần trăm chính xác tuyệt đối.

**Q: Bottleneck thật ở N lớn là tính toán hay truyền dữ liệu?**
Ở V1, truyền cả ma trận IoU N×N về CPU qua PCIe có thể vượt cả thời gian GPU tính toán, vì dung lượng tăng O(n²) trong khi thời gian tính giảm dần theo số thread. Đây chính là động lực thiết kế của V2 (nén còn O(n²/64) qua bitmask).

---

## I. Kế hoạch / phạm vi dự án

**Q: Vì sao 3 mức mục tiêu 75/100/125% thay vì 1 mục tiêu?**
Có phương án dự phòng: nếu V3 (khó nhất) không kịp, vẫn đạt 100% chỉ với V1+V2 — tránh "được ăn cả, ngã về không".

**Q: Vì sao batch size = 32?**
Không phải nhóm tự chọn — do catalog đề tài A4 quy định sẵn ("process 10.000 boxes at batch size 32 in under 5ms"), nhóm bám theo để so sánh được với chuẩn chấm điểm. **Hiện chưa implement** ở bất kỳ version nào — mỗi lần chạy vẫn xử lý 1 tập box, đây là việc còn thiếu nhóm nói thẳng ở slide 8.

**Q: Rủi ro nào nhóm lo nhất?**
(1) V2/V3 chưa từng benchmark trên GPU thật — số liệu kỳ vọng trong code có thể sai lệch so với thực đo; (2) sai số floating-point/tie-break có thể đổi tập box giữ lại dù IoU gần giống hệt (đã biết trước, chấp nhận dung sai 1e-4 + stable sort); (3) batch size 32 chưa làm — cần thêm thời gian trước khi nộp bản cuối.

---

## J. Câu hỏi "bẫy"

**Q: Sao không dùng luôn `torchvision.ops.nms`?**
Dùng làm ground truth để kiểm chứng, không phải giải pháp cuối. Mục tiêu môn học là tự viết và hiểu kernel CUDA. Trong thực tế, framework production (TensorRT...) cũng tự viết kernel NMS riêng tối ưu, không dùng bản NumPy tuần tự.

**Q: N nhỏ (ví dụ 10 box) có nên dùng GPU không?**
Không — như N=100 chỉ nhanh 1.2×, phí khởi động lấn át lợi ích. Production thường có ngưỡng: N nhỏ chạy CPU, N đủ lớn mới đẩy lên GPU.

**Q: Tính cả ma trận N×N (V1) có lãng phí không vì nhiều cặp chắc chắn IoU=0?**
Đúng, đây là điểm lãng phí thật của V1. Chấp nhận được vì: (1) phép tính IoU rất rẻ, (2) đổi lại code cực đơn giản, song song hoá toàn bộ mà không cần biết trước cấu trúc không gian box. Thiết kế nâng cao hơn (ngoài phạm vi đồ án) có thể dùng cấu trúc không gian (lưới/cây) để bỏ qua trước — đó là tối ưu thuật toán khác, không phải mục tiêu chính (mục tiêu là song song hoá).

---

## Mẹo khi trả lời (nếu không chắc)

- Không biết chắc → nói thật: "Phần này em chưa triển khai/đo thực tế, nhưng theo lý thuyết thì..." — được đánh giá cao hơn bịa số (xem bài học chung ở `CROSS_GROUP_LESSONS.md`).
- Nếu bị hỏi quá sâu về dòng code cụ thể của V2/V3 mà chưa kịp nhớ — nhắc lại: code đã hoàn chỉnh, đang ở giai đoạn chờ verify benchmark thật, sẵn sàng demo trực tiếp nếu cần.
- Nếu quên số liệu chính xác — nói đúng bậc độ lớn ("khoảng 250 mili giây", "tầm 10 lần") vẫn tốt hơn đứng im.
