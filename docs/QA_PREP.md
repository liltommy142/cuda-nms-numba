# Chuẩn bị Hỏi-Đáp chi tiết

> Mục tiêu: **hiểu** để trả lời tự nhiên, không phải học thuộc lòng từng chữ. Mỗi câu có phần "Vì sao" để bạn nắm được lý do đằng sau — nếu thầy hỏi biến thể khác, bạn vẫn trả lời được vì hiểu bản chất chứ không phải nhớ máy móc.

---

## A. Khái niệm cơ bản — chắc chắn bị hỏi nếu có bạn không rành GPU

**Q: NMS là gì, giải thích đơn giản?**
NMS (Non-Maximum Suppression) là bước "dọn dẹp" sau khi mô hình phát hiện vật thể. Mô hình như YOLO thường vẽ ra hàng nghìn khung hình ứng viên quanh cùng 1 vật thể (vì nó không chắc chắn 100% khung nào đúng nhất). NMS sắp xếp các khung theo độ tự tin, giữ khung tự tin nhất, xóa các khung khác chồng lấp nhiều lên nó, rồi lặp lại với khung tự tin tiếp theo còn sót — cho đến khi mỗi vật thể chỉ còn đúng 1 khung.

**Q: IoU là gì, tính như thế nào?**
IoU = Intersection over Union = (diện tích phần giao nhau) / (diện tích phần hợp nhất) của 2 khung hình. Giá trị từ 0 (không chạm nhau) đến 1 (trùng khít). Dùng để đo "2 khung này có đang chỉ cùng 1 vật thể không" — IoU cao → khả năng cao là trùng lặp → loại bớt 1 khung.

**Q: Vì sao NMS truyền thống là O(n²)?**
Vì với mỗi khung được giữ lại, thuật toán phải so sánh nó với **tất cả** khung còn lại chưa bị loại để tính IoU. Trường hợp xấu nhất (không khung nào bị loại sớm): n khung được giữ × so sánh với ~n khung còn lại = n×n = n² phép so sánh.

**Q: CUDA / kernel / thread / block / grid là gì?**
- **CUDA**: nền tảng của NVIDIA cho phép code chạy trực tiếp trên GPU.
- **Kernel**: 1 hàm được viết để chạy song song — hàng nghìn bản sao của cùng hàm này chạy đồng thời, mỗi bản xử lý 1 phần dữ liệu khác nhau.
- **Thread**: 1 đơn vị thực thi nhỏ nhất, mỗi thread chạy 1 bản kernel, biết "tôi là ai" qua tọa độ riêng.
- **Block**: 1 nhóm thread (ở đây 16×16 = 256 thread/block).
- **Grid**: toàn bộ tập hợp block cần để phủ hết dữ liệu.

**Q: Vì sao dự án dùng Numba thay vì CUDA C/C++?**
Đây là ràng buộc của môn học — Numba `@cuda.jit` là công cụ chính thức được yêu cầu dùng, không phải nhóm tự chọn. Numba cho phép viết kernel CUDA bằng cú pháp Python gần giống NumPy, dễ đối chiếu trực tiếp với bản CPU.

---

## B. Vì sao bài toán này hợp với GPU

**Q: "Embarrassingly parallel" nghĩa là gì, và vì sao tính IoU thuộc loại này?**
Là loại bài toán mà các phần việc **hoàn toàn độc lập nhau**, không cần chờ đợi hay trao đổi kết quả. IoU(box_i, box_j) không phụ thuộc vào bất kỳ cặp nào khác đang được tính — nên có thể giao N² thread tính N² cặp cùng lúc mà không lo tranh chấp dữ liệu hay cần đồng bộ hóa (`cuda.syncthreads()`).

**Q: Vậy tại sao không song song hóa luôn cả phần suppression?**
Vì phần suppression có **phụ thuộc dữ liệu tuần tự** thật sự: quyết định "box B có bị loại không" phụ thuộc vào việc "box A (điểm cao hơn) đã được giữ lại hay chưa" — mà quyết định về A lại phụ thuộc vào các box điểm cao hơn A nữa. Đây là chuỗi phụ thuộc (data dependency chain), không thể tính song song ngây thơ được. Đây chính là lý do cả dự án cần 3 phiên bản GPU (V1→V2→V3) thay vì làm xong ngay trong 1 bước.

**Q: GPU V1 chỉ song song hóa phần nào?**
Chỉ phần tính ma trận IoU N×N. Sau khi có ma trận, phần suppression **vẫn chạy tuần tự trên CPU** — chỉ khác là tra bảng có sẵn (O(1) mỗi lần) thay vì phải tính lại IoU (điều CPU baseline phải làm). Đây là lý do V1 được gọi là "naive" — chưa đụng vào phần khó (suppression).

---

## C. Câu hỏi đào sâu vào code / thiết kế cụ thể

**Q: Vì sao chọn block size 16×16 = 256 thread, không phải số khác?**
256 là bội số của **warp size = 32** (đơn vị lập lịch cơ bản của GPU NVIDIA — mỗi lần GPU thực thi 1 "warp" gồm đúng 32 thread cùng lúc). Chọn bội số của 32 giúp không lãng phí thread nào trong warp. 16×16 cũng là cách chia tự nhiên cho bài toán 2 chiều (ma trận N×N) — mỗi block phủ 1 ô vuông 16×16 của ma trận.

**Q: "Bounds guard" (`if i >= n or j >= n: return`) để làm gì?**
Vì số block GPU cấp luôn được làm tròn **lên** (ceil) — ví dụ N=100 không chia hết cho 16, GPU sẽ cấp dư ra một số thread "thừa" ở rìa (block cuối cùng có thể có thread ứng với chỉ số vượt quá 100). Không chặn thì các thread thừa này sẽ ghi/đọc ra ngoài mảng → lỗi hoặc dữ liệu rác.

**Q: Vì sao phải sort box theo score TRƯỚC KHI đưa lên GPU?**
Vì bước suppression cần duyệt theo thứ tự điểm giảm dần. Nếu sort trước, hàng thứ `i` của ma trận IoU tự động tương ứng với thứ hạng `i` — vòng lặp suppression chỉ cần dùng chỉ số liên tiếp `i+1:` để lấy "tất cả box có điểm thấp hơn box hiện tại", không cần tra cứu gián tiếp qua mảng thứ tự nào khác. Giúp code đơn giản và nhanh hơn.

**Q: Vì sao tính TOÀN BỘ ma trận IoU N×N, không chỉ tính khi cần?**
Vì phần tính IoU là phần song song hoàn hảo — dồn hết việc đó cho GPU tận dụng tối đa hàng nghìn thread cùng lúc. Đổi lại, phần suppression trên CPU chỉ còn là tra bảng có sẵn, không cần tính toán gì thêm. Đây là đánh đổi có chủ đích: tốn thêm bộ nhớ (N² ô) để đổi lấy tốc độ.

**Q: Đánh đổi đó có giới hạn gì không?**
Có — bộ nhớ ma trận tăng theo O(n²). N=10.000 → ma trận ~400MB (chấp nhận được, GPU T4 có 16GB VRAM). Nhưng N=100.000 → ~40GB, vượt xa VRAM của hầu hết GPU miễn phí. Đây là lý do thiết kế "tính toàn bộ ma trận" **không mở rộng được (scale)** lên N cực lớn — điểm mà GPU V2/V3 (không lưu toàn bộ ma trận cùng lúc) sẽ cải thiện.

**Q: `cuda.synchronize()` để làm gì, bỏ được không?**
Lệnh phát kernel cho GPU chạy là **bất đồng bộ (non-blocking)** — CPU ra lệnh xong là chạy tiếp ngay, không đợi GPU làm xong. Nếu bỏ `synchronize()`, CPU có thể đọc kết quả ma trận IoU **trước khi** GPU ghi xong → dữ liệu sai/thiếu (race condition). Không thể bỏ.

**Q: Vì sao công thức IoU viết lại 2 lần (1 lần NumPy cho CPU, 1 lần scalar cho kernel) thay vì dùng chung 1 hàm?**
Vì code chạy **bên trong** kernel CUDA (`@cuda.jit`) bị giới hạn — không được gọi hàm NumPy cấp cao, chỉ dùng được phép toán scalar (`max`, `min`...) mà Numba biên dịch được sang mã máy GPU. Nên phải viết lại logic bằng tay, dù công thức toán học giống hệt nhau.

---

## D. Câu hỏi về tính đúng đắn / kiểm thử

**Q: Làm sao biết GPU tính đúng?**
So sánh kết quả với `torchvision.ops.nms` — thư viện NMS đã được kiểm chứng rộng rãi trong ngành, dùng làm "ground truth" bên ngoài. Có 2 lớp kiểm tra: (1) từng giá trị IoU của kernel phải khớp với công thức CPU trong dung sai 1e-4; (2) **tập hợp box cuối cùng được giữ lại** phải khớp giữa CPU và GPU.

**Q: Vì sao chấp nhận dung sai 1e-4 thay vì đòi khớp tuyệt đối?**
Vì máy tính lưu số thập phân không tuyệt đối chính xác — 2 cách tính khác thứ tự phép cộng/trừ (CPU dùng NumPy vector hóa, GPU dùng scalar operations) có thể cho kết quả lệch nhau ở chữ số rất nhỏ (vài ULP — đơn vị sai số nhỏ nhất máy tính có thể biểu diễn). Đây là hiện tượng bình thường trong tính toán số, không phải lỗi logic.

**Q: Vì sao cần "stable sort" (sắp xếp ổn định)?**
Khi 2 box có điểm số **bằng nhau tuyệt đối**, sort thường có thể xếp chúng theo thứ tự bất kỳ (không tất định) — chạy 2 lần cùng dữ liệu có thể ra 2 kết quả khác nhau. Stable sort đảm bảo giữ nguyên thứ tự gốc khi bằng điểm, giúp cả CPU và GPU cho kết quả **tất định (deterministic)** và khớp nhau khi so sánh.

**Q: Nếu IoU của CPU và GPU khớp nhau nhưng tập box giữ lại lại khác nhau thì sao?**
Đây là rủi ro đã lường trước trong proposal (mục Risk Analysis): khi 2 box có điểm số rất gần nhau (near-tie), 1 sai số cực nhỏ trong IoU cũng có thể làm đảo ngược thứ tự "ai được giữ trước", dẫn đến toàn bộ chuỗi suppression phía sau khác đi — dù từng giá trị IoU vẫn nằm trong dung sai cho phép. Nhóm coi đây là hiện tượng đã biết trước (documented), không phải bug, và báo cáo thêm "tỉ lệ khớp tập box" bên cạnh sai số IoU.

---

## E. Câu hỏi về hiệu năng / benchmark

**Q: Vì sao ở N=100, GPU chỉ nhanh 1.2 lần — gần như không có tác dụng?**
Vì mỗi lần gọi GPU đều tốn 1 khoản "phí cố định" (overhead): khởi động kernel, truyền dữ liệu qua lại giữa CPU-GPU qua đường PCIe. Khi N nhỏ, phần việc thật sự cần tính toán rất ít, nên khoản phí cố định này chiếm tỷ trọng lớn, ăn hết lợi ích song song hóa. GPU chỉ thật sự "đáng đồng tiền" khi khối lượng việc đủ lớn để bù lại chi phí khởi động.

**Q: GPU V1 có làm giảm độ phức tạp O(n²) xuống không?**
**Không.** Về mặt lý thuyết (Big-O), GPU V1 vẫn là O(n²) — chỉ là O(n²) đó được chia cho p thread chạy cùng lúc, nên thời gian thực tế ≈ O(n²/p). Cải thiện nằm ở **hằng số nhân** (tốc độ thực thi), không phải đổi sang bậc phức tạp thấp hơn như O(n log n). Đây là điểm hay bị hỏi để kiểm tra hiểu sâu — nhiều người nhầm "dùng GPU" với "giảm độ phức tạp thuật toán", 2 khái niệm khác nhau.

**Q: Đo thời gian bao gồm những gì, có tính cả JIT compile không?**
Đo bằng `time.perf_counter()` bao quanh toàn bộ hàm — gồm sort, truyền dữ liệu lên/xuống GPU, chạy kernel, và vòng lặp suppression. **Không tính** thời gian biên dịch JIT lần đầu (Numba compile kernel thành mã máy khi gọi lần đầu) — nhóm chủ động "warm-up" (gọi thử 1 lần nhỏ trước khi đo) để loại phần này ra, tránh làm sai lệch kết quả đo.

**Q: Bottleneck thật sự ở N lớn là tính toán hay truyền dữ liệu?**
Ở N rất lớn, việc truyền **toàn bộ ma trận IoU N×N** từ GPU về CPU qua đường PCIe có thể trở thành bottleneck lớn hơn cả bản thân việc GPU tính toán — vì dung lượng truyền tăng theo O(n²) trong khi thời gian tính giảm dần nhờ càng nhiều thread càng nhanh. Đây là 1 trong các giới hạn được ghi rõ trong tài liệu kỹ thuật của nhóm, và là động lực để V2/V3 tránh truyền cả ma trận về.

---

## F. Câu hỏi về kế hoạch GPU V2 / V3 (chưa code, cần hiểu ý tưởng để không bị hỏi bí)

**Q: GPU V2 sẽ làm gì khác V1?**
2 điểm chính: (1) **Batched NMS** — xử lý nhiều ảnh (nhiều tập box) cùng lúc thay vì từng ảnh 1; (2) dùng **parallel reduction** để xây dựng "mặt nạ suppression" (suppressed mask) song song thay vì vòng lặp tuần tự `for i in range(n)` hiện tại trên CPU.

**Q: Parallel reduction là gì, ví dụ dễ hiểu?**
Kỹ thuật gộp nhiều giá trị lại thành 1 kết quả (tổng, max, OR...) bằng cách chia đôi và gộp dần theo cặp, thay vì gộp tuần tự từng cái một. Ví dụ: 8 người muốn biết tổng tiền cả nhóm — thay vì 1 người cộng lần lượt 8 số (7 bước), chia 4 cặp cộng song song trước, rồi 2 cặp kết quả cộng tiếp, chỉ 3 bước. Số bước giảm từ O(n) xuống O(log n).

**Q: Matrix NMS (GPU V3) khác gì so với NMS truyền thống?**
NMS truyền thống là **hard suppression** — loại bỏ hẳn (0 hoặc 1) box có IoU vượt ngưỡng, và quyết định này phải làm tuần tự theo thứ tự điểm số. Matrix NMS (Wang et al. 2020) dùng **soft suppression** — thay vì loại bỏ cứng, nó **giảm dần điểm số** (decay factor) của box dựa trên mức độ chồng lấp, và toàn bộ phép tính decay này có thể biểu diễn dưới dạng phép toán ma trận — tính được **hoàn toàn song song, không cần biết thứ tự box nào được xử lý trước**. Đây là lý do nó loại bỏ được chuỗi phụ thuộc tuần tự.

**Q: Vì sao không làm V3 luôn từ đầu, đỡ phải qua V1/V2?**
Vì đây là bài học có chủ đích của môn: đi từ naive (V1, dễ làm nhưng vẫn còn phần tuần tự) → tối ưu bộ nhớ/song song hóa dần (V2) → giải pháp triệt để nhất (V3, khó nhất, đòi hỏi hiểu thuật toán mới). Ngoài ra V3 rủi ro cao nhất về tiến độ nên nhóm xếp làm mục tiêu stretch (125%), không đặt cược toàn bộ điểm số vào nó.

---

## G. Câu hỏi về quản lý dự án / phạm vi

**Q: Vì sao chọn 3 mức mục tiêu 75/100/125% thay vì chỉ 1 mục tiêu duy nhất?**
Để có phương án dự phòng thực tế: nếu GPU V3 (khó nhất, phụ thuộc học thêm 1 thuật toán mới) không kịp tiến độ, nhóm vẫn đảm bảo đạt mốc 100% chỉ với V1+V2 — tránh tình huống "được ăn cả, ngã về không".

**Q: Vì sao batch size chọn 32?**
Đây là mốc do **catalog đề tài chính thức của môn (đề A4)** quy định sẵn, không phải nhóm tự chọn — nhóm bám theo để kết quả so sánh được trực tiếp với chuẩn chấm điểm.

**Q: Hiện tại đã đạt <5ms chưa?**
Chưa — mục tiêu đó thuộc mốc 125% (V3), chưa triển khai. GPU V1 hiện đo được 255.7ms tại N=10.000, cách mục tiêu khoảng 51 lần — là kết quả bình thường ở giai đoạn naive, không phải dấu hiệu trễ tiến độ.

**Q: Rủi ro nào nhóm lo nhất, và phương án dự phòng?**
3 rủi ro đã xác định: (1) Colab có thể chặn quyền dùng công cụ profiling chuyên sâu (`nvprof`) → dùng wall-clock timing thay thế; (2) sai số floating-point/tie-break có thể làm tập box giữ lại khác nhau dù IoU gần giống hệt → chấp nhận dung sai 1e-4 và stable sort; (3) V3 khó nhất, xếp lịch cuối → đã scope là stretch goal, không ảnh hưởng mốc 100%.

---

## H. Câu hỏi "bẫy" — kiểm tra hiểu sâu, hay gặp ở giảng viên khó tính

**Q: Sao không dùng luôn `torchvision.ops.nms` cho xong, viết lại làm gì?**
`torchvision.ops.nms` được dùng làm **ground truth để kiểm chứng**, không phải giải pháp cuối. Mục tiêu môn học là tự tay viết và hiểu kernel CUDA — hiểu được vì sao NMS chậm và cách GPU tăng tốc nó, không phải chỉ gọi thư viện có sẵn. Trong thực tế, các framework production (TensorRT, ...) cũng tự viết kernel NMS tối ưu riêng thay vì dùng bản NumPy tuần tự — đây là bài toán có giá trị thực tế trong ngành.

**Q: Nếu N nhỏ (ví dụ ảnh chỉ có 10 box) thì có nên dùng GPU không?**
Không nên — như đã thấy ở benchmark N=100, GPU chỉ nhanh hơn 1.2 lần vì chi phí khởi động lấn át lợi ích. Với N rất nhỏ, chạy trên CPU thậm chí có thể nhanh hơn. Đây là lý do trong thực tế, hệ thống production thường có ngưỡng: N nhỏ thì chạy CPU, N đủ lớn mới đẩy lên GPU.

**Q: Việc "tính cả ma trận IoU N×N" có lãng phí không, vì nhiều cặp sẽ không bao giờ liên quan tới nhau (box ở 2 góc ảnh xa nhau)?**
Đúng, đây là điểm lãng phí thật sự của thiết kế V1 — tính cả cặp box cách xa nhau, chắc chắn IoU=0, vẫn tốn 1 phép tính. Nhưng đánh đổi này chấp nhận được vì (1) phép tính IoU rất rẻ (vài phép so sánh số), (2) đổi lại code cực kỳ đơn giản, dễ song song hóa toàn bộ mà không cần biết trước cấu trúc không gian của box. Các thiết kế nâng cao hơn (ngoài phạm vi đồ án) có thể dùng cấu trúc không gian (lưới/cây) để bỏ qua trước các cặp chắc chắn không giao nhau — nhưng đó là tối ưu thuật toán, không phải mục tiêu chính của bài (mục tiêu là song song hóa).

**Q: Từ O(n²) làm sao biết đây là "worst case", trường hợp tốt thì sao?**
Trường hợp tốt là khi nhiều box bị loại sớm (suppress) — phần `remaining` trong vòng lặp co lại nhanh, ít phép so sánh hơn. Trường hợp xấu nhất là khi hầu như không box nào chồng lấp nhau (mọi box đều được giữ) — khi đó mỗi box giữ lại vẫn phải so với gần như toàn bộ box còn lại. Bảng benchmark trong proposal (100/1.000/10.000 box) cho thấy thời gian tăng "gần bậc hai" chứ không tuyệt đối bậc hai — vì dữ liệu synthetic có 1 phần box bị loại sớm trong thực tế.

---

## Mẹo khi trả lời (nếu không chắc câu trả lời)

- Không biết chắc → nói thật: *"Phần này em chưa triển khai/đo thực tế, nhưng theo lý thuyết thì..."* — thầy cô đánh giá cao sự trung thực hơn là bịa.
- Nếu bị hỏi về phần GPU V2/V3 (chưa code) quá chi tiết (ví dụ dòng code cụ thể) → nhắc lại: đây là proposal, phần đó đang ở giai đoạn thiết kế ý tưởng, chưa cài đặt, và trỏ về đúng mục tiêu 100%/125% đã đề ra.
- Nếu quên số liệu chính xác → nói ước lượng đúng bậc độ lớn ("khoảng 250ms", "tầm 10 lần") vẫn tốt hơn là đứng im.
