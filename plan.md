# Kế hoạch nhóm — Proposal môn Applied Parallel Programming (CSC14116)
## Topic: A4 · Real-Time Non-Maximum Suppression (NMS)

---

## 0. Việc cần xác nhận với giảng viên TRƯỚC KHI làm (do tài liệu mâu thuẫn nhau)

- [ ] Baseline nộp dạng `.py` (theo Catalog) hay `.ipynb` (theo Project Description + slide Introduction)?
- [ ] Deadline chính xác của proposal (check Moodle)
- [ ] Team size cho phép: 2 hay 2-3 người?
- [ ] Cách nộp proposal: điền đúng file `Proposal_template.docx` hay theo cấu trúc trong Catalog PDF?

👉 Trong lúc chờ trả lời, kế hoạch dưới đây làm theo hướng **an toàn nhất: thỏa mãn cả 3 nguồn cùng lúc** — không mục nào bị thiếu dù thầy kiểm tra theo tiêu chí nào.

---

## 1. Cấu trúc thật của Proposal (theo file .docx chính thức) — đây là cái phải điền

File gốc dùng khung của CMU 15-418 (Problem/Background/Challenge/Resources/Goals). Dưới đây là **nội dung cụ thể cho A4 (NMS)** để điền vào từng mục:

### 〈PROJECT NAME〉
Gợi ý: **"GPU-Accelerated Non-Maximum Suppression for Real-Time Object Detection"**

### Group name / List of members
Điền tên nhóm + họ tên + MSSV từng thành viên.

### Keywords (5 từ khóa)
`CUDA`, `Numba`, `Non-Maximum Suppression`, `Object Detection`, `IoU Parallelization`

### List of references
- Bodla et al. (2017), Soft-NMS
- Wang et al. (2020), SOLOv2 (Matrix NMS)
- Hosang et al. (2017), Learning NMS
- torchvision.ops.nms / box_iou source code
- NVIDIA TensorRT NMS plugin (tham khảo, không copy)

---

### 1. Problem Statement

**Problem** (3-5 câu):
> Object detector hiện đại (YOLO, SSD) sinh ra hàng nghìn candidate bounding box mỗi ảnh. NMS lọc các box này xuống danh sách detection cuối cùng bằng cách loại bỏ box chồng lấn (overlap) cao với box điểm tin cậy cao hơn. Trên CPU, thuật toán greedy NMS chạy tuần tự O(n²) và trở thành nút thắt throughput khi số lượng box lớn (batch size lớn, ảnh nhiều vật thể). Bài toán này phù hợp GPU vì việc tính IoU giữa mọi cặp box là hoàn toàn độc lập giữa các cặp — có thể tính song song hàng loạt cặp cùng lúc.

**Dataset / Input:**
- Dataset: Sinh synthetic bounding box bằng `numpy.random`, hoặc dùng box thật xuất ra từ model **YOLOv5s pretrained** chạy trên vài chục ảnh COCO validation.
- Input size benchmark: N ∈ {100, 1,000, 10,000} box, batch size 32 (theo đúng target trong catalog).
- Cách load: `torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)` để lấy box thật; hoặc `numpy.random.uniform` để sinh box giả lập kiểm tra hiệu năng thuần túy.

**Why GPU-suitable:**
> Phép toán trung tâm là tính IoU (Intersection over Union) giữa mọi cặp (box_i, box_j) — đây là phép toán độc lập hoàn toàn giữa các cặp, có thể gán 1 thread cho 1 cặp box (N² thread cho N box). Đây là bài toán "embarrassingly parallel" kinh điển, tương tự tính khoảng cách giữa mọi cặp điểm.

---

### 2. Background

Mô tả pipeline + pseudocode ngắn:

```
Input: boxes[N], scores[N], IoU_threshold
1. Sắp xếp box theo scores giảm dần
2. Với mỗi box (theo thứ tự score giảm dần):
   - Nếu box chưa bị suppress:
     - Giữ box này lại (là 1 detection cuối cùng)
     - Với mọi box còn lại có score thấp hơn:
         Tính IoU(box_hiện_tại, box_khác)
         Nếu IoU > threshold: đánh dấu suppress box_khác
Output: danh sách box được giữ lại
```

Phần **có thể song song hóa**: bước tính IoU giữa mọi cặp (box_i, box_j) — đây là ma trận N×N độc lập, tính được đồng thời toàn bộ trên GPU (1 thread/cặp). Phần **khó song song hóa tự nhiên**: bước suppression tuần tự (phụ thuộc kết quả của box trước để quyết định box sau) — đây chính là "the challenge" ở mục 3.

---

### 3. The Challenge

> Thách thức chính không nằm ở việc tính IoU (dễ song song), mà ở bước **suppression tuần tự**: quyết định giữ/loại box B phụ thuộc vào việc box A (điểm cao hơn) đã bị giữ lại hay chưa — đây là một chuỗi phụ thuộc dữ liệu (data dependency), không thể song song hóa trực tiếp theo kiểu naive. Để giải quyết, nhóm sẽ tìm hiểu và cài đặt **Matrix NMS** (Wang et al. 2020) — kỹ thuật thay thế suppression tuần tự bằng một phép tính soft-suppression toàn phần song song, dựa trên decay factor thay vì loại bỏ cứng. Đây cũng là điều nhóm muốn học: cách biến một thuật toán có phụ thuộc tuần tự thành công thức có thể tính song song hoàn toàn.

---

### 4. Resources

- **Code base xuất phát**: Bắt đầu từ scratch cho CPU baseline (NumPy); dùng `torchvision.ops.nms` và `torchvision.ops.box_iou` làm reference để verify correctness — không copy code, chỉ dùng để so sánh kết quả.
- **Model pretrained**: YOLOv5s (`torch.hub`) để lấy box thật cho test case thực tế.
- **Paper tham khảo**: Bodla et al. 2017 (Soft-NMS), Wang et al. 2020 (Matrix NMS, Section 3.3) — dùng cho thiết kế GPU V3.
- **Máy tính**: Google Colab / Kaggle Notebook (GPU miễn phí) — theo đúng công cụ được dạy trong slide 02-Process_Tool.
- **Ngôn ngữ/thư viện GPU**: Numba (`@cuda.jit`) theo đúng công cụ chính thức của môn (không dùng raw CUDA C/C++).
- **Chưa có, cần tìm thêm**: Chưa rõ cách benchmark chính xác GPU activity trên Colab bằng `nvprof` (lệnh này có thể không chạy được trên Colab do hạn chế quyền) — cần thử nghiệm trước.

---

### 5. Goals and Deliverables

**100% (mục tiêu bắt buộc phải đạt):**
- CPU baseline chạy đúng, verify khớp với `torchvision.ops.nms` (dung sai 1e-4)
- GPU V1: parallel IoU matrix kernel (1 thread/cặp box) — đúng nhưng chưa cần nhanh
- GPU V2: batched NMS với parallel reduction cho suppression mask, coalesced memory access
- Đạt tối thiểu **15× speedup** so với CPU NMS baseline ở N=10,000 box (mức bảo thủ, thấp hơn target chính thức 30-80× để chắc ăn)
- *Lý do tin đạt được*: vì bước IoU là embarrassingly parallel, chỉ riêng việc chuyển từ vòng lặp Python sang kernel GPU cho phép tính hàng nghìn cặp đồng thời đã mang lại tăng tốc đáng kể ở N lớn.

**125% (nếu vượt tiến độ):**
- GPU V3: cài đặt đầy đủ Matrix NMS (Wang et al. 2020), đạt target chính thức của catalog: **30-80× speedup**, xử lý 10,000 box/batch 32 dưới 5ms
- So sánh thêm Soft-NMS vs Matrix NMS về trade-off tốc độ/độ chính xác

**75% (nếu tiến độ chậm hơn dự kiến):**
- Chỉ hoàn thành GPU V1 (naive IoU kernel), đo được correctness + baseline speedup, chưa tối ưu memory/compute
- Vẫn có đủ số liệu để phân tích tại sao chưa đạt target và hướng cải thiện

**Demo tại buổi seminar:**
- Biểu đồ so sánh latency CPU vs GPU V1 vs V2 (vs V3 nếu có) theo N ∈ {100, 1000, 10,000}
- Demo trực quan: ảnh có nhiều box chồng lấn → sau NMS chỉ còn box đúng, chạy real-time nếu đạt target
- Bảng so sánh kết quả detection cuối cùng giữa CPU và GPU (chứng minh correctness)

---

### Weekly schedule (điền theo timeline thật của lớp — xác nhận lại với giảng viên)

| Việc | Tuần 5-6 | Tuần 7 | Tuần 8 | Tuần 9-10 |
|---|---|---|---|---|
| Người A | CPU baseline + profiling | GPU V1 (naive IoU kernel) | GPU V2 (memory opt) | Benchmark tổng hợp |
| Người B | Viết proposal + Git setup | Verify V1 vs torchvision | GPU V3 (Matrix NMS) | Report + slide + demo |

*(Đổi lại số tuần theo lịch thật môn — vì slide Introduction ghi Tuần 6-13 là present hàng tuần, khác với "Level ladder" theo tuần 5-10 trong Project Description. Cần xác nhận lịch chính xác.)*

---

## 4. 🚨 SPRINT PLAN — Team 2 người, còn 4 ngày 11 giờ đến deadline PROPOSAL

Kế hoạch thực tế cho 4 ngày tới (thay bảng Weekly schedule dài hạn ở trên, vì đó là cho cả kỳ). Chia việc theo **"code" vs "viết"**, chạy song song, ráp lại cuối ngày 2. Tuần này CHƯA cần viết kernel GPU — chỉ cần CPU baseline + proposal.

### Ngày 1 (hôm nay)

| Người A — Code | Người B — Viết & Setup |
|---|---|
| Setup Git repo, venv, `requirements.txt` | Điền trước các mục không cần số liệu: Title, Keywords, References, Problem Statement, Background, The Challenge, Resources |
| Chạy thử `yolov5s` lấy box mẫu thật từ vài ảnh COCO | Soạn khung "Goals and Deliverables" (75/100/125%) theo nội dung mẫu ở mục 1 |
| Viết `cpu_baseline.py` (+ song song 1 bản `.ipynb` để phòng cả 2 định dạng) | **Nhắn hỏi giảng viên 4 câu ở mục 0 ngay hôm nay**, không chờ |

⏰ Mốc cuối ngày 1: `cpu_baseline.py` chạy được, in ra thời gian chạy thật.

### Ngày 2

| Người A — Code | Người B — Viết & Setup |
|---|---|
| Chạy `cProfile`, xuất bảng top hàm tốn thời gian | Dựa vào số liệu profiling của A, viết đoạn phân tích bottleneck (đưa % thời gian NMS chiếm vào mục Background/Challenge) |
| Viết `tests/test_correctness.py` so với `torchvision.ops.nms` | Hoàn thiện bảng Weekly schedule (dùng bảng 2 người phía trên) |
| Push code lên Git, viết README.md | Ghép toàn bộ nội dung vào file `Proposal_template.docx` |

⏰ Mốc cuối ngày 2: Có số liệu profiling thật + draft proposal gần hoàn chỉnh.

### Ngày 3 (buffer + hoàn thiện)

- Cả 2 người cùng đọc lại toàn bộ proposal, đối chiếu checklist mục 2
- Kiểm tra proposal có đủ: performance target là **con số cụ thể**, risk analysis, division of work — dù dùng docx thật, vẫn nhét đủ các ý này vào (xem mục 1)
- Test lại `cpu_baseline.py`/`.ipynb` chạy được trên máy KHÔNG có GPU (bắt buộc theo catalog — lỗi là 0 điểm)
- Xuất file PDF/Word cuối, kiểm tra chính tả

### Ngày 4 (buffer cuối, ~11 giờ trước deadline)

- Nộp sớm hơn deadline vài giờ để tránh lỗi kỹ thuật (Moodle treo, sai định dạng file)
- Double-check Git repo public/accessible được cho giảng viên
- Xác nhận đã nộp thành công

⚠️ Lưu ý từ slide 01-Introduction: **"Each team member must be able to explain every part of the code"** — dù chia việc để chạy nhanh trong 4 ngày này, cả 2 người vẫn phải đọc và hiểu phần việc của người kia trước khi nộp, vì điểm participation tính riêng từng cá nhân (0.3 tổng điểm).

---

## 2. Checklist việc cần làm ngay tuần này

- [ ] Nhắn hỏi giảng viên 4 câu ở mục 0
- [ ] Tạo Git repo (theo đúng process trong slide 02: `git init`, venv, requirements.txt)
- [ ] Setup Colab/Kaggle notebook — đây là công cụ chính thức môn dùng, không cần máy có GPU riêng
- [ ] Chạy thử `yolov5s` lấy vài box mẫu thật
- [ ] Viết CPU baseline NMS bằng NumPy thuần theo đúng template trong Project Description.ipynb (hàm `load_data`, `run_cpu`, `verify`, `benchmark`)
- [ ] Chạy `cProfile`, lưu output — bắt buộc để viết "Background/Challenge" có số liệu thật
- [ ] Điền file `Proposal_template.docx` theo nội dung mục 1 ở trên
- [ ] Viết `tests/test_correctness.py` so với `torchvision.ops.nms`
- [ ] Phân công cụ thể theo bảng Weekly schedule

---

## 3. Lưu ý kỹ thuật riêng cho A4 (từ slide 03-Numba)

- Môn dùng **Numba**, không phải CUDA C/C++ hay CuPy thuần cho phần viết kernel. Cụ thể:
  - `@cuda.jit`: dùng cho kernel GPU thật (GPU V1, V2, V3) — cho phép thread cooperate với nhau (cần cho shared memory ở V2).
  - `@jit` / `@jit(parallel=True)`: có thể dùng cho CPU baseline nếu muốn tăng tốc CPU-only trước khi lên GPU, nhưng **không tính là "CPU baseline thuần"** nếu bài yêu cầu pure NumPy — cần hỏi rõ giảng viên có được dùng Numba CPU cho baseline hay không.
- Numba cần dữ liệu dạng NumPy array + loop tường minh — rất hợp với việc viết IoU kernel (loop qua từng cặp box).