# Vấn đáp đồ án CUDA NMS với Numba

Tài liệu này dùng để ôn và trả lời vấn đáp cho đề tài A4 **Real-Time Non-Maximum Suppression for Object Detection**. Nội dung được đối chiếu với đề bài, proposal, mã nguồn hiện tại, test và evidence đã lưu trong repository.

Mỗi câu có hai mức:

- **Trả lời ngắn:** câu nên nói trước, thường trong 15-30 giây.
- **Nếu bị hỏi sâu:** phần giải thích bổ sung, chỉ nói khi giảng viên hỏi tiếp.

## 1. Những điều phải nói đúng

1. CPU, V1 và V2 là **class-aware greedy hard NMS**.
2. V1 song song hóa việc tính IoU nhưng vẫn giải quyết keep/suppress trên CPU.
3. V2 tạo suppression mask trên GPU, nén bằng `uint64`, nhưng greedy resolver vẫn ở CPU.
4. V3 là **Matrix NMS soft-decay**, có ngữ nghĩa khác hard NMS. Không được nói V3 cho cùng danh sách box với CPU/V1/V2.
5. Benchmark đã lưu là **NMS-only trên dữ liệu synthetic**, không phải thời gian inference end-to-end của YOLO.
6. Evidence RTX 4060 Ti thuộc source commit `7ee76cd5f6e12b87ddee247d58c9fd6ac866245b`, không phải phép đo trực tiếp của HEAD hiện tại.
7. Mục tiêu chính thức `< 5 ms` là cho **cả batch 32 ảnh, mỗi ảnh 10.000 boxes**. Kết quả đã lưu là `947.180 ms/batch`, nên mục tiêu này **không đạt**.

## 2. Bài nói mở đầu 90 giây

> Nhóm em thực hiện đề tài tăng tốc Non-Maximum Suppression bằng CUDA Numba. NMS là bước hậu xử lý của object detector, dùng để loại các bounding box trùng lặp và giữ lại box có độ tin cậy cao hơn.
>
> Nhóm xây dựng ba phiên bản có cùng contract đầu vào. CPU baseline chạy greedy hard NMS tuần tự theo từng class. GPU V1 giao một CUDA thread cho mỗi cặp box để tạo ma trận IoU đầy đủ, sau đó copy ma trận về CPU và thực hiện greedy suppression. GPU V2 đổi dữ liệu sang Structure of Arrays, dùng shared memory và nén quan hệ suppress bằng bitmask 64-bit. V2 giảm đáng kể dữ liệu trung gian nhưng quyết định greedy cuối vẫn nằm trên CPU.
>
> Nhóm kiểm tra CPU, V1 và V2 bằng `torchvision.ops.nms` theo từng class. Evidence trên RTX 4060 Ti cho thấy tại 10.000 candidates, CPU mất 308.965 ms, V1 mất 113.991 ms và V2 mất 33.327 ms, tương ứng V2 nhanh hơn CPU 9.27 lần. Tuy nhiên batch 32 mất 947.180 ms nên chưa đạt mục tiêu dưới 5 ms cho cả batch.
>
> V3 là Matrix NMS, giảm score thay vì xóa box trực tiếp. Vì khác ngữ nghĩa hard NMS nên nhóm tách V3 khỏi bảng parity CPU/V1/V2. Hạn chế chính hiện tại là transfer host-device và greedy resolver chưa được loại khỏi CPU.

## 3. Bài toán và kiến thức NMS

### Câu 1. NMS là gì?

**Trả lời ngắn:** NMS là bước hậu xử lý của object detection. Nó giữ box có score cao và loại các box cùng class bị chồng lấp quá nhiều với box đã giữ.

**Nếu bị hỏi sâu:** Với hard NMS, các box được xét theo score giảm dần. Mỗi lần chọn box tốt nhất chưa bị suppress, sau đó suppress các box có IoU với nó lớn hơn ngưỡng. Output của project là chỉ số của các candidate được giữ.

### Câu 2. Tại sao detector cần NMS?

**Trả lời ngắn:** Detector thường dự đoán nhiều box gần giống nhau cho cùng một vật thể. Nếu không có NMS, output sẽ chứa nhiều detection trùng lặp.

**Nếu bị hỏi sâu:** NMS không thay thế model inference. Nó chỉ làm sạch tập candidate sau khi model đã sinh box, score và class.

### Câu 3. Input contract của project là gì?

**Trả lời ngắn:** Một ảnh có `boxes float32 (N,4)` theo định dạng `xyxy`, `scores float32 (N,)` và `class_ids int32 (N,)`.

**Nếu bị hỏi sâu:** Code kiểm tra `x2 > x1`, `y2 > y1`, mọi tọa độ và score phải hữu hạn. Các mảng được chuyển thành contiguous để truy cập ổn định và thuận lợi khi copy sang GPU.

### Câu 4. Công thức IoU là gì?

Với hai box (A) và (B):

$$
IoU(A,B)=\frac{|A\cap B|}{|A\cup B|}
=\frac{S_{intersection}}{S_A+S_B-S_{intersection}}
$$

Trong đó:

$$
w_{intersection}=\max(0,\min(x_{2A},x_{2B})-\max(x_{1A},x_{1B}))
$$

$$
h_{intersection}=\max(0,\min(y_{2A},y_{2B})-\max(y_{1A},y_{1B}))
$$

$$
S_{intersection}=w_{intersection}\times h_{intersection}
$$

**Trả lời ngắn:** IoU bằng diện tích giao chia diện tích hợp. IoU bằng 0 khi không chồng lấp và bằng 1 khi hai box trùng nhau.

### Câu 5. Project dùng điều kiện suppress nào?

**Trả lời ngắn:** Box điểm thấp bị suppress khi cùng class và `IoU > iou_threshold`.

**Nếu bị hỏi sâu:** Code dùng dấu `>` chứ không dùng `>=`. Ngưỡng phải hữu hạn và nằm trong đoạn `[0,1]`.

### Câu 6. Vì sao NMS phải chạy theo từng class?

**Trả lời ngắn:** Hai box chồng nhau nhưng biểu diễn hai class khác nhau vẫn có thể cùng đúng, ví dụ người đang đi xe đạp. Vì vậy project chỉ cho box cùng class suppress nhau.

**Nếu bị hỏi sâu:** Code chia candidate thành từng class, chạy NMS riêng, sau đó ghép chỉ số lại theo score toàn cục. Đây là class-aware NMS.

### Câu 7. Vì sao greedy hard NMS có phụ thuộc tuần tự?

**Trả lời ngắn:** Box sau có bị suppress hay không phụ thuộc vào việc box điểm cao hơn trước đó có được giữ hay đã bị suppress. Vì vậy quyết định tại bước sau phụ thuộc trạng thái của bước trước.

**Nếu bị hỏi sâu:** Các phép IoU pairwise độc lập, nhưng chuỗi quyết định keep/suppress không hoàn toàn độc lập. Đây là lý do V1 và V2 chưa thể đưa toàn bộ hard NMS lên GPU chỉ bằng cách tính mọi IoU song song.

### Câu 8. Độ phức tạp của greedy NMS là bao nhiêu?

**Trả lời ngắn:** Sort tốn (O(N\log N)); phần so sánh IoU trong trường hợp xấu tốn (O(N^2)), nên toàn bộ bị chi phối bởi (O(N^2)).

**Nếu bị hỏi sâu:** GPU không làm mất số lượng cặp cần xét. Nó chia công việc (O(N^2)) cho nhiều thread để giảm thời gian thực tế.

### Câu 9. Stable ordering có tác dụng gì?

**Trả lời ngắn:** Nó làm kết quả tất định khi nhiều box có cùng score.

**Nếu bị hỏi sâu:** Project sắp theo score giảm dần, sau đó dùng original input index để phá hòa. Nếu không thống nhất tie-break, hai thuật toán có thể có IoU giống nhau nhưng trả danh sách keep khác nhau.

### Câu 10. NMS có làm thay đổi độ chính xác của detector không?

**Trả lời ngắn:** Có thể. Ngưỡng IoU quá thấp sẽ xóa nhầm vật thể gần nhau; quá cao sẽ giữ nhiều detection trùng. Vì vậy đây là trade-off của hậu xử lý.

**Nếu bị hỏi sâu:** Benchmark của project chủ yếu kiểm tra correctness và latency NMS. Project chưa cung cấp đánh giá mAP end-to-end trên COCO, nên không được tuyên bố chất lượng detector tăng hoặc không đổi trên toàn bộ dataset.

## 4. CPU baseline

### Câu 11. CPU baseline hoạt động thế nào?

**Trả lời ngắn:** CPU chia box theo class, sắp theo score, giữ box tốt nhất chưa bị suppress, tính IoU của nó với các box còn lại rồi đánh dấu những box vượt ngưỡng.

**Nếu bị hỏi sâu:** Hàm `iou_one_to_many()` dùng NumPy để tính IoU từ một box đến nhiều box. Vòng greedy vẫn chạy theo rank và duy trì mảng Boolean `suppressed`.

### Câu 12. Tại sao cần CPU baseline?

**Trả lời ngắn:** CPU baseline vừa là mốc tốc độ, vừa là implementation dễ hiểu để đối chiếu correctness cho GPU V1 và V2.

**Nếu bị hỏi sâu:** Project còn dùng `torchvision.ops.nms` làm oracle độc lập. CPU baseline không nên là oracle duy nhất vì CPU và GPU có thể cùng lặp lại một lỗi thiết kế.

### Câu 13. CPU baseline có hoàn toàn là Python loop không?

**Trả lời ngắn:** Không. Quyết định greedy là loop tuần tự, nhưng phép IoU one-to-many được vector hóa bằng NumPy.

**Nếu bị hỏi sâu:** Vectorization giúp phần số học chạy trong code native nhanh hơn loop Python, nhưng dependency giữa các lần chọn box vẫn còn.

### Câu 14. CPU dùng bao nhiêu bộ nhớ phụ?

**Trả lời ngắn:** CPU baseline không lưu toàn bộ ma trận (N\times N). Nó chủ yếu giữ mảng suppressed và vector IoU cho các box còn lại, nên bộ nhớ phụ tuyến tính theo (N).

### Câu 15. Profile ban đầu cho thấy gì?

**Trả lời ngắn:** Proposal ghi nhận tại (N=10.000), phần suppression loop và `iou_one_to_many` chiếm hơn 99% thời gian đo, nên NMS là phần được chọn để tăng tốc.

**Nếu bị hỏi sâu:** Đây là profile của NMS synthetic trong proposal, không phải bằng chứng rằng NMS luôn là bottleneck lớn nhất của toàn bộ YOLO inference trên mọi thiết bị.

## 5. Kiến thức CUDA cần nói được

### Câu 16. Host và device là gì?

**Trả lời ngắn:** Host là CPU và RAM; device là GPU và VRAM. Dữ liệu phải được copy giữa hai miền bộ nhớ trước và sau khi chạy kernel.

### Câu 17. Kernel CUDA khác hàm Python bình thường thế nào?

**Trả lời ngắn:** Kernel được hàng loạt GPU thread chạy song song. Mỗi thread xác định phần dữ liệu của mình bằng index và thường ghi kết quả vào mảng output; kernel CUDA không trả một object Python bằng `return` như hàm thông thường.

**Nếu bị hỏi sâu:** `return` bên trong kernel của project chỉ dùng để kết thúc sớm một thread, ví dụ bounds guard, không trả dữ liệu về host.

### Câu 18. Thread, block, grid và warp là gì?

**Trả lời ngắn:** Thread là đơn vị thực thi nhỏ nhất; block là nhóm thread có thể dùng shared memory và đồng bộ; grid là toàn bộ các block của một kernel launch; warp là nhóm 32 thread được GPU NVIDIA lập lịch cùng nhau.

### Câu 19. Vì sao phải có bounds guard?

**Trả lời ngắn:** Số block được làm tròn lên để phủ đủ (N), nên có thể sinh thread có index vượt kích thước dữ liệu. Bounds guard ngăn truy cập ngoài mảng.

### Câu 20. Tăng số thread có luôn nhanh hơn không?

**Trả lời ngắn:** Không. Quá ít thread không tận dụng GPU, nhưng quá nhiều hoặc block size không phù hợp có thể tăng overhead, dùng quá nhiều register/shared memory, giảm occupancy hoặc tạo công việc dư.

**Nếu bị hỏi sâu:** Phải benchmark block size và xem cả workload. Với input nhỏ, launch và transfer overhead có thể lớn hơn lợi ích song song.

### Câu 21. Vì sao cần warm-up trước benchmark GPU?

**Trả lời ngắn:** Numba biên dịch kernel theo cơ chế JIT ở lần gọi đầu. Warm-up tách chi phí biên dịch khỏi thời gian chạy ổn định cần đo.

### Câu 22. Khi nào cần `cuda.synchronize()`?

**Trả lời ngắn:** Kernel launch bất đồng bộ so với host. Cần synchronize trước khi kết thúc timer hoặc trước khi dùng kết quả phụ thuộc để tránh đo thiếu thời gian GPU.

## 6. GPU V1

### Câu 23. V1 song song hóa phần nào?

**Trả lời ngắn:** V1 song song hóa ma trận IoU. Mỗi thread xử lý một cặp `(row, column)` và ghi một phần tử của ma trận (N\times N).

### Câu 24. Cấu hình launch của V1 là gì?

**Trả lời ngắn:** V1 dùng block hai chiều `(16,16)`, tức 256 thread. Grid có kích thước `ceil(N/16) × ceil(N/16)`.

### Câu 25. Luồng dữ liệu của V1 là gì?

**Trả lời ngắn:** Chia theo class và sort trên host, copy box sang GPU, tạo full IoU matrix trên GPU, copy matrix về host, rồi CPU chạy greedy resolver.

### Câu 26. Hạn chế lớn nhất của V1 là gì?

**Trả lời ngắn:** Ma trận IoU tốn bộ nhớ và băng thông (O(N^2)), đồng thời phải copy toàn bộ về host.

**Nếu bị hỏi sâu:** Với một partition có 10.000 boxes, ma trận có 100 triệu phần tử `float32`, khoảng 400 MB thập phân hay 381.5 MiB, chưa tính input và overhead.

### Câu 27. Vì sao V1 vẫn có giá trị dù chưa tối ưu?

**Trả lời ngắn:** V1 là bước chuyển trực tiếp và dễ kiểm chứng từ CPU sang GPU. Nó chứng minh phần pairwise IoU có thể song song hóa và tạo mốc để đánh giá tối ưu của V2.

### Câu 28. V1 có phải GPU NMS hoàn toàn không?

**Trả lời ngắn:** Không. Chỉ ma trận IoU chạy trên GPU; quyết định greedy vẫn chạy trên CPU.

## 7. GPU V2

### Câu 29. V2 cải tiến gì so với V1?

**Trả lời ngắn:** V2 dùng SoA để đọc tọa độ liền mạch hơn, dùng shared memory để cache target boxes và lưu quan hệ suppress bằng packed `uint64` thay vì full float IoU matrix.

### Câu 30. SoA khác AoS thế nào?

**Trả lời ngắn:** AoS lưu mỗi box thành `[x1,y1,x2,y2]`; SoA tách thành bốn mảng `x1[]`, `y1[]`, `x2[]`, `y2[]`. Khi các thread gần nhau đọc cùng một tọa độ của nhiều box, SoA giúp truy cập global memory dễ coalesce hơn.

### Câu 31. Bitmask 64-bit hoạt động thế nào?

**Trả lời ngắn:** Một `uint64` biểu diễn quan hệ suppress với tối đa 64 target boxes. Bit thứ (k) bằng 1 nghĩa là anchor box suppress target box tương ứng.

**Nếu bị hỏi sâu:** Kernel chỉ xét `candidate > anchor` vì box đã được sort theo score. Mỗi anchor không cần lưu quan hệ với chính nó hoặc box có rank cao hơn.

### Câu 32. V2 giảm bộ nhớ bao nhiêu?

**Trả lời ngắn:** Full matrix dùng một `float32` 32 bit cho mỗi cặp; packed mask dùng khoảng một bit cho mỗi cặp, nên lý thuyết giảm xấp xỉ 32 lần, bỏ qua padding và metadata.

**Nếu bị hỏi sâu:** Với (N=10.000), `words=ceil(10000/64)=157`. Mask một ảnh có `157 × 10000` phần tử `uint64`, khoảng 12.56 MB thập phân hay 11.98 MiB, so với khoảng 400 MB của full float matrix.

### Câu 33. Grid và block của bitmask kernel là gì?

**Trả lời ngắn:** Một block có 64 thread. Grid ba chiều là `(words, words, batch_size)`, trong đó hai chiều đầu mô tả các word của anchor và target, chiều thứ ba là ảnh trong batch.

### Câu 34. Shared memory được dùng thế nào trong V2?

**Trả lời ngắn:** 64 thread cùng nạp tối đa 64 target boxes vào bốn mảng tọa độ trong shared memory. Sau `cuda.syncthreads()`, mỗi anchor thread tái sử dụng tile đó để tính IoU với các target.

### Câu 35. V2 có dùng parallel reduction không?

**Trả lời ngắn:** Implementation V2 hiện tại không dùng tree/warp reduction. Nó xây suppression mask song song rồi CPU OR các word mask trong vòng greedy.

**Nếu bị hỏi sâu:** Đề catalog mô tả V2 với parallel reduction, nhưng code hiện tại thực hiện mask packing. Không nên gọi thao tác pack bit hoặc OR tuần tự trên host là parallel reduction.

### Câu 36. V2 có hoàn toàn chạy trên GPU không?

**Trả lời ngắn:** Không. Sort, class partition và greedy mask resolution còn ở host. GPU chịu trách nhiệm xây quan hệ suppress.

### Câu 37. V2 xử lý batch 32 thế nào?

**Trả lời ngắn:** Nếu toàn batch chỉ có một class, V2 sort từng ảnh rồi dùng một kernel launch có chiều grid `z=batch_size`. Nếu multi-class, code hiện tại partition từng ảnh theo class và gọi lại đường xử lý tương ứng.

**Nếu bị hỏi sâu:** Vì vậy đường multi-class không giữ được một fused launch duy nhất cho toàn bộ batch. Đây là một lý do cần cẩn thận khi diễn giải throughput thực tế.

### Câu 38. Resolver mask trên CPU làm gì?

**Trả lời ngắn:** CPU duyệt anchor theo score. Nếu bit của anchor chưa bị đánh dấu thì giữ anchor, sau đó OR mask của nó vào tập suppressed cho các word còn lại.

## 8. GPU V3 Matrix NMS

### Câu 39. V3 khác V1/V2 ở điểm cốt lõi nào?

**Trả lời ngắn:** V1/V2 là greedy hard NMS và xóa box; V3 là Matrix NMS, giảm score bằng decay factor rồi lọc theo score threshold.

### Câu 40. Vì sao V3 dễ song song hơn hard NMS?

**Trả lời ngắn:** Decay của mỗi box có thể được tính từ quan hệ với các box score cao hơn mà không cần cập nhật chuỗi trạng thái keep/suppress tuần tự như greedy hard NMS.

### Câu 41. V3 dùng cấu hình kernel nào?

**Trả lời ngắn:** V3 dùng một block 256 thread cho mỗi box. Các thread trong block dùng grid-stride loop để xét các box điểm cao hơn rồi tree-reduction trong shared memory.

### Câu 42. Hai kernel chính của V3 làm gì?

**Trả lời ngắn:** Kernel thứ nhất tìm `iou_max` bằng parallel max reduction. Kernel thứ hai tìm decay nhỏ nhất bằng parallel min reduction và nhân decay vào score.

### Câu 43. Công thức decay trong code là gì?

**Trả lời ngắn:** V3 tìm hệ số decay nhỏ nhất do các box điểm cao hơn gây ra, rồi nhân hệ số đó với score ban đầu. Code hỗ trợ hai cách giảm điểm là linear và gaussian.

Với box (j), code xét box điểm cao hơn (i). Khi (IoU(i,j)>iou\_max[i]):

Linear:

$$
decay=\frac{1-IoU(i,j)}{\max(1-iou\_max[i],10^{-9})}
$$

Gaussian:

$$
decay=\exp\left(\frac{iou\_max[i]^2-IoU(i,j)^2}{\sigma}\right)
$$

Score cuối:

$$
score'_j=score_j\times\min_i(decay_{ij})
$$

### Câu 44. Có được so V3 với `torchvision.ops.nms` bằng exact keep indices không?

**Trả lời ngắn:** Không. `torchvision.ops.nms` là hard NMS, còn V3 là soft-decay. V3 phải so với `matrix_nms_reference()` của cùng công thức và kiểm tra cả decayed scores.

### Câu 45. V3 hiện có class-aware không?

**Trả lời ngắn:** Đường V3 hiện tại chỉ nhận boxes và scores, không nhận `class_ids`. Vì vậy không được tuyên bố V3 có cùng class-aware contract với CPU/V1/V2.

### Câu 46. Có được dùng số “hơn 50×” ghi trong comment V3 không?

**Trả lời ngắn:** Không, trừ khi có evidence benchmark gắn với đúng commit, môi trường và protocol. Comment hoặc kỳ vọng trong code không phải bằng chứng thực nghiệm.

## 9. Kiểm thử correctness

### Câu 47. Project kiểm tra CPU/V1/V2 đúng bằng cách nào?

**Trả lời ngắn:** Chạy `torchvision.ops.nms` riêng cho từng class làm oracle, sau đó so ordered keep indices. IoU matrix còn được kiểm tra đường chéo, tính đối xứng và sai số với CPU.

### Câu 48. Tại sao cần tolerance `1e-4`?

**Trả lời ngắn:** CPU và GPU có thể khác thứ tự phép toán và cách làm tròn float32, nên giá trị IoU có thể lệch vài đơn vị rất nhỏ dù logic đúng.

**Nếu bị hỏi sâu:** Tolerance áp dụng cho giá trị số thực. Với output hard NMS đã thống nhất stable ordering, project kiểm tra exact ordered indices.

### Câu 49. Những edge case nào đã được test?

**Trả lời ngắn:** Box trùng nhau, box không chồng lấp, khác class, input rỗng, score ties, nhiều threshold, partial 64-box block, batch size 3 và 32, cùng parity ở nhiều kích thước tới 10.000.

### Câu 50. Tại sao test box khác class quan trọng?

**Trả lời ngắn:** Nó bắt lỗi implementation class-agnostic. Hai box giống hệt nhau nhưng khác class phải cùng được giữ.

### Câu 51. `81 passed, 1 skipped` chứng minh điều gì?

**Trả lời ngắn:** Nó chứng minh test suite tại evidence commit đã chạy trên môi trường CUDA ghi nhận và chỉ skip test YOLO do checkpoint không khả dụng tại lần chạy đó.

**Nếu bị hỏi sâu:** Nó không tự động chứng minh mọi commit về sau vẫn đúng, không chứng minh performance target đạt, và không thay thế việc đọc phạm vi từng test.

## 10. Benchmark và cách đọc kết quả

### Câu 52. Protocol benchmark đã lưu là gì?

**Trả lời ngắn:** Synthetic deterministic candidates, seed 0, hai warm-up, bảy lần đo, báo median; chạy trên RTX 4060 Ti với Python 3.11.9, NumPy 1.26.4 và Numba 0.67.0.

### Câu 53. Kết quả single-image là bao nhiêu?

**Trả lời ngắn:** Ở 10.000 candidates, CPU mất 308.965 ms, V1 mất 113.991 ms và V2 mất 33.327 ms; V2 nhanh hơn CPU 9.27 lần. Ở 100 candidates, GPU lại chậm hơn CPU do overhead.

| Candidates | CPU | V1 | V2 | V1 speedup | V2 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.378 ms | 2.467 ms | 5.065 ms | 0.56× | 0.27× |
| 1.000 | 16.259 ms | 4.501 ms | 7.625 ms | 3.61× | 2.13× |
| 10.000 | 308.965 ms | 113.991 ms | 33.327 ms | 2.71× | 9.27× |

**Cách nói:** GPU không luôn nhanh hơn CPU. Ở (N=100), overhead khiến cả V1 và V2 chậm hơn CPU. Khi (N) lớn, lượng pairwise work đủ lớn để GPU phát huy song song.

### Câu 54. Vì sao V2 chậm hơn V1 ở N nhỏ và N=1.000?

**Trả lời ngắn:** V2 có thêm sort, chuyển SoA, cấp phát mask, shared-memory tiling và host mask resolution. Khi workload chưa đủ lớn, lợi ích giảm transfer chưa bù được overhead đó.

**Nếu bị hỏi sâu:** Đây là giải thích hợp lý từ kiến trúc và số đo, không phải kernel profile chi tiết. Muốn kết luận chính xác tỷ lệ từng nguyên nhân cần Nsight hoặc profiling theo phase.

### Câu 55. Vì sao V2 thắng rõ ở N=10.000?

**Trả lời ngắn:** Ở kích thước lớn, full float matrix và transfer của V1 trở nên đắt. V2 chỉ truyền packed mask nhỏ hơn nhiều nên đạt 33.327 ms so với 113.991 ms của V1.

### Câu 56. Kết quả batch 32 là gì?

**Trả lời ngắn:** Batch 32, mỗi ảnh 10.000 candidates, V2 có median `947.180 ms/batch`, tương đương `29.599 ms/image` để tham khảo.

### Câu 57. Project có đạt mục tiêu dưới 5 ms không?

**Trả lời ngắn:** Không. Đề yêu cầu dưới 5 ms cho cả batch, trong khi kết quả là 947.180 ms. Không được lấy 29.599 ms/image rồi so với mục tiêu batch, và con số per-image cũng vẫn lớn hơn 5 ms.

### Câu 58. Project có đạt mục tiêu speedup 15× hoặc 30-80× không?

**Trả lời ngắn:** Không theo evidence chính CPU/V1/V2. Speedup cao nhất được báo ở (N=10.000) là V2 `9.27×`. V3 không có evidence tương đương trong gói kết quả chính để dùng chứng minh mục tiêu stretch.

### Câu 59. Benchmark đo những phase nào?

**Trả lời ngắn:** Benchmark V2 batch đo end-to-end của lời gọi NMS: host sort, transfer, GPU mask kernel, copy mask về và host greedy resolution. Nó vẫn loại model inference và image preprocessing.

### Câu 60. Tại sao dùng median thay vì chỉ đo một lần?

**Trả lời ngắn:** Runtime có nhiễu do scheduling, cache và hệ thống. Nhiều sample sau warm-up và lấy median giảm ảnh hưởng của outlier hơn một lần đo đơn lẻ.

### Câu 61. Có được so benchmark giữa T4 và RTX 4060 Ti trực tiếp không?

**Trả lời ngắn:** Chỉ dùng để tham khảo, không dùng để kết luận phiên bản code nào nhanh hơn nếu commit, dependency và protocol khác nhau. So sánh công bằng phải giữ cùng code, input, warm-up, repeat và timing scope.

### Câu 62. Tại sao không gọi đây là real-time object detection hoàn chỉnh?

**Trả lời ngắn:** Số đo chính chỉ bao gồm NMS trên synthetic candidates. Nó không gồm model loading, preprocessing, YOLO forward và postprocessing đầy đủ.

## 11. Câu hỏi xoáy và cách trả lời an toàn

### Câu 63. Đề yêu cầu V2 parallel reduction nhưng code không có, giải thích sao?

**Trả lời ngắn:** Nhóm hoàn thành packed suppression mask và coalesced SoA, nhưng chưa hiện thực tree/warp reduction cho greedy resolution. Đây là phần chưa đạt đúng toàn bộ mô tả V2 của catalog và là hướng cải tiến tiếp theo.

### Câu 64. Vì sao mục tiêu thấp hơn đề quá nhiều?

**Trả lời ngắn:** Pipeline hiện chưa device-resident. Sort, partition, transfer và greedy resolver còn trên CPU; với batch 32 còn tạo lượng mask và công việc rất lớn. Nhóm báo đúng số đo và không đổi đơn vị để làm kết quả đẹp hơn.

### Câu 65. Bottleneck hiện tại là gì?

**Trả lời ngắn:** V1 bị full matrix và transfer; V2 giảm transfer nhưng vẫn còn host orchestration, mask download và greedy CPU resolver. Với batch multi-class còn có thêm partition và nhiều đường xử lý.

**Nếu bị hỏi sâu:** Chưa có phase profile đầy đủ trên evidence RTX 4060 Ti, nên chỉ nên gọi đây là bottleneck theo thiết kế. Cần đo riêng sort, H2D, kernel, D2H và resolver để định lượng.

### Câu 66. Làm thế nào cải tiến V2?

**Trả lời ngắn:** Tái sử dụng device buffers, giảm cấp phát, giữ dữ liệu trên GPU, gom class partitions tốt hơn, overlap transfer bằng stream và nghiên cứu resolver song song hoặc thuật toán khác ít dependency hơn.

### Câu 67. Tại sao không dùng luôn `torchvision.ops.nms`?

**Trả lời ngắn:** Mục tiêu môn học là tự cài đặt và phân tích thuật toán CUDA bằng Numba. `torchvision` chỉ làm oracle correctness, không thay kernel của project.

### Câu 68. Tại sao dùng Numba thay vì CUDA C++?

**Trả lời ngắn:** Numba là công cụ GPU chính của môn, cho phép viết kernel CUDA bằng Python và vẫn thể hiện rõ grid, block, thread, shared memory và host-device transfer.

### Câu 69. Synthetic data có đủ thuyết phục không?

**Trả lời ngắn:** Synthetic data phù hợp để kiểm soát (N), seed và đo scaling NMS. Nhưng nó không thay thế đánh giá phân phối box thật hoặc pipeline detector end-to-end.

### Câu 70. Project có detector integration không?

**Trả lời ngắn:** Repo có adapter đọc raw YOLOv5 candidates trước NMS và benchmark tách candidate extraction với NMS. Tuy nhiên evidence chính Seminar 3 không tuyên bố detector inference vì môi trường chạy evidence không có checkpoint phù hợp.

### Câu 71. Nếu GPU output khác CPU thì xử lý thế nào?

**Trả lời ngắn:** Trước hết phân biệt sai số float với sai khác logic. Kiểm tra input contract, stable ordering, class partition, IoU tolerance và exact keep indices; không đánh giá speedup cho một implementation chưa qua correctness.

### Câu 72. Nếu có nhiều box cùng score thì box nào được xét trước?

**Trả lời ngắn:** Box có original input index nhỏ hơn được ưu tiên. Project dùng stable deterministic ordering để CPU/V1/V2 thống nhất.

### Câu 73. Nếu threshold bằng 1 thì sao?

**Trả lời ngắn:** Vì điều kiện suppress là `IoU > threshold`, IoU không vượt quá 1 nên không box nào bị suppress bởi ngưỡng 1.

### Câu 74. Nếu threshold bằng 0 thì sao?

**Trả lời ngắn:** Mọi box cùng class có phần giao dương với box đã giữ sẽ bị suppress, vì IoU dương sẽ lớn hơn 0. Box không chồng lấp vẫn được giữ.

### Câu 75. Tại sao chỉ tính tam giác trên của quan hệ suppress?

**Trả lời ngắn:** Sau khi sort theo score, chỉ box rank cao hơn mới có quyền suppress box rank thấp hơn. Tính chiều ngược lại là dư thừa cho greedy hard NMS.

### Câu 76. V1 full matrix có tính dữ liệu dư không?

**Trả lời ngắn:** Có. Kernel V1 tính cả hai tam giác và đường chéo dù ma trận IoU đối xứng. Thiết kế này cố ý đơn giản nhưng tốn compute và memory.

### Câu 77. Vì sao bitmask không tự giải quyết dependency greedy?

**Trả lời ngắn:** Bitmask chỉ nén quan hệ “nếu anchor được giữ thì nó suppress ai”. Ta vẫn phải biết anchor có thực sự được giữ hay đã bị box trước suppress, nên resolver vẫn phải đi theo thứ tự score.

### Câu 78. Có thể gọi V3 là bản tối ưu trực tiếp của V2 không?

**Trả lời ngắn:** Không hoàn toàn. V3 thay đổi thuật toán từ hard suppression sang soft score decay. Nó là một trade-off thuật toán để tăng khả năng song song, không chỉ là tối ưu implementation giữ nguyên output.

## 12. Phân công và trách nhiệm nhóm

### Câu 79. Ai làm phần nào?

**Trả lời an toàn:** Phải trả lời theo bản phân công cuối mà cả hai thành viên thống nhất và có thể đối chiếu với Git history. `TEAM_PLAN.md` hiện ghi Phùng Quốc Tuấn là primary implementation owner; Lê Quang Tân phụ trách review báo cáo, evidence, presentation và giải thích code chung.

**Cảnh báo:** Proposal cũ lại ghi Lê Quang Tân triển khai GPU V1 và các GPU test. Hai tài liệu đang không thống nhất. Nhóm phải thống nhất cách mô tả trung thực trước ngày vấn đáp; không học thuộc hai câu trả lời mâu thuẫn.

### Câu 80. Nếu giảng viên hỏi phần không trực tiếp code thì trả lời sao?

**Trả lời ngắn:** Theo yêu cầu môn, cả hai thành viên phải hiểu toàn bộ pipeline. Trả lời từ data contract, data flow, kernel mapping, correctness và limitation; không đổ rằng “phần này bạn kia làm”.

### Câu 81. Đóng góp kỹ thuật quan trọng nhất của project là gì?

**Trả lời ngắn:** Xây dựng contract class-aware thống nhất, một CPU oracle rõ ràng, V1 pairwise IoU, V2 SoA packed-mask, test parity và benchmark có provenance. Điểm quan trọng là project báo trung thực giới hạn và mục tiêu chưa đạt.

### Câu 82. Kết luận một câu về đồ án?

**Trả lời ngắn:** Project chứng minh GPU có lợi khi số candidate lớn và V2 giảm mạnh chi phí trung gian so với V1, nhưng chưa đạt target vì hard-NMS dependency và host-device overhead vẫn còn.

## 13. Những câu tuyệt đối không nên nói

- Không nói: “V2 hoàn toàn chạy trên GPU.”
- Không nói: “Bitmask chính là parallel reduction.”
- Không nói: “V3 giống hệt hard NMS nhưng nhanh hơn.”
- Không nói: “Project đạt mục tiêu dưới 5 ms.”
- Không nói: “29.599 ms/image tương đương đạt mục tiêu 5 ms/batch.”
- Không nói: “GPU luôn nhanh hơn CPU.”
- Không nói: “Đã benchmark toàn bộ YOLO end-to-end” khi evidence chỉ là NMS synthetic.
- Không dùng số performance trong comment, proposal hoặc evidence commit khác như kết quả của HEAD hiện tại.
- Không khẳng định V3 class-aware khi hàm hiện tại không nhận `class_ids`.
- Không tự bịa phase profiling, occupancy hoặc memory bandwidth nếu chưa có log đo tương ứng.

## 14. Checklist học trước khi vào vấn đáp

### Phải nói không nhìn tài liệu

- [ ] Giải thích NMS và công thức IoU.
- [ ] Nói đúng input contract và class-aware semantics.
- [ ] Mô tả luồng CPU baseline.
- [ ] Vẽ được data flow V1 và V2.
- [ ] Giải thích thread/block/grid của V1 và V2.
- [ ] Giải thích SoA, coalescing, shared memory và bitmask.
- [ ] Giải thích vì sao greedy resolver còn ở CPU.
- [ ] Phân biệt hard NMS với Matrix NMS.
- [ ] Đọc đúng bảng benchmark và đơn vị ms.
- [ ] Nói thẳng target `<5 ms/batch` chưa đạt.
- [ ] Phân biệt evidence đã chạy với kỳ vọng trong proposal.
- [ ] Thống nhất phân công thành viên theo bằng chứng thật.

### Năm số cần nhớ

1. `N = 100, 1.000, 10.000`.
2. Tại `N=10.000`: CPU `308.965 ms`.
3. Tại `N=10.000`: V1 `113.991 ms`.
4. Tại `N=10.000`: V2 `33.327 ms`, speedup `9.27×`.
5. Batch 32 × 10.000: `947.180 ms/batch`, **không đạt `<5 ms/batch`**.

## 15. Nguồn đối chiếu trong repository

- Đề chính thức: `Project Topic Catalog.pdf`, mục A4.
- Proposal: `CSC14116 - Proposal.docx`.
- Danh sách và câu hỏi seminar: `docs/ALPP_22KHMT+KHDL-SeminarList.xlsx`; sheet Nhóm 11 hiện chưa ghi câu hỏi.
- Candidate contract: `src/common/candidates.py`.
- Oracle: `src/common/oracle.py`.
- CPU: `src/baseline/core.py`.
- V1: `src/v1/kernel.py`, `src/v1/core.py`.
- V2: `src/v2/kernels.py`, `src/v2/core.py`.
- V3: `src/gpu_v3.py`.
- Test: `tests/test_correctness.py`.
- Evidence: `submission/seminar_3/evidence/`.
- Kết quả và giới hạn: `submission/seminar_3/README.md`.
- Phân công cuối: `submission/seminar_3/TEAM_PLAN.md`.
