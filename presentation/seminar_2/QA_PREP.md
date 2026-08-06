# Q&A chuẩn bị Seminar 2

## Vì sao NMS là O(N²)?

Mỗi box được giữ có thể phải so IoU với các box còn lại. Trường hợp xấu không
có suppress sớm thì số cặp tăng bậc hai.

## Vì sao không suppress box khác class?

Class khác nhau có ý nghĩa vật thể khác nhau. NMS đúng cho detector phải chạy
độc lập từng class; nếu không, person và bicycle chồng nhau có thể xóa nhau.

## Vì sao dùng torchvision?

`torchvision.ops.nms` là oracle ngoài cho hard NMS. Project không gọi nó để
thay kernel CUDA, chỉ dùng để kiểm tra CPU/V1/V2 theo từng class.

## V1 song song hóa phần nào?

Chỉ tính IoU pairwise. Greedy keep/suppress sau đó vẫn serial trên CPU.

## Vì sao V1 chưa tốt?

Full IoU matrix cần `O(N²)` float memory và phải copy về host. Ở N lớn, transfer
và memory trở thành bottleneck.

## SoA của V2 giúp gì?

Thread gần nhau đọc phần tử gần nhau trong từng mảng tọa độ, nên GPU coalesce
memory access tốt hơn AoS.

## Bitmask có phải parallel reduction không?

Không. V2 pack quan hệ suppress vào `uint64`; nó không thực hiện tree/warp
reduction giữa thread.

## V2 đã hoàn toàn parallel chưa?

Chưa. GPU tạo mask song song nhưng resolver greedy phải biết trạng thái rank
cao hơn trước, nên vẫn chạy serial trên CPU.

## Batch 32 nghĩa là gì?

Là 32 ảnh độc lập. Nó khác word 64 box trong bitmask. Batch đa class phải giữ
đúng semantics bằng partition theo class.

## Vì sao V3 không so với torchvision NMS?

V3 là Matrix NMS/soft-decay, giảm score thay vì hard suppress. So output index
với greedy NMS là sai câu hỏi; dùng `matrix_nms_reference()` thay thế.

## Vì sao benchmark synthetic và detector tách nhau?

Synthetic test kiểm tra scaling NMS với N cố định. Detector report đo raw
candidate extraction và NMS riêng, nên không trộn model-inference time vào
NMS-only speedup.

## Có thể công bố số T4 cũ không?

Không như số của code hiện tại. Evidence T4 hiện có trước class-aware rebuild;
chỉ dùng làm lịch sử. Rerun current commit trước khi thay số trên slide.
