# Lời nói gợi ý cho Seminar 2

## Mở đầu

“Nhóm em làm NMS — bước dọn các bounding box trùng lặp sau detector. Catalog
yêu cầu pipeline detector thật, hard NMS đúng, và benchmark theo N=100, 1.000,
10.000.”

## CPU và contract

“Mỗi candidate gồm box, score và class id. NMS phải chạy theo từng class; nếu
hai box cùng vị trí nhưng khác class thì không được xóa lẫn nhau. CPU greedy là
mốc để kiểm tra V1 và V2.”

## V1

“V1 giao một CUDA thread cho một cặp IoU. Nó song song phần pairwise rất rõ,
nhưng vẫn phải copy full matrix N×N về CPU và quyết định greedy còn tuần tự.”

## V2

“V2 đổi layout sang SoA và nén suppress relation bằng bitmask 64-bit. GPU làm
pairwise work hiệu quả hơn, nhưng resolver theo score vẫn ở CPU vì dependency
của greedy NMS chưa biến mất.”

## Detector và benchmark

“Chúng em tách hai phép đo: synthetic NMS để thấy scaling, và raw YOLO
candidate extraction cộng NMS để chứng minh integration. Không lấy số NMS-only
để gọi là inference end-to-end.”

## Evidence gate và giới hạn

“Trên máy local hiện tại, suite có 30 test pass và 41 test CUDA bị skip vì
không có NVIDIA CUDA. Vì vậy nhóm không công bố số GPU cho commit mới trước
khi rerun parity và benchmark. V1 còn copy dense matrix về CPU; V2 giảm phần
đó bằng bitmask nhưng quyết định greedy cuối vẫn ở CPU.”

## Kết

“Seminar này chốt ở Baseline, V1 và V2. Local correctness đã pass; GPU numbers
chỉ được công bố sau CUDA rerun trên đúng commit. Điểm chính là trade-off: V1
dễ hiểu, V2 giảm traffic, và greedy dependency là giới hạn còn lại.”
