# Kịch bản bảo vệ Project 27 (12 phút)

Tài liệu này dùng cho một người trình bày toàn bộ dự án. Slide và báo cáo lấy số từ run thật. Nếu giảng viên yêu cầu mở code, ưu tiên `fedavg.py`, `server.py`, `partition.py` và `outputs/audit.json`.

| Slide | Thời lượng | Nội dung |
|---|---:|---|
| 1 | 30 giây | Chủ đề, phạm vi và 15 run thật |
| 2 | 40 giây | Câu hỏi label skew, biến giữ cố định |
| 3 | 50 giây | Fashion-MNIST, train/validation/test, chống leakage |
| 4 | 65 giây | Quota, hai class ưu thế, TV |
| 5 | 45 giây | CNN, logits và cross entropy |
| 6 | 65 giây | Broadcast, local update, weighted FedAvg |
| 7 | 50 giây | 5 cấu hình × 3 seed, cùng ngân sách |
| 8 | 70 giây | Final mean ± sample SD |
| 9 | 65 giây | Curves theo round và protocol test |
| 10 | 60 giây | E1R30/E3R10, cùng lượt ảnh |
| 11 | 65 giây | Class recall và confusion pair |
| 12 | 115 giây | Kết luận, hạn chế, demo và buffer |

Tổng **720 giây = 12 phút**. Slide 12 gồm 35 giây demo, 45 giây kết luận và 35 giây buffer chuyển giao diện. Speaker notes trong PPTX hướng dẫn phần giải thích, có thể nói gọn theo lịch.

## Demo 35 giây

1. Mở `streamlit run demo/app.py` trước buổi bảo vệ, tải artifact trước nếu cần.
2. Chọn FedAvg strong và trung bình 3 seed, đọc final accuracy và macro-F1.
3. Chọn seed 42, xem distribution heatmap và confusion matrix.
4. Kéo slider tới một ảnh test, đọc nhãn thật, dự đoán và xác suất.
5. Nói rõ đây là checkpoint đã train trước, demo không chạy train trực tiếp.

## Câu hỏi thường gặp

**Vì sao không dùng Flower?** Tôi tự viết FedAvg để thể hiện rõ snapshot và phép trung bình từng tensor. Không cần framework distributed cho nghiên cứu mô phỏng này.

**Vì sao mỗi client không train nối tiếp trên cùng model?** Mỗi client phải nhận cùng global snapshot. Nếu client sau bắt đầu từ client trước thì thành sequential training, không đúng FedAvg. Test reset snapshot kiểm tra điều này.

**Vì sao strong vẫn có 10 class?** 92% ở hai class ưu thế, 8% còn lại chia cho tám class. Strong có 2.484 mẫu/class ưu thế và 54 mẫu/class khác.

**Vì sao FedAvg có trọng số khi các client bằng nhau?** Công thức chung dùng n_k/N. Thiết kế hiện tại n_k bằng nhau nên ra 0,1, nhưng test dùng số mẫu không bằng nhau để kiểm tra công thức.

**E3 có được train nhiều hơn không?** E3R10 có cùng 30 lượt train pool với E1R30. E tăng và R giảm đồng thời. Tôi đánh giá trade-off compute cố định và số lần tổng hợp, không cô lập tác động riêng của E.

**Vì sao macro recall bằng accuracy?** Test cân bằng 1.000 mẫu/class, nên trung bình recall đều class trùng accuracy. Macro-F1 tính F1 mỗi class rồi trung bình, vẫn có thể khác.

**Chọn checkpoint bằng test có leakage không?** Tôi chọn final checkpoint cho bảng chính, best-validation cho bảng phụ, hòa chọn mốc sớm. Test curves đánh giá sau train để mô tả. Không dùng max test để tune.

**Vì sao FL-IID chưa gần centralized?** FedAvg trung bình các điểm cuối của nhiều optimizer trajectories. Dù cùng ngân sách ảnh, quá trình tối ưu khác centralized trên minibatch trộn toàn pool. So sánh chỉ cho thiết kế này.

**Accuracy giảm có chứng minh gradient conflict không?** Chưa. Client drift là cách giải thích dựa trên lý thuyết và curves. Project chưa đo cosine gradient hay disagreement trực tiếp.

**FL có riêng tư tuyệt đối không?** Chưa. Server API không nhận ảnh train, nhưng model updates có thể rò rỉ thông tin. Chưa có differential privacy hoặc secure aggregation.

**Có train trên 10 máy không?** Đây là 10 client mô phỏng trên một máy. Dataset được chia index, từng client chỉ dùng subset để cập nhật. Không có cách ly bộ nhớ hoặc mạng thực.

**Có tái lập số liệu không?** Có seed, configs, split/partition indices, initial hash, commit, library versions, history và checkpoints. Bitwise reproduction giới hạn cùng môi trường/backend. CPU/CUDA/MPS có thể khác chút số học.

**Có thể kết luận rộng không?** Kết quả chỉ trong Fashion-MNIST, CNN nhỏ, balanced synthetic label skew, full participation và ba seed. Tôi chưa tuyên bố thống kê mạnh.
