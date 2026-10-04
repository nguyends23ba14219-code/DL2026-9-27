# Kịch bản overview Nhóm 9 - Topic 27 (tối đa 3 phút)

Dùng `9_27_Overview.pptx`, đúng **4 slide**. Nhóm chọn một người trình bày overview. Sau đó có **12 phút Q/A**, giảng viên có thể hỏi bất kỳ thành viên nào. Deck 12 slide cũ chỉ dùng ôn tập.

| Slide | Thời gian | Nội dung |
|---|---:|---|
| 1. Problem & Research Question | 0:00–0:40 | Bài toán label skew và ba RQ |
| 2. Method & Experiments | 0:40–1:25 | Dataset/CNN/FedAvg, ba setup, 5 config x 3 seed |
| 3. Key Results | 1:25–2:15 | Mean ± SD, giảm 5,52 pp từ IID tới strong; E3 chênh +0,04 pp |
| 4. Conclusion / Demo | 2:15–2:55 | Kết luận, giới hạn, minh họa inference |

Tổng dự kiến 175 giây, còn 5 giây đệm. Speaker notes trong PPTX có nội dung tiếng Anh tương ứng. Đừng train trong giờ overview. Chuẩn bị sẵn checkpoint và mở demo trước giờ thi.

## Demo ngắn trên slide 4

Chọn strong seed 42, mở một ảnh test và đọc ground truth/prediction. Nêu đây là inference từ model đã train. Nếu chuyển cửa sổ lâu, dùng ảnh lỗi có sẵn trên slide để kết thúc đúng giờ.

## Chuẩn bị Q/A cho mọi thành viên

Mỗi người cần giải thích snapshot đầu round, trọng số n/N, quota strong vẫn có đủ 10 class, split tránh leakage, final versus best-validation checkpoint, và vì sao E3R10 không phải ablation giữ R cố định. Mở `fedavg.py`, `server.py`, `partition.py`, `DATA.md` và `outputs/audit.json` khi cần bằng chứng.

## Câu hỏi thường gặp

**Vì sao không dùng Flower?** Nhóm tự viết FedAvg để thể hiện rõ snapshot và phép trung bình từng tensor. Không cần framework distributed cho nghiên cứu mô phỏng này.

**Vì sao mỗi client không train nối tiếp trên cùng model?** Mỗi client phải nhận cùng global snapshot. Nếu client sau bắt đầu từ client trước thì thành sequential training, không đúng FedAvg. Test reset snapshot kiểm tra điều này.

**Vì sao strong vẫn có 10 class?** 92% ở hai class ưu thế, 8% còn lại chia cho tám class. Strong có 2.484 mẫu/class ưu thế và 54 mẫu/class khác.

**Vì sao FedAvg có trọng số khi các client bằng nhau?** Công thức chung dùng n_k/N. Thiết kế hiện tại n_k bằng nhau nên ra 0,1, nhưng test dùng số mẫu không bằng nhau để kiểm tra công thức.

**E3 có được train nhiều hơn không?** E3R10 có cùng 30 lượt train pool với E1R30. E tăng và R giảm đồng thời. Nhóm đánh giá trade-off compute cố định và số lần tổng hợp, không cô lập tác động riêng của E.

**Vì sao macro recall bằng accuracy?** Test cân bằng 1.000 mẫu/class, nên trung bình recall đều class trùng accuracy. Macro-F1 tính F1 mỗi class rồi trung bình, vẫn có thể khác.

**Chọn checkpoint bằng test có leakage không?** Nhóm chọn final checkpoint cho bảng chính, best-validation cho bảng phụ, hòa chọn mốc sớm. Test curves đánh giá sau train để mô tả. Không dùng max test để tune.

**Vì sao FL-IID chưa gần centralized?** FedAvg trung bình các điểm cuối của nhiều optimizer trajectories. Dù cùng ngân sách ảnh, quá trình tối ưu khác centralized trên minibatch trộn toàn pool. So sánh chỉ cho thiết kế này.

**Accuracy giảm có chứng minh gradient conflict không?** Chưa. Client drift là cách giải thích dựa trên lý thuyết và curves. Project chưa đo cosine gradient hay disagreement trực tiếp.

**FL có riêng tư tuyệt đối không?** Chưa. Server API không nhận ảnh train, nhưng model updates có thể rò rỉ thông tin. Chưa có differential privacy hoặc secure aggregation.

**Có train trên 10 máy không?** Đây là 10 client mô phỏng trên một máy. Dataset được chia index, từng client chỉ dùng subset để cập nhật. Không có cách ly bộ nhớ hoặc mạng thực.

**Có tái lập số liệu không?** Có seed, configs, split/partition indices, initial hash, commit, library versions, history và checkpoints. Bitwise reproduction giới hạn cùng môi trường/backend. CPU/CUDA/MPS có thể khác chút số học.

**Có thể kết luận rộng không?** Kết quả chỉ trong Fashion-MNIST, CNN nhỏ, balanced synthetic label skew, full participation và ba seed. Nhóm chưa tuyên bố thống kê mạnh.
