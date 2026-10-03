from pathlib import Path

import numpy as np

from src.evaluation.tables import LABELS, SETTINGS


def make_report(entries, summary, output_dir):
    rows = {row["setting"]: row for row in summary}
    table = "| Cấu hình | Test accuracy (%) | Macro-F1 | Test CE | Seeds |\n|---|---:|---:|---:|---:|\n"
    for setting in SETTINGS:
        row = rows[setting]
        table += f"| {LABELS[setting]} | {100 * row['accuracy_mean']:.2f} ± {100 * row['accuracy_sd']:.2f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_sd']:.4f} | {row['loss_mean']:.4f} ± {row['loss_sd']:.4f} | {row['n_seeds']} |\n"
    cen = rows["centralized"]["accuracy_mean"]
    iid = rows["iid"]["accuracy_mean"]
    mild = rows["mild_non_iid"]["accuracy_mean"]
    strong = rows["strong_non_iid"]["accuracy_mean"]
    e3 = rows["strong_e3"]["accuracy_mean"]
    cls = np.array([[c["recall"] for c in e["per_class"]] for e in entries if e["setting"] == "strong_non_iid"]).mean(0)
    worst = int(cls.argmin())
    from src.data.dataset import CLASS_NAMES

    illustration = next(e for e in entries if e["setting"] == "strong_non_iid" and e["seed"] == 42)
    matrix = np.array(illustration["confusion_matrix"])
    np.fill_diagonal(matrix, 0)
    true, pred = np.unravel_index(matrix.argmax(), matrix.shape)
    checkpoints = "\n".join(
        f"| {LABELS[s]} | {rows[s]['r80_reached']} / 3 | {rows[s]['r80_round_mean']:.2f} | {rows[s]['r80_effective_epochs_mean']:.2f} |"
        for s in SETTINGS
    )
    hardware = entries[0]["manifest"]["hardware"]
    text = f"""# Project 27: Federated Image Classification under Non-IID Data

## Tóm tắt

Dự án tự triển khai FedAvg bằng PyTorch trên Fashion-MNIST để khảo sát ảnh hưởng của label skew. Nghiên cứu hoàn tất **15 lần chạy thật**, gồm 5 cấu hình × 3 seed (42, 43, 44), với 10 client mô phỏng, CNN 105.866 tham số và cùng ngân sách 30 lượt toàn bộ train pool. Bảng chính dùng checkpoint cuối ngân sách. Test accuracy trung bình của centralized là {cen * 100:.2f}%, IID {iid * 100:.2f}%, mild {mild * 100:.2f}%, strong {strong * 100:.2f}% và strong E3R10 {e3 * 100:.2f}%.

Tất cả số liệu trong báo cáo sinh từ `outputs/runs/*/seed_*/final_metrics.json` và CSV. Accuracy trong log nằm trong [0,1]. Bảng chuyển accuracy sang phần trăm. Độ lệch chuẩn dùng mẫu, ddof=1, n=3. Dự án không chia công việc theo thành viên.

## 1. Bài toán và phạm vi

Câu hỏi chính là: mức độ khác biệt tỷ lệ nhãn giữa các client ảnh hưởng thế nào đến chất lượng bộ phân loại ảnh federated? Một client chỉ train trên tập dữ liệu cục bộ. Server nhận trọng số và số mẫu để tổng hợp thành global model. Toàn bộ hệ thống chạy trong một chương trình hoặc các tiến trình nghiên cứu trên một máy, không phải hệ thống triển khai trên 10 thiết bị độc lập.

Không truyền ảnh trong bước aggregation giúp giữ dữ liệu train ngoài API server. Điều này chưa tạo bảo đảm riêng tư tuyệt đối: cập nhật mô hình vẫn có thể rò rỉ thông tin. Dự án không triển khai secure aggregation hay differential privacy.

## 2. Dữ liệu và chống leakage

Fashion-MNIST gồm 60.000 ảnh official train và 10.000 ảnh official test, ảnh xám 28×28, 10 nhãn. Official train chia stratified cố định với split seed=2026 thành 54.000 ảnh train (5.400/class) và 6.000 validation (600/class). Các index được lưu trong `outputs/splits/seed_2026.npz`.

Tiền xử lý là `(uint8/255 - 0.5)/0.5`, không augmentation, không ước lượng thống kê từ test. Train loop chỉ nhận train indices. Validation dùng eval và inference_mode. Test được đánh giá sau khi hoàn tất train và đã chọn checkpoint bằng validation. Kiến trúc, SGD, learning rate, seeds, quota và ngân sách đã khóa trước các lần chạy chính thức. Không thay cấu hình sau khi xem test.

## 3. Thiết kế balanced label skew

Tại mỗi seed, tạo một hoán vị nhãn π cố định. Client k ưu thế ở π[k] và π[(k+1) mod 10]. Mỗi class ưu thế tại đúng hai client. Tỷ lệ được quy định bởi:

`p(k,c) = (1 − λ)/10 + (λ/2) × I[c thuộc hai class ưu thế]`.

| Phân phối | λ | Mỗi class ưu thế | Mỗi class khác | Tổng/client | TV |
|---|---:|---:|---:|---:|---:|
| IID | 0 | 540 | 540 | 5.400 | 0 |
| Mild | 0,5 | 1.620 | 270 | 5.400 | 0,4 |
| Strong | 0,9 | 2.484 | 54 | 5.400 | 0,72 |

Strong vẫn chứa đủ 10 class. Hai class ưu thế chiếm 92% dữ liệu của mỗi client. Quota giữ cố định cả số mẫu/client và tổng mỗi class toàn hệ thống, giúp tách ảnh hưởng label skew khỏi ảnh hưởng số mẫu. Các kiểm tra bảo đảm không lặp index, không thiếu index, tổng hàng/cột đúng và TV=0,8λ. Giữ cùng π cho IID/mild/strong/E3 trong một seed.

![Phân phối client](../{output_dir}/figures/01_client_distributions.png)

## 4. CNN và thuật toán

CNN gồm Conv(1,16,3,padding=1), ReLU, MaxPool(2), Conv(16,32,3,padding=1), ReLU, MaxPool(2), Flatten(1.568), Linear(1.568,64), ReLU, Linear(64,10). Tổng số tham số là 105.866. Mô hình trả 10 logits. CrossEntropyLoss nhận logits trực tiếp. Không BatchNorm, dropout hoặc pretrained weights.

Mỗi round, server chụp global snapshot. Cả 10 client nhận đúng snapshot này, train local bằng SGD rồi trả state dict cùng n_samples. FedAvg thực hiện `w_next = Σ (n_k / Σ n_k) × w_k`. Với quota cân bằng, mỗi client có hệ số 0,1. Các tensor được clone để tránh dùng chung tham chiếu. SGD không momentum, không weight decay, learning rate=0,01, batch size=64, giữ batch cuối. Server không backprop trên train images.

Centralized và FL dùng cùng hàm xử lý minibatch. RNG của initialization, partition và shuffle được quản lý riêng. Shuffle xác định từ seed/round/client/epoch nên resume tại ranh giới round tái lập đúng mà không phụ thuộc thứ tự chạy. Hash initial weights được lưu trong manifest và bằng nhau giữa 5 cấu hình trong mỗi seed.

## 5. Ma trận thực nghiệm

| Cấu hình | R/epochs | Local E | Lượt toàn pool | Số lượt ảnh | Seeds |
|---|---:|---:|---:|---:|---|
| Centralized | 30 epochs | 1 | 30 | 1.620.000 | 42,43,44 |
| IID | 30 rounds | 1 | 30 | 1.620.000 | 42,43,44 |
| Mild | 30 rounds | 1 | 30 | 1.620.000 | 42,43,44 |
| Strong | 30 rounds | 1 | 30 | 1.620.000 | 42,43,44 |
| Strong E3 | 10 rounds | 3 | 30 | 1.620.000 | 42,43,44 |

So E1R30 với E3R10 giữ số lượt ảnh, thay đồng thời số local epoch và tần suất tổng hợp. Đây là trade-off dưới ngân sách cố định, không đo ảnh hưởng riêng của E khi mọi biến khác giữ nguyên. Số optimizer steps hơi khác do batch cuối: centralized 25.320, FL 25.500. Không suy ra thời gian chạy bằng nhau.

Thiết bị thực tế: `{hardware["device"]}` trên `{hardware["platform"]}`. Các lần chạy CPU dùng một thread mỗi tiến trình và có chạy đồng thời, vì vậy wall time phản ánh cả tranh chấp tài nguyên. Không dùng runtime để tuyên bố tăng tốc phần cứng hay communication thực tế. Manifest lưu phiên bản thư viện, code commit, hash initialization, quota, config và trạng thái.

## 6. Kết quả cuối ngân sách

{table}

![So sánh cuối ngân sách](../{output_dir}/figures/04_final_comparison.png)

Chênh lệch IID trừ centralized là {(iid - cen) * 100:+.2f} điểm phần trăm. Mild trừ IID là {(mild - iid) * 100:+.2f} điểm, strong trừ IID là {(strong - iid) * 100:+.2f} điểm. Đây là số đo trong thiết kế này, chưa chứng minh centralized luôn là cận trên hay label skew luôn làm giảm accuracy đơn điệu.

![Test accuracy](../{output_dir}/figures/02_test_accuracy.png)

Đường test được đánh giá lại từ checkpoint sau khi train hoàn tất. Dải màu là mean ± sample SD qua ba seed. Không dùng đường test để chọn hyperparameter hoặc checkpoint. Centralized epoch không đưa vào trục communication round. Mốc đầu round 0 cũng được ghi.

## 7. Loss và mốc validation

Weighted local train loss là tổng cross entropy trong các cập nhật local chia cho tổng lượt ảnh trong round. Global validation loss đánh giá model sau aggregation trên tập validation cố định. Hai đại lượng đo trên các model và phân phối khác nhau.

![Loss](../{output_dir}/figures/03_losses.png)

R@80% là round đầu chuỗi ba round liên tiếp có validation accuracy ≥80%, không tính round 0. Khi không đạt, CSV để trống. Trung bình mốc dưới đây chỉ tính trên seed đạt, luôn kèm số seed đạt.

| Cấu hình | Seed đạt | Round trung bình khi đạt | Effective epochs trung bình |
|---|---:|---:|---:|
{checkpoints}

Mốc ba round tương ứng lượng compute khác nhau giữa E1/E3 nên chỉ là chỉ báo phụ, không phải hội tụ toán học. Bảng phụ `outputs/tables/per_seed.csv` có test accuracy tại checkpoint validation accuracy cao nhất, với quy tắc hòa chọn mốc sớm nhất. Không chọn max test accuracy.

## 8. Local epochs và phân tích lỗi

E3R10 có test accuracy trung bình {e3 * 100:.2f}%, chênh {(e3 - strong) * 100:+.2f} điểm phần trăm so với E1R30. Số lần tổng hợp giảm từ 30 xuống 10 trong cùng 1.620.000 lượt ảnh. Client local đi xa hơn giữa các lần tổng hợp có thể làm tăng drift. Các số đo ở đây không trực tiếp đo gradient conflict.

![Local epochs](../{output_dir}/figures/06_local_epochs.png)

Trong strong E1, class có recall trung bình thấp nhất là **{CLASS_NAMES[worst]}**, recall={cls[worst]:.4f}. Ở confusion matrix seed 42 được chọn trước, cặp nhầm nhiều nhất ngoài đường chéo là **{CLASS_NAMES[true]} → {CLASS_NAMES[pred]}**, {int(matrix[true, pred])} ảnh. Điều này mô tả lỗi thật, không khẳng định nguyên nhân duy nhất là label skew.

![Per class](../{output_dir}/figures/07_per_class.png)

![Confusion matrix chuẩn hóa](../{output_dir}/figures/05_confusion_normalized.png)

Chia validation thành 10 tập độc lập, 600 mẫu/client, cùng π và λ nhưng khác train indices. Strong có 276 mẫu mỗi class ưu thế và 6 mẫu mỗi class còn lại. Đánh giá cùng final global model trên từng tập. Không diễn giải đây là 10 mô hình cá nhân. Client index không đại diện một class pair cố định giữa các seed vì π thay theo seed, nên biểu đồ theo client là mô tả tổng hợp sơ bộ. Histogram từng client/seed nằm trong CSV. Macro-F1 local có thể nhiễu do số mẫu nhỏ ở vài class.

![Local validation](../{output_dir}/figures/07_local_validation.png)

## 9. Kiểm thử, tái lập và demo

Kiểm thử bao gồm split không leakage, quota/coverage/TV, FedAvg có trọng số bằng tensor tính tay, metric với đáp án biết trước, số tham số CNN, mọi client bắt đầu cùng snapshot, overfit tập nhỏ và resume cho hash trọng số giống lần chạy liên tục. Bộ kiểm tra không cần tải dataset để chạy CI. Kiểm tra end-to-end dùng dữ liệu giả chỉ phục vụ correctness, không đưa vào bảng nghiên cứu.

Lệnh chạy, khôi phục, phân tích và mở Streamlit nằm trong README. Tất cả 15 run có history.csv, post-hoc test_history.csv, predictions.npz, final_metrics.json và manifest. Checkpoint mọi round lưu local hoặc archive release, không commit dataset/cache. Demo đọc các artifact đã train, hiển thị rõ nguồn. Demo có chế độ từng seed hoặc trung bình 3 seed, heatmap, curves, metrics, confusion matrix và dự đoán ảnh test. Nếu artifact thiếu, hiển thị “Chưa có kết quả”.

## 10. Hạn chế và kết luận

Kết luận chỉ áp dụng Fashion-MNIST, CNN nhỏ, balanced synthetic label skew, 10 client và full participation. Ba seed cung cấp đánh giá độ ổn định sơ bộ, chưa đủ cho tuyên bố thống kê mạnh. Mô phỏng chưa có mạng thực, client bỏ cuộc, domain shift, data quantity skew hoặc privacy guarantee. Không đo gradient disagreement trực tiếp, nên cơ chế client drift là cách giải thích phù hợp cần kiểm chứng thêm.

Dự án đã cung cấp pipeline hoàn chỉnh, phép so sánh cùng ngân sách, 15 run tái lập, phân tích class/client và demo từ artifact thật. Kết quả định lượng nằm trong bảng và có thể truy ngược đến từng seed. Future work có thể khảo sát partial participation, Dirichlet, FedProx/SCAFFOLD hoặc secure aggregation trong nghiên cứu riêng.

## Tài liệu tham khảo

1. McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. https://proceedings.mlr.press/v54/mcmahan17a.html
2. Xiao et al. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. https://github.com/zalandoresearch/fashion-mnist
3. Karimireddy et al. (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. https://proceedings.mlr.press/v119/karimireddy20a.html
4. Zhu et al. (2019). Deep Leakage from Gradients. https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html
5. PyTorch. Reproducibility. https://docs.pytorch.org/docs/stable/notes/randomness.html
"""
    directory = Path("reports")
    directory.mkdir(exist_ok=True)
    (directory / "report_vi.md").write_text(text)
