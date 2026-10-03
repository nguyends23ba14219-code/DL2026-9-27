# Project 27 — Federated Image Classification under Non-IID Data

Dự án Deep Learning hoàn chỉnh: tự triển khai **FedAvg bằng PyTorch**, khảo sát label skew trên Fashion-MNIST, so sánh với centralized, phân tích 3 seed và demo Streamlit. Toàn bộ code và tài liệu dùng một quy trình thống nhất, không phân chia thành viên.

**Protocol:** 10 simulated clients, Small CNN 105.866 parameters, 54.000 train / 6.000 validation / 10.000 official test, split seed 2026, run seeds 42/43/44. Năm cấu hình × ba seed = 15 runs. Bảng kết quả thật sẽ có trong [báo cáo](reports/report_vi.md) và [CSV](outputs/tables/summary.csv).

## Cài đặt

Python **3.12**. Khuyến nghị môi trường ảo. CPU chạy được toàn bộ, CUDA hoặc Apple MPS là tùy chọn.

```bash
git clone https://github.com/nguyends23ba14219-code/federated-image-classification.git
cd federated-image-classification
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Các phiên bản thư viện chính được pin trong `requirements.txt`. Môi trường dùng cho kết quả có snapshot trong `requirements-lock.txt`. Đã kiểm tra Python 3.12 trên macOS arm64, CPU. Notebook Colab cung cấp đường chạy CUDA nhưng không tuyên bố đã kiểm tra trực tiếp trên Colab. Reproducibility bitwise áp dụng cùng môi trường/backend, không bảo đảm giống bit qua CPU/CUDA/MPS.

## Chạy một cấu hình

Lần đầu tự tải Fashion-MNIST bằng HTTPS và kiểm tra checksum MD5 chính thức. Không commit cache dataset. Khuyến nghị `--output-dir outputs/reproduction` nếu muốn chạy lại kết quả đã có trong repo.

```bash
python main.py --config configs/iid.yaml --seed 42 --device cpu --output-dir outputs/reproduction
# Khôi phục từ round hoàn tất gần nhất, với đúng config cũ:
python main.py --config configs/iid.yaml --seed 42 --device cpu --output-dir outputs/reproduction --resume
```

Chọn `--device cuda`, `mps`, `cpu` hoặc `auto`. Resume tại ranh giới round, bỏ phần round dở. Checkpoint lưu cả history/config, shuffle dùng seed xác định theo round/client/epoch. Không ghi đè run khác config. Khi một run đã hoàn tất, `--resume` bỏ qua run đó.

## Chạy đủ 15 runs

```bash
python experiments/run_suite.py --device cpu --output-dir outputs/reproduction
python experiments/analyze.py --output-dir outputs/reproduction
```

Suite chạy lần lượt để phù hợp máy phổ thông và tự resume. Có thể chạy từng seed bằng `--seeds 42` hoặc chọn `--settings centralized iid`. Mỗi run chính có 1.620.000 lượt ảnh train. Tổng study có 24.300.000 lượt ảnh. Thời gian phụ thuộc máy, có thể kéo dài. Cấu hình chính không dùng dữ liệu rút gọn.

| Setting | λ | Local epochs | Rounds / epochs | Seeds |
|---|---:|---:|---:|---|
| Centralized | — | 1 | 30 epochs | 42,43,44 |
| IID | 0 | 1 | 30 rounds | 42,43,44 |
| Mild Non-IID | 0,5 | 1 | 30 rounds | 42,43,44 |
| Strong Non-IID | 0,9 | 1 | 30 rounds | 42,43,44 |
| Strong E3 | 0,9 | 3 | 10 rounds | 42,43,44 |

Các client có đúng 5.400 ảnh. Strong có 92% thuộc hai class ưu thế, vẫn chứa đủ 10 class. E3R10 và E1R30 có cùng ngân sách lượt ảnh. Centralized và các FL setting cùng seed có cùng khởi tạo.

## Kết quả và demo có sẵn

Repo lưu logs, metrics, predictions, split/partition indices, bảng và figures. Checkpoint mọi round nằm trong GitHub Release `v1.0.0`, tránh làm lớn Git history. Demo bundle chứa 15 checkpoint cuối và 100 ví dụ official test đã chọn cố định. Gói demo không cần GPU hoặc tải toàn bộ dataset.

```bash
# Repo riêng tư: gh auth login trước, gh có quyền truy cập tài khoản của anh.
python experiments/fetch_artifacts.py
streamlit run demo/app.py
```

Gói tải được kiểm tra SHA-256 bằng `artifacts/release_manifest.json`. Để tải mọi round và khôi phục đúng archive:

```bash
python experiments/fetch_artifacts.py --full
```

Artifact đã được train trước. Demo hiển thị đúng setting/seed, trung bình và SD, phân phối client, đường accuracy, confusion matrix và dự đoán một ảnh test. Thiếu artifact sẽ hiển thị “Chưa có kết quả”. Với output tự chạy, đặt `P27_OUTPUT_DIR=outputs/reproduction streamlit run demo/app.py` và bảo đảm dùng checkpoint cùng output đó.

## Tái tạo bảng, figures và báo cáo

```bash
python experiments/analyze.py
```

Lệnh yêu cầu đủ 15 run hoàn tất, không tự điền số liệu thiếu. Sinh 7 nhóm figures (PNG/PDF), bảng mean ± sample SD (ddof=1), số từng seed, class/client và `reports/report_vi.md`. Bảng chính dùng checkpoint cuối ngân sách. Bảng phụ dùng checkpoint validation tốt nhất, hòa chọn mốc sớm. Test curves được đánh giá sau train, không dùng test để tune.

## Cấu trúc

```text
configs/                  # Cấu hình cố định
src/data/                 # HTTPS download, stratified split, exact quotas, tensor loader
src/models/cnn.py         # 105.866 parameters
src/federated/            # client, server, weighted FedAvg
src/training/             # Cùng minibatch loop cho local và centralized
src/evaluation/           # Metrics, CSV tables, plots và report generator
src/runner.py             # Locked protocol, checkpoint/resume, post-hoc test
experiments/              # Suite, analysis, artifact export/fetch, audit
outputs/                  # Split/partition indices, run logs, metrics, figures, tables
notebooks/                # Colab quickstart, gọi lại code repo
artifacts/                # Release manifest, tải model/demo bundle tại đây
reports/                  # Báo cáo, slide bảo vệ và kịch bản 12 phút
 demo/app.py              # Streamlit chỉ đọc kết quả đã train
 tests/                   # Data integrity, FedAvg, metrics, CNN, resume, sanity checks
```

## Phạm vi và nguồn

FL là mô phỏng 10 client trên một máy, full participation. Server aggregation chỉ nhận state dict và sample counts, không nhận train images. Không triển khai privacy guarantee, secure aggregation, differential privacy, mạng thật hoặc personal models. Kết luận giới hạn Fashion-MNIST, CNN nhỏ và synthetic balanced label skew. Runtime trong study đo khi nhiều tiến trình CPU chạy đồng thời nên không dùng để kết luận tăng tốc.

Nguồn: [FedAvg (McMahan et al., 2017)](https://proceedings.mlr.press/v54/mcmahan17a.html), [Fashion-MNIST chính thức](https://github.com/zalandoresearch/fashion-mnist), [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html), [Deep Leakage from Gradients](https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html). Dataset do Zalando Research cung cấp theo MIT. Code dự án dùng MIT.
