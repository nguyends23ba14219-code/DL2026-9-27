"""Read-only demo of previously trained Fashion-MNIST artifacts."""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import CLASS_NAMES, HTTPSFashionMNIST
from src.evaluation.tables import LABELS, SETTINGS
from src.models.cnn import SmallCNN

OUTPUT = Path(os.environ.get("P27_OUTPUT_DIR", str(ROOT / "outputs")))
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Project 27 · Federated Learning", layout="wide")
st.title("Federated Image Classification")
st.caption("Fashion-MNIST · 10 client mô phỏng · Kết quả đã train trước, đọc từ artifact thật.")
setting = st.sidebar.selectbox("Cấu hình", SETTINGS, format_func=lambda s: LABELS[s])
choice = st.sidebar.selectbox("Kết quả", ["Trung bình 3 seed", "Seed 42", "Seed 43", "Seed 44"])
seeds = [42, 43, 44] if choice == "Trung bình 3 seed" else [int(choice[-2:])]
paths = [OUTPUT / "runs" / setting / f"seed_{seed}" for seed in seeds]
if not all((path / "final_metrics.json").exists() for path in paths):
    st.info("Chưa có kết quả cho lựa chọn này.")
    st.stop()
results = [json.loads((path / "final_metrics.json").read_text()) for path in paths]
manifests = [json.loads((path / "manifest.json").read_text()) for path in paths]
if any(m["status"] != "complete" for m in manifests):
    st.info("Chưa có kết quả hoàn tất cho lựa chọn này.")
    st.stop()
config = manifests[0]["config"]
st.write(
    f"**{LABELS[setting]}**: {config['rounds']} {'epochs' if config['mode'] == 'centralized' else 'rounds'}, E={config['local_epochs']}, batch={config['batch_size']}, SGD lr={config['learning_rate']}"
)
cols = st.columns(3)
for col, metric, title in zip(cols, ["accuracy", "macro_f1", "loss"], ["Test accuracy", "Macro-F1", "Test loss"]):
    values = np.array([result["final_test"][metric] for result in results])
    display = f"{100 * values.mean():.2f}%" if metric == "accuracy" else f"{values.mean():.4f}"
    sd = values.std(ddof=1) if len(values) > 1 else 0
    col.metric(title, display, delta=None)
    col.caption(f"SD={100 * sd:.2f} điểm %" if metric == "accuracy" else f"SD={sd:.4f}")
curves = [pd.read_csv(path / "test_history.csv") for path in paths]
values = np.stack([frame.global_test_accuracy for frame in curves])
fig, ax = plt.subplots(figsize=(9, 3.5))
xs = curves[0]["effective_epochs"]
ax.plot(xs, values.mean(0), color="#007f73")
if len(values) > 1:
    ax.fill_between(
        xs, values.mean(0) - values.std(0, ddof=1), values.mean(0) + values.std(0, ddof=1), alpha=0.2, color="#007f73"
    )
ax.set(xlabel="Effective epochs", ylabel="Global test accuracy", ylim=(0, 1))
ax.grid(alpha=0.2)
st.pyplot(fig)
plt.close(fig)
st.caption("Đường test đánh giá sau khi train xong. Checkpoint chọn theo validation, không chọn bằng test.")
left, right = st.columns(2)
with left:
    if manifests[0]["partition"]:
        st.write("Phân phối train, seed đầu tiên của lựa chọn")
        counts = np.array(manifests[0]["partition"]["counts"])
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(counts, vmin=0, vmax=2484, cmap="YlGnBu")
        ax.set(xlabel="Class", ylabel="Client", xticks=range(10), yticks=range(10))
        fig.colorbar(image, ax=ax, label="Số ảnh train")
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Quota chính xác"):
            st.dataframe(pd.DataFrame(counts, columns=CLASS_NAMES), width="stretch")
    else:
        st.write("Centralized train trên toàn bộ 54.000 ảnh.")
with right:
    st.write("Confusion matrix chuẩn hóa theo nhãn thật")
    cm = np.mean([np.array(result["final_test"]["confusion_matrix_normalized"]) for result in results], axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thật", xticks=range(10), yticks=range(10))
    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)
with st.expander("Cấu hình, provenance và kết quả từng seed"):
    st.json(config)
    st.dataframe(
        pd.DataFrame(
            [
                {"seed": seed, **{key: result["final_test"][key] for key in ["accuracy", "macro_f1", "loss"]}}
                for seed, result in zip(seeds, results)
            ]
        )
    )
    st.json(
        {
            "initial_hashes": [m["initial_state_sha256"] for m in manifests],
            "code_commits": [m["code_commit"] for m in manifests],
        }
    )
st.subheader("Dự đoán ảnh test có sẵn")
example_path = ARTIFACTS / "demo_examples.npz"
if example_path.exists():
    examples = np.load(example_path)
    images, labels, official_indices = examples["images"], examples["labels"], examples["official_indices"]
else:
    try:
        dataset = HTTPSFashionMNIST(str(ROOT / "data"), train=False, download=False)
        official_indices = np.concatenate([np.flatnonzero(dataset.targets.numpy() == c)[:10] for c in range(10)])
        images = dataset.data.numpy()[official_indices]
        labels = dataset.targets.numpy()[official_indices]
    except RuntimeError:
        st.info("Tải artifact demo từ GitHub Release bằng python experiments/fetch_artifacts.py để xem ảnh offline.")
        st.stop()
example = st.slider("Ảnh minh họa (100 ảnh test cố định)", 0, len(images) - 1, 0)
seed = 42 if len(seeds) > 1 else seeds[0]
checkpoint = ARTIFACTS / "models" / setting / f"seed_{seed}.pt"
if os.environ.get("P27_OUTPUT_DIR") or not checkpoint.exists():
    checkpoint = OUTPUT / "runs" / setting / f"seed_{seed}" / "checkpoints" / f"round_{config['rounds']:03d}.pt"
if checkpoint.exists():
    model = SmallCNN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state"])
    model.eval()
    image = torch.from_numpy(images[example]).float().unsqueeze(0).unsqueeze(0) / 255
    with torch.inference_mode():
        probabilities = model((image - 0.5) / 0.5).softmax(1)[0].numpy()
    st.image(
        images[example],
        width=168,
        caption=f"Official test #{official_indices[example]} · Nhãn thật: {CLASS_NAMES[int(labels[example])]} · Dự đoán: {CLASS_NAMES[int(probabilities.argmax())]}",
    )
    st.bar_chart(pd.DataFrame({"Xác suất": probabilities}, index=CLASS_NAMES))
else:
    st.info("Chưa có checkpoint cho cấu hình/seed này. Tải demo artifact hoặc chạy cấu hình tương ứng.")
