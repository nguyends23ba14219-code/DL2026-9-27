from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import CLASS_NAMES
from src.evaluation.tables import LABELS, SETTINGS

COLORS = ["#27374D", "#007F73", "#E69F00", "#C44752", "#7055A4"]


def save(fig, directory, name):
    fig.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def curve(ax, entries, setting, filename, metric, x_key, label=None):
    frames = [pd.read_csv(Path(e["path"]) / filename) for e in entries if e["setting"] == setting]
    if not frames:
        return
    xs = frames[0][x_key].to_numpy()
    values = np.stack([f[metric].to_numpy() for f in frames])
    mean = values.mean(0)
    sd = values.std(0, ddof=1) if len(frames) > 1 else np.zeros_like(mean)
    color = COLORS[SETTINGS.index(setting)]
    ax.plot(xs, mean, color=color, label=label or LABELS[setting])
    ax.fill_between(xs, mean - sd, mean + sd, color=color, alpha=0.15)


def make_plots(entries, output_dir):
    directory = Path(output_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, grid = plt.subplots(2, 2, figsize=(9.4, 8.3), constrained_layout=True)
    axes = grid.flatten()[:3]
    grid.flatten()[3].axis("off")
    for ax, setting in zip(axes, SETTINGS[1:4]):
        item = next(e for e in entries if e["setting"] == setting and e["seed"] == 42)
        counts = np.array(item["manifest"]["partition"]["counts"])
        image = ax.imshow(counts, vmin=0, vmax=2484, cmap="YlGnBu")
        ax.set(title=LABELS[setting], xlabel="Class", ylabel="Client")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        for k in range(10):
            for c in range(10):
                ax.text(
                    c,
                    k,
                    str(counts[k, c]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if counts[k, c] > 1500 else "#222222",
                )
    fig.colorbar(image, ax=axes, label="Training samples")
    save(fig, directory, "01_client_distributions")
    for name, filename, metric, y_label in [
        ("02_test_accuracy", "test_history.csv", "global_test_accuracy", "Global test accuracy"),
        ("02_validation_accuracy", "history.csv", "global_val_accuracy", "Global validation accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for setting in SETTINGS[1:4]:
            curve(ax, entries, setting, filename, metric, "round")
        ax.set(xlabel="Communication round", ylabel=y_label, ylim=(0, 1))
        ax.legend()
        ax.grid(alpha=0.2)
        save(fig, directory, name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for setting in SETTINGS[1:4]:
        curve(axes[0], entries, setting, "history.csv", "weighted_local_train_loss", "round")
        curve(axes[1], entries, setting, "history.csv", "global_val_loss", "round")
    axes[0].set(title="Loss during local updates", ylabel="Weighted local training CE", xlabel="Communication round")
    axes[1].set(title="Loss after aggregation", ylabel="Global validation CE", xlabel="Communication round")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    save(fig, directory, "03_losses")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
        means, sds = [], []
        for setting in SETTINGS:
            values = [e[metric] for e in entries if e["setting"] == setting]
            means.append(np.mean(values))
            sds.append(np.std(values, ddof=1))
        ax.bar(range(5), means, yerr=sds, color=COLORS, capsize=5)
        ax.set_xticks(range(5), [LABELS[s] for s in SETTINGS], rotation=20, ha="right")
        ax.set(ylabel="Final test " + metric, ylim=(0, 1))
    fig.tight_layout()
    save(fig, directory, "04_final_comparison")
    for normalized in [False, True]:
        fig, grid = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
        axes = grid.flatten()
        for ax, setting in zip(axes, SETTINGS[:4]):
            item = next(e for e in entries if e["setting"] == setting and e["seed"] == 42)
            key = "confusion_matrix_normalized" if normalized else "confusion_matrix"
            cm = np.array(item[key])
            image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalized else 1000)
            ax.set(title=LABELS[setting], xlabel="Predicted label", ylabel="True label")
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
        fig.colorbar(image, ax=axes, label="Row proportion" if normalized else "Sample count")
        save(fig, directory, "05_confusion_" + ("normalized" if normalized else "counts"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for setting in ["strong_non_iid", "strong_e3"]:
        curve(axes[0], entries, setting, "test_history.csv", "global_test_accuracy", "round")
        curve(axes[1], entries, setting, "test_history.csv", "global_test_accuracy", "effective_epochs")
    axes[0].set(xlabel="Communication round", ylabel="Global test accuracy")
    axes[1].set(xlabel="Effective epochs (complete train-pool passes)", ylabel="Global test accuracy")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    save(fig, directory, "06_local_epochs")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.8))
    for setting in SETTINGS[:4]:
        group = [e for e in entries if e["setting"] == setting]
        for ax, metric in zip(axes, ["recall", "f1"]):
            values = np.array([[c[metric] for c in e["per_class"]] for e in group])
            ax.errorbar(
                range(10),
                values.mean(0),
                yerr=values.std(0, ddof=1),
                label=LABELS[setting],
                color=COLORS[SETTINGS.index(setting)],
                capsize=2,
            )
    for ax, metric in zip(axes, ["recall", "F1"]):
        ax.set(ylabel="Test per-class " + metric, ylim=(0, 1))
        ax.set_xticks(range(10), CLASS_NAMES, rotation=40, ha="right")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, directory, "07_per_class")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for setting in SETTINGS[1:]:
        group = [e for e in entries if e["setting"] == setting]
        values = np.array([[c["accuracy"] for c in e["raw"]["local_validation"]] for e in group])
        ax.errorbar(
            range(10),
            values.mean(0),
            yerr=values.std(0, ddof=1),
            label=LABELS[setting],
            color=COLORS[SETTINGS.index(setting)],
            capsize=3,
        )
    ax.set(xlabel="Client validation subset", ylabel="Final global-model accuracy", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, directory, "07_local_validation")
