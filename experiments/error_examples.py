"""Illustrate the first eight official-test errors for the largest confusion pair."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import CLASS_NAMES, HTTPSFashionMNIST
from src.evaluation.plots import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    output = Path(args.output_dir)
    predictions = np.load(output / "runs/strong_non_iid/seed_42/predictions.npz")
    dataset = HTTPSFashionMNIST(args.data_dir, train=False, download=True)
    labels, predicted = predictions["labels"], predictions["predictions"]
    if not np.array_equal(labels, dataset.targets.numpy()):
        raise ValueError("Predictions do not follow official test order")
    counts = np.zeros((10, 10), dtype=int)
    np.add.at(counts, (labels, predicted), 1)
    np.fill_diagonal(counts, 0)
    true, pred = np.unravel_index(counts.argmax(), counts.shape)
    selected = np.flatnonzero((labels == true) & (predicted == pred))[:8]
    if not len(selected):
        raise ValueError("No errors to illustrate")
    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    rows = []
    for ax, index in zip(axes.flat, selected):
        ax.imshow(dataset.data[index].numpy(), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Test #{index}\nTrue: {CLASS_NAMES[true]}\nPred: {CLASS_NAMES[pred]}", fontsize=10)
        ax.axis("off")
        rows.append({"test_index": int(index), "true_label": int(true), "predicted_label": int(pred)})
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")
    fig.suptitle("Strong E1R30, seed 42: first errors in largest confusion pair", fontsize=12)
    fig.tight_layout(h_pad=3, rect=(0, 0, 1, 0.94))
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    save(fig, figures, "07_error_examples")
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "error_examples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["test_index", "true_label", "predicted_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} actual errors; pair contains {counts[true, pred]} test images.")


if __name__ == "__main__":
    main()
