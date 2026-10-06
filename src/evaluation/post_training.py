"""Evaluate saved global models after training and validation selection are finished."""

import csv

import numpy as np
import torch

from src.evaluation.metrics import evaluate
from src.utils.logging import atomic_json


def r80(rows):
    """First post-initial round with three consecutive validation accuracies >= 80%."""
    for index in range(1, len(rows) - 2):
        if all(row["global_val_accuracy"] >= 0.8 for row in rows[index : index + 3]):
            return {"round": rows[index]["round"], "effective_epochs": rows[index]["effective_epochs"]}
    return None


def evaluate_checkpoints(model, data, config, output, rows, local_validation=None):
    """Write test curves, final metrics and final-model local validation metrics."""
    # max keeps the first tied row. Freeze this selection before using test data.
    best_round = max(rows, key=lambda row: row["global_val_accuracy"])["round"]
    device = next(model.parameters()).device
    test_x, test_y = data["test_x"].to(device), data["test_y"].to(device)
    curve, selected = [], {}
    for round_index in range(config["rounds"] + 1):
        checkpoint = output / "checkpoints" / f"round_{round_index:03d}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state"])
        metrics, labels, predictions = evaluate(model, test_x, test_y)
        curve.append(
            {
                "round": round_index,
                "effective_epochs": rows[round_index]["effective_epochs"],
                "global_test_accuracy": metrics["accuracy"],
                "global_test_loss": metrics["loss"],
                "global_test_macro_f1": metrics["macro_f1"],
            }
        )
        if round_index == best_round:
            selected["best_validation_test"] = metrics
        if round_index == config["rounds"]:
            selected["final_test"] = metrics
            np.savez_compressed(output / "predictions.npz", labels=labels, predictions=predictions)

    with (output / "test_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)
    selected.update(
        {
            "best_validation_round": best_round,
            "r80_validation": r80(rows),
            "examples_seen": rows[-1]["examples_seen"],
            "optimizer_steps": rows[-1]["optimizer_steps"],
            "training_seconds": rows[-1]["elapsed_seconds"],
        }
    )
    # The loop leaves the model at the final budget checkpoint, shared by all clients.
    if local_validation is not None:
        train_x, train_y = data["x"].to(device), data["y"].to(device)
        selected["local_validation"] = []
        for client_id, indices in enumerate(local_validation):
            metrics, _, _ = evaluate(model, train_x, train_y, indices)
            selected["local_validation"].append(
                {
                    "client": client_id,
                    "class_counts": np.bincount(data["labels"][indices], minlength=10).tolist(),
                    **metrics,
                }
            )
    atomic_json(output / "final_metrics.json", selected)
    return selected
