"""Independent audit of the exported study, never trains or tunes a model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import classification_metrics
from src.evaluation.tables import read_results
from src.utils.logging import atomic_json


def audit(output_dir):
    entries = read_results(output_dir)
    assert len(entries) == 15, f"Expected 15 completed runs, found {len(entries)}"
    checks = []
    for seed in [42, 43, 44]:
        group = [e for e in entries if e["seed"] == seed]
        assert len({e["manifest"]["initial_state_sha256"] for e in group}) == 1
        federated = [e for e in group if e["setting"] != "centralized"]
        assert len({tuple(e["manifest"]["partition"]["class_permutation"]) for e in federated}) == 1
        strong, e3 = [next(e for e in group if e["setting"] == s) for s in ["strong_non_iid", "strong_e3"]]
        assert strong["manifest"]["partition"]["sha256"] == e3["manifest"]["partition"]["sha256"]
        checks.append(f"Seed {seed}: common initialization, common class pairing, identical strong E1/E3 partition")
    split = np.load(Path(output_dir) / "splits/seed_2026.npz")
    assert len(split["train"]) == 54000 and len(split["val"]) == 6000
    assert len(np.unique(np.concatenate([split["train"], split["val"]]))) == 60000
    for e in entries:
        path = Path(e["path"])
        config = e["manifest"]["config"]
        assert e["examples_seen"] == 1620000
        assert e["optimizer_steps"] == (25320 if config["mode"] == "centralized" else 25500)
        history = pd.read_csv(path / "history.csv")
        curve = pd.read_csv(path / "test_history.csv")
        assert len(history) == config["rounds"] + 1 and len(curve) == len(history)
        assert history["round"].tolist() == list(range(config["rounds"] + 1))
        assert int(history.loc[history.global_val_accuracy.idxmax(), "round"]) == e["best_validation_round"]
        assert np.isclose(curve.iloc[-1].global_test_accuracy, e["accuracy"])
        assert e["n_samples"] == 10000 and np.all(np.array(e["confusion_matrix"]).sum(1) == 1000)
        assert np.isclose(e["macro_recall"], e["accuracy"])
        predictions = np.load(path / "predictions.npz")
        recomputed = classification_metrics(predictions["labels"], predictions["predictions"], e["loss"])
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            assert np.isclose(e[metric], recomputed[metric])
        partition_manifest = e["manifest"]["partition"]
        if partition_manifest:
            counts = np.array(partition_manifest["counts"])
            assert np.all(counts.sum(0) == 5400) and np.all(counts.sum(1) == 5400)
            partition_path = Path(output_dir) / "partitions" / f"lambda_{config['lambda']}_seed_{e['seed']}"
            indices = np.load(partition_path / "indices.npz")
            assert np.array_equal(
                np.sort(np.concatenate([indices[f"train_{k}"] for k in range(10)])), np.sort(split["train"])
            )
            assert np.array_equal(
                np.sort(np.concatenate([indices[f"val_{k}"] for k in range(10)])), np.sort(split["val"])
            )
        checks.append(
            f"{e['setting']} seed {e['seed']}: complete budget, validation selection, 10k predictions, recomputed metrics, partition coverage"
        )
    result = {
        "status": "passed",
        "completed_runs": len(entries),
        "total_training_examples_seen": sum(e["examples_seen"] for e in entries),
        "checks": checks,
    }
    atomic_json(Path(output_dir) / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    print(json.dumps(audit(parser.parse_args().output_dir), indent=2))
