"""Locked experimental protocol, atomic per-round checkpoints and post-training test audit."""

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from filelock import FileLock, Timeout

from src.data.dataset import load_data
from src.data.partition import partition
from src.evaluation.metrics import evaluate
from src.federated.server import federated_round
from src.models.cnn import SmallCNN
from src.training.centralized_train import centralized_epoch
from src.utils.logging import atomic_json
from src.utils.seed import seed_all, snapshot, state_hash, stream_seed

FIELDS = [
    "round",
    "effective_epochs",
    "examples_seen",
    "optimizer_steps",
    "weighted_local_train_loss",
    "global_val_loss",
    "global_val_accuracy",
    "global_val_macro_f1",
    "elapsed_seconds",
]


def load_config(path, seed=None):
    path = Path(path)
    config = yaml.safe_load((path.parent / "base.yaml").read_text())
    config.update(yaml.safe_load(path.read_text()))
    if seed is not None:
        config["seed"] = seed
    if config["mode"] not in ("centralized", "federated") or config["rounds"] < 1:
        raise ValueError("Invalid mode or training budget")
    if config["clients"] != 10 or config["dataset"] != "Fashion-MNIST":
        raise ValueError("This protocol supports Fashion-MNIST with exactly 10 simulated clients")
    if config["mode"] == "centralized" and config["local_epochs"] != 1:
        raise ValueError("Centralized uses one train-pool pass per epoch; local_epochs must be 1")
    if config["local_epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Local epochs, batch size and learning rate must be positive")
    return config


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def atomic_checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_history(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def r80(rows):
    for i in range(1, len(rows) - 2):
        if all(row["global_val_accuracy"] >= 0.8 for row in rows[i : i + 3]):
            return {"round": rows[i]["round"], "effective_epochs": rows[i]["effective_epochs"]}
    return None


def _run(config, resume=False, data=None):
    torch.set_num_threads(config.get("cpu_threads", 1))
    device = select_device(config["device"])
    seed_all(config["seed"])
    output = Path(config["output_dir"]) / "runs" / config["setting"] / f"seed_{config['seed']}"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing and existing["config"] != config:
        raise ValueError("Output already belongs to a different config; use a new output directory")
    if existing and not resume:
        raise FileExistsError("Run already exists. Pass --resume or use a fresh output directory")
    if existing and existing["status"] == "complete":
        print(f"Already complete: {output}", flush=True)
        return output
    if existing and existing["hardware"]["device"] != str(device):
        raise ValueError("Resolved backend changed during resume; use the original device or a new output directory")
    data = load_data(config) if data is None else data
    x, y = data["x"].to(device), data["y"].to(device)
    model = SmallCNN().to(device)
    initial = snapshot(model)
    if existing and existing["initial_state_sha256"] != state_hash(initial):
        raise ValueError("Initialization changed during resume; restore the recorded environment before continuing")
    partition_seed = stream_seed(config["seed"], "partition")
    clients = local_val = None
    partition_manifest = None
    if config["mode"] == "federated":
        clients, counts, partition_manifest = partition(
            data["train_idx"], data["labels"], config["lambda"], partition_seed
        )
        local_val, _, val_manifest = partition(data["val_idx"], data["labels"], config["lambda"], partition_seed)
        partition_dir = Path(config["output_dir"]) / "partitions" / f"lambda_{config['lambda']}_seed_{config['seed']}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            partition_dir / "indices.npz",
            **{f"train_{k}": v for k, v in enumerate(clients)},
            **{f"val_{k}": v for k, v in enumerate(local_val)},
        )
        atomic_json(partition_dir / "manifest.json", {"train": partition_manifest, "validation": val_manifest})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted"
    import torchvision
    import sklearn

    manifest = {
        "status": "running",
        "config": config,
        "code_commit": commit,
        "initial_state_sha256": state_hash(initial),
        "partition": partition_manifest,
        "hardware": {"device": str(device), "platform": platform.platform(), "processor": platform.processor()},
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "protocol": {
            "checkpoint_selection": "final budget and earliest best validation",
            "test_access": "after all training rounds",
            "test_curve": "post-hoc description only",
            "rng": "stateless seed streams per round/client/epoch",
        },
        "last_round": 0,
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    initial_path = checkpoint_dir / "round_000.pt"
    if not initial_path.exists():
        atomic_checkpoint(initial_path, {"state": initial, "round": 0, "config": config})
    rows, start_round, total_seen, total_steps, previous_time = [], 0, 0, 0, 0.0
    # Each immutable round file is the commit marker. Ignore incomplete temporary files.
    available = sorted(checkpoint_dir.glob("round_*.pt"))
    if resume and available:
        payload = torch.load(available[-1], map_location="cpu", weights_only=True)
        if payload["config"] != config:
            raise ValueError("Resume config mismatch")
        model.load_state_dict(payload["state"])
        start_round = payload["round"]
        rows = payload.get("history", [])
        if rows:
            total_seen = rows[-1]["examples_seen"]
            total_steps = rows[-1]["optimizer_steps"]
            previous_time = rows[-1]["elapsed_seconds"]
    if not rows:
        val, _, _ = evaluate(model, x, y, data["val_idx"])
        rows.append(dict(zip(FIELDS, [0, 0, 0, 0, 0, val["loss"], val["accuracy"], val["macro_f1"], 0])))
        atomic_checkpoint(initial_path, {"state": initial, "round": 0, "config": config, "history": rows})
    begin = time.perf_counter()
    atomic_json(manifest_path, manifest)
    try:
        for t in range(start_round + 1, config["rounds"] + 1):
            if clients is None:
                stats = centralized_epoch(model, x, y, data["train_idx"], config, t)
            else:
                stats = federated_round(model, x, y, clients, config, t)
            val, _, _ = evaluate(model, x, y, data["val_idx"])
            total_seen += stats["examples_seen"]
            total_steps += stats["optimizer_steps"]
            effective = t * (1 if clients is None else config["local_epochs"])
            rows.append(
                dict(
                    zip(
                        FIELDS,
                        [
                            t,
                            effective,
                            total_seen,
                            total_steps,
                            stats["loss_sum"] / stats["examples_seen"],
                            val["loss"],
                            val["accuracy"],
                            val["macro_f1"],
                            previous_time + time.perf_counter() - begin,
                        ],
                    )
                )
            )
            atomic_checkpoint(
                checkpoint_dir / f"round_{t:03d}.pt",
                {"state": snapshot(model), "round": t, "config": config, "history": rows},
            )
            write_history(output / "history.csv", rows)
            manifest["last_round"] = t
            atomic_json(manifest_path, manifest)
            print(
                f"{config['setting']} seed={config['seed']} round={t}/{config['rounds']} val_acc={val['accuracy']:.4f} elapsed={rows[-1]['elapsed_seconds']:.1f}s",
                flush=True,
            )
        # Selection is frozen from validation before test is materialized on the device.
        best_round = max(rows, key=lambda row: row["global_val_accuracy"])["round"]
        tx, ty = data["test_x"].to(device), data["test_y"].to(device)
        curve = []
        selected = {}
        for t in range(config["rounds"] + 1):
            payload = torch.load(checkpoint_dir / f"round_{t:03d}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state"])
            metrics, true, pred = evaluate(model, tx, ty)
            curve.append(
                {
                    "round": t,
                    "effective_epochs": rows[t]["effective_epochs"],
                    "global_test_accuracy": metrics["accuracy"],
                    "global_test_loss": metrics["loss"],
                    "global_test_macro_f1": metrics["macro_f1"],
                }
            )
            if t == best_round:
                selected["best_validation_test"] = metrics
            if t == config["rounds"]:
                selected["final_test"] = metrics
                np.savez_compressed(output / "predictions.npz", labels=true, predictions=pred)
        write_test_curve = output / "test_history.csv"
        with write_test_curve.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=curve[0].keys())
            writer.writeheader()
            writer.writerows(curve)
        selected.update(
            {
                "best_validation_round": best_round,
                "r80_validation": r80(rows),
                "examples_seen": total_seen,
                "optimizer_steps": total_steps,
                "training_seconds": rows[-1]["elapsed_seconds"],
            }
        )
        if local_val is not None:
            selected["local_validation"] = []
            for k, indices in enumerate(local_val):
                metrics, _, _ = evaluate(model, x, y, indices)
                selected["local_validation"].append(
                    {
                        "client": k,
                        "class_counts": np.bincount(data["labels"][indices], minlength=10).tolist(),
                        **metrics,
                    }
                )
        atomic_json(output / "final_metrics.json", selected)
        manifest.update(
            {
                "status": "complete",
                "last_round": config["rounds"],
                "wall_seconds": previous_time + time.perf_counter() - begin,
            }
        )
        atomic_json(manifest_path, manifest)
        return output
    except BaseException as error:
        manifest.update(
            {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)}
        )
        atomic_json(manifest_path, manifest)
        raise


def run(config, resume=False, data=None):
    output = Path(config["output_dir"]) / "runs" / config["setting"] / f"seed_{config['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    lock = FileLock(output / ".run.lock", timeout=0)
    try:
        with lock:
            return _run(config, resume, data)
    except Timeout as error:
        raise RuntimeError(f"Another process is already writing this run: {output}") from error
