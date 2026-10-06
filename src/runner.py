"""Locked experimental protocol, atomic per-round checkpoints and post-training test audit."""

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
import yaml
from filelock import FileLock, Timeout

from src.data.dataset import load_data
from src.data.partition import prepare_partitions
from src.evaluation.metrics import evaluate
from src.evaluation.post_training import evaluate_checkpoints
from src.evaluation.post_training import r80 as r80  # Preserve the original import path.
from src.federated.server import federated_round
from src.models.cnn import SmallCNN
from src.training.centralized_train import centralized_epoch
from src.utils.logging import atomic_json
from src.utils.seed import seed_all, snapshot, state_hash

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
    """Merge one setting with base.yaml, then validate the locked study protocol."""
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
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
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


def _build_manifest(config, device, initial_state, partition_manifest):
    """Record provenance and protocol alongside each run."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted"
    return {
        "status": "running",
        "config": config,
        "code_commit": commit,
        "initial_state_sha256": state_hash(initial_state),
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


def _restore_history(model, checkpoint_dir, config, resume):
    """Restore weights and history from the most recent completed round."""
    # Each immutable round file is the commit marker. Ignore incomplete temporary files.
    available = sorted(checkpoint_dir.glob("round_*.pt"))
    if resume and available:
        payload = torch.load(available[-1], map_location="cpu", weights_only=True)
        if payload["config"] != config:
            raise ValueError("Resume config mismatch")
        model.load_state_dict(payload["state"])
        start_round = payload["round"]
        return payload.get("history", []), start_round
    return [], 0


def _history_row(round_index, config, validation, *, examples_seen=0, optimizer_steps=0, training_loss=0, elapsed=0):
    """Name every CSV field explicitly to avoid positional field/value mismatches."""
    epochs_per_round = 1 if config["mode"] == "centralized" else config["local_epochs"]
    return {
        "round": round_index,
        "effective_epochs": round_index * epochs_per_round,
        "examples_seen": examples_seen,
        "optimizer_steps": optimizer_steps,
        "weighted_local_train_loss": training_loss,
        "global_val_loss": validation["loss"],
        "global_val_accuracy": validation["accuracy"],
        "global_val_macro_f1": validation["macro_f1"],
        "elapsed_seconds": elapsed,
    }


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
    # Keep one shared training tensor pair on the device, including local validation.
    data = {**data, "x": data["x"].to(device), "y": data["y"].to(device)}
    x, y = data["x"], data["y"]
    model = SmallCNN().to(device)
    initial = snapshot(model)
    if existing and existing["initial_state_sha256"] != state_hash(initial):
        raise ValueError("Initialization changed during resume; restore the recorded environment before continuing")
    clients = local_validation = partition_manifest = None
    if config["mode"] == "federated":
        clients, local_validation, partitions = prepare_partitions(
            data, config["output_dir"], config["lambda"], config["seed"]
        )
        partition_manifest = partitions["train"]
    manifest = _build_manifest(config, device, initial, partition_manifest)
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    initial_path = checkpoint_dir / "round_000.pt"
    if not initial_path.exists():
        atomic_checkpoint(initial_path, {"state": initial, "round": 0, "config": config})
    rows, start_round = _restore_history(model, checkpoint_dir, config, resume)
    if not rows:
        val, _, _ = evaluate(model, x, y, data["val_idx"])
        rows.append(_history_row(0, config, val))
        atomic_checkpoint(initial_path, {"state": initial, "round": 0, "config": config, "history": rows})
    total_seen = rows[-1]["examples_seen"]
    total_steps = rows[-1]["optimizer_steps"]
    previous_time = rows[-1]["elapsed_seconds"]
    begin = time.perf_counter()
    atomic_json(manifest_path, manifest)
    try:
        for round_index in range(start_round + 1, config["rounds"] + 1):
            if clients is None:
                stats = centralized_epoch(model, x, y, data["train_idx"], config, round_index)
            else:
                stats = federated_round(model, x, y, clients, config, round_index)
            val, _, _ = evaluate(model, x, y, data["val_idx"])
            total_seen += stats["examples_seen"]
            total_steps += stats["optimizer_steps"]
            rows.append(
                _history_row(
                    round_index,
                    config,
                    val,
                    examples_seen=total_seen,
                    optimizer_steps=total_steps,
                    training_loss=stats["loss_sum"] / stats["examples_seen"],
                    elapsed=previous_time + time.perf_counter() - begin,
                )
            )
            atomic_checkpoint(
                checkpoint_dir / f"round_{round_index:03d}.pt",
                {"state": snapshot(model), "round": round_index, "config": config, "history": rows},
            )
            write_history(output / "history.csv", rows)
            manifest["last_round"] = round_index
            atomic_json(manifest_path, manifest)
            print(
                f"{config['setting']} seed={config['seed']} round={round_index}/{config['rounds']} "
                f"val_acc={val['accuracy']:.4f} elapsed={rows[-1]['elapsed_seconds']:.1f}s",
                flush=True,
            )
        evaluate_checkpoints(model, data, config, output, rows, local_validation)
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
    """Train/evaluate one experiment while holding an exclusive output-directory lock."""
    output = Path(config["output_dir"]) / "runs" / config["setting"] / f"seed_{config['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    lock = FileLock(output / ".run.lock", timeout=0)
    try:
        with lock:
            return _run(config, resume, data)
    except Timeout as error:
        raise RuntimeError(f"Another process is already writing this run: {output}") from error
