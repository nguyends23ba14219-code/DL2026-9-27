import hashlib
from pathlib import Path

import numpy as np

from src.utils.logging import atomic_json
from src.utils.seed import stream_seed


def stratified_split(labels, seed=2026, val_per_class=600):
    """Reserve the same number of validation images from each of the ten classes."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for class_id in range(10):
        indices = rng.permutation(np.flatnonzero(labels == class_id))
        if len(indices) <= val_per_class:
            raise ValueError("Each class must have more than val_per_class samples")
        val.extend(indices[:val_per_class])
        train.extend(indices[val_per_class:])
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64)


def partition(indices, labels, severity, seed):
    """Exact balanced cyclic quotas. No replacement, equal row/column totals."""
    if severity not in (0.0, 0.5, 0.9):
        raise ValueError("Supported lambda values: 0, 0.5, 0.9")
    indices, labels = np.asarray(indices), np.asarray(labels)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Input indices must be unique")
    totals = np.bincount(labels[indices], minlength=10)
    if not np.all(totals == totals[0]):
        raise ValueError("Partition requires balanced classes")
    samples_per_class = int(totals[0])
    base = samples_per_class * (1 - severity) / 10
    extra = samples_per_class * severity / 2
    if not np.isclose(base, round(base)) or not np.isclose(extra, round(extra)):
        raise ValueError("Dataset size does not support exact integer quotas")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(10)
    counts = np.full((10, 10), round(base), dtype=np.int64)
    # Each client has two adjacent dominant classes; every class appears in two pairs.
    for client_id in range(10):
        dominant_classes = permutation[[client_id, (client_id + 1) % 10]]
        counts[client_id, dominant_classes] += round(extra)
    clients = [[] for _ in range(10)]
    for class_id in range(10):
        pool = rng.permutation(indices[labels[indices] == class_id])
        offset = 0
        for client_id in range(10):
            quota = int(counts[client_id, class_id])
            clients[client_id].extend(pool[offset : offset + quota])
            offset += quota
    clients = [np.array(client, dtype=np.int64) for client in clients]
    combined = np.concatenate(clients)
    assert np.array_equal(np.sort(combined), np.sort(indices))
    assert np.all(counts.sum(0) == samples_per_class) and np.all(counts.sum(1) == samples_per_class)
    manifest = {
        "lambda": severity,
        "seed": seed,
        "class_permutation": permutation.tolist(),
        "counts": counts.tolist(),
        "samples_per_client": samples_per_class,
        "total_variation": (np.abs(counts / samples_per_class - 0.1).sum(1) / 2).tolist(),
        "sha256": hashlib.sha256(combined.tobytes()).hexdigest(),
    }
    return clients, counts, manifest


def prepare_partitions(data, output_dir, severity, seed):
    """Build and save matching train/validation partitions for one experiment seed."""
    partition_seed = stream_seed(seed, "partition")
    clients, _, train_manifest = partition(data["train_idx"], data["labels"], severity, partition_seed)
    validation, _, val_manifest = partition(data["val_idx"], data["labels"], severity, partition_seed)
    directory = Path(output_dir) / "partitions" / f"lambda_{severity}_seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "indices.npz",
        **{f"train_{client_id}": indices for client_id, indices in enumerate(clients)},
        **{f"val_{client_id}": indices for client_id, indices in enumerate(validation)},
    )
    manifest = {"train": train_manifest, "validation": val_manifest}
    atomic_json(directory / "manifest.json", manifest)
    return clients, validation, manifest
