"""Reproduce the official dataset split and all client partitions without training."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_data
from src.data.partition import partition
from src.runner import load_config
from src.utils.logging import atomic_json
from src.utils.seed import stream_seed


def prepare(config, seeds=(42, 43, 44), data=None):
    data = load_data(config) if data is None else data
    manifests = []
    for seed in seeds:
        partition_seed = stream_seed(seed, "partition")
        for severity in (0, 0.5, 0.9):
            clients, _, train_manifest = partition(data["train_idx"], data["labels"], severity, partition_seed)
            validation, _, val_manifest = partition(data["val_idx"], data["labels"], severity, partition_seed)
            directory = Path(config["output_dir"]) / "partitions" / f"lambda_{severity}_seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / "indices.npz",
                **{f"train_{k}": values for k, values in enumerate(clients)},
                **{f"val_{k}": values for k, values in enumerate(validation)},
            )
            manifest = {"train": train_manifest, "validation": val_manifest}
            atomic_json(directory / "manifest.json", manifest)
            manifests.append(manifest)
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/reproduction")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/iid.yaml")
    config.update(data_dir=args.data_dir, output_dir=args.output_dir)
    manifests = prepare(config, args.seeds)
    print(f"Verified 54k/6k/10k split and saved {len(manifests)} train/validation partitions.")


if __name__ == "__main__":
    main()
