"""Export the full normalized dataset and verify indices against the recorded study."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from torchvision.datasets.utils import check_integrity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.prepare_data import prepare
from src.data.dataset import CLASS_NAMES, HTTPSFashionMNIST, load_data
from src.runner import load_config
from src.utils.logging import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--release-dir", required=True, type=Path)
    args = parser.parse_args()
    args.release_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        destination = staging / "artifacts/processed-data"
        destination.mkdir(parents=True)
        config = load_config(ROOT / "configs/iid.yaml")
        config.update(data_dir=args.data_dir, output_dir=str(destination))
        data = load_data(config)
        official = HTTPSFashionMNIST(args.data_dir, train=True, download=True)
        for filename, md5 in official.resources:
            if not check_integrity(str(Path(official.raw_folder) / filename), md5):
                raise RuntimeError(f"Official source checksum mismatch: {filename}")
        prepare(config, data=data)
        for generated in destination.rglob("*.npz"):
            relative = generated.relative_to(destination)
            recorded = ROOT / "outputs" / relative
            with np.load(generated, allow_pickle=False) as current, np.load(recorded, allow_pickle=False) as saved:
                if set(current.files) != set(saved.files):
                    raise RuntimeError(f"Index keys differ: {relative}")
                for key in current.files:
                    if not np.array_equal(current[key], saved[key]):
                        raise RuntimeError(f"Recorded study indices differ: {relative}/{key}")
        np.savez_compressed(destination / "official_train.npz", images=data["x"].numpy(), labels=data["y"].numpy())
        np.savez_compressed(
            destination / "official_test.npz", images=data["test_x"].numpy(), labels=data["test_y"].numpy()
        )
        shutil.copy2(ROOT / "artifacts/FASHION-MNIST-LICENSE.txt", destination / "FASHION-MNIST-LICENSE.txt")
        atomic_json(destination / "manifest.json", {
            "source": "https://github.com/zalandoresearch/fashion-mnist",
            "version": "original 2017 IDX distribution",
            "official_md5": dict(official.resources),
            "class_names": CLASS_NAMES,
            "images": {"dtype": "float32", "shape": "N x 1 x 28 x 28", "range": [-1, 1]},
            "labels": {"dtype": "int64", "range": [0, 9]},
            "normalization": "(uint8 / 255 - 0.5) / 0.5",
            "counts": {"train": 54000, "validation": 6000, "test": 10000},
            "split_seed": 2026, "run_seeds": [42, 43, 44], "lambda": [0, 0.5, 0.9],
            "indexing": "train/validation indices address official_train.npz in original IDX order",
            "recorded_indices_verified": True,
        })
        (destination / "README.md").write_text(
            "# Complete processed Fashion-MNIST study data\n\n"
            "official_train.npz contains 60,000 normalized official training images and labels. "
            "official_test.npz contains all 10,000 normalized test images and labels. "
            "Images are float32 N x 1 x 28 x 28; labels are int64. Do not normalize again.\n\n"
            "splits/seed_2026.npz selects 54,000 train and 6,000 validation rows in official_train.npz. "
            "partitions/lambda_{0,0.5,0.9}_seed_{42,43,44}/indices.npz contains train_0..train_9 "
            "and val_0..val_9 in the same original row coordinate system. "
            "These arrays match the committed study arrays exactly. Strong E1/E3 share partitions.\n\n"
            "All NPZ files can be opened with numpy.load(path, allow_pickle=False). "
            "See root DATA.md for an executable loading example, provenance and reproduction commands. "
            "The original MIT dataset license is included.\n"
        )
        archive = args.release_dir / "processed-data.zip"
        files = sorted(p for p in staging.rglob("*") if p.is_file())
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:
            for file in files:
                bundle.write(file, file.relative_to(staging))
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest_path = ROOT / "artifacts/release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"][archive.name] = {"sha256": digest, "bytes": archive.stat().st_size, "files": len(files)}
    atomic_json(manifest_path, manifest)
    (args.release_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{metadata['sha256']}  {name}\n" for name, metadata in manifest["assets"].items())
    )
    print(json.dumps(manifest["assets"][archive.name], indent=2))


if __name__ == "__main__":
    main()
