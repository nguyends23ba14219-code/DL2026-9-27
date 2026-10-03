"""Bundle official-study checkpoints and deterministic test examples for releases."""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import HTTPSFashionMNIST
from src.evaluation.tables import read_results
from src.utils.logging import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    args = parser.parse_args()
    release_dir = Path(args.release_dir).resolve()
    release_dir.mkdir(parents=True, exist_ok=True)
    entries = read_results(ROOT / "outputs" / "official")
    if len(entries) != 15:
        raise RuntimeError("Release requires all 15 completed study runs")
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    dataset = HTTPSFashionMNIST(str(ROOT / "data"), train=False, download=False)
    indices = np.concatenate([np.flatnonzero(dataset.targets.numpy() == c)[:10] for c in range(10)])
    np.savez_compressed(
        artifacts / "demo_examples.npz",
        images=dataset.data.numpy()[indices],
        labels=dataset.targets.numpy()[indices],
        official_indices=indices,
    )
    demo_files = [artifacts / "demo_examples.npz", artifacts / "FASHION-MNIST-LICENSE.txt"]
    checkpoint_files = []
    for entry in entries:
        path = Path(entry["path"])
        config = entry["manifest"]["config"]
        checkpoint = path / "checkpoints" / f"round_{config['rounds']:03d}.pt"
        target = artifacts / "models" / entry["setting"] / f"seed_{entry['seed']}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target)
        demo_files.append(target)
        checkpoint_files.extend(sorted((path / "checkpoints").glob("round_*.pt")))
    assets = {}
    for name, files in [("demo-artifacts.zip", demo_files), ("checkpoints.zip", checkpoint_files)]:
        archive = release_dir / name
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for file in files:
                bundle.write(file, file.relative_to(ROOT))
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        assets[name] = {"sha256": digest, "bytes": archive.stat().st_size, "files": len(files)}
    atomic_json(
        artifacts / "release_manifest.json",
        {
            "tag": "v1.0.0",
            "repository": "nguyends23ba14219-code/federated-image-classification",
            "assets": assets,
            "demo_examples": {
                "source": "Fashion-MNIST official test",
                "selection": "first 10 official indices per class",
                "license": "MIT, Zalando Research",
            },
        },
    )
    (release_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{metadata['sha256']}  {name}\n" for name, metadata in assets.items())
    )
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
