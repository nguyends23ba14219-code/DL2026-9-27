import json
import hashlib
import zipfile

import numpy as np
import pytest

from experiments.package_artifacts import round_checkpoints
from experiments.fetch_artifacts import verify_and_extract
from experiments.prepare_data import prepare
from src.data.partition import partition
from src.utils.seed import stream_seed


def test_prepared_partitions_match_training_protocol(tmp_path):
    labels = np.repeat(np.arange(10), 200)
    train = np.concatenate([np.arange(c * 200, c * 200 + 100) for c in range(10)])
    val = np.concatenate([np.arange(c * 200 + 100, (c + 1) * 200) for c in range(10)])
    data = {"labels": labels, "train_idx": train, "val_idx": val}
    prepare({"output_dir": str(tmp_path)}, seeds=[43], data=data)
    clients, _, expected = partition(train, labels, 0.9, stream_seed(43, "partition"))
    directory = tmp_path / "partitions/lambda_0.9_seed_43"
    saved = np.load(directory / "indices.npz")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["train"] == expected
    for k in range(10):
        assert np.array_equal(saved[f"train_{k}"], clients[k])
    assert np.array_equal(np.sort(np.concatenate([saved[f"val_{k}"] for k in range(10)])), val)


def test_archive_rejects_missing_intermediate_round(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "round_000.pt").touch()
    (directory / "round_002.pt").touch()
    with pytest.raises(RuntimeError, match="round_001.pt"):
        round_checkpoints(tmp_path, 2)
    (directory / "round_001.pt").touch()
    assert len(round_checkpoints(tmp_path, 2)) == 3


def test_public_archive_verifies_and_extracts(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("artifacts/processed-data/README.md", "verified study data")
    metadata = {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size}
    destination = tmp_path / "fresh-repository"
    verify_and_extract(archive, metadata, destination)
    assert (destination / "artifacts/processed-data/README.md").read_text() == "verified study data"


@pytest.mark.parametrize("invalid", ["sha256", "bytes"])
def test_archive_integrity_failure_does_not_extract(tmp_path, invalid):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("data.txt", "study data")
    metadata = {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size}
    metadata[invalid] = "0" * 64 if invalid == "sha256" else archive.stat().st_size + 1
    destination = tmp_path / "fresh-repository"
    with pytest.raises(RuntimeError, match="SHA-256 or size"):
        verify_and_extract(archive, metadata, destination)
    assert not destination.exists()


def test_archive_traversal_rejected_before_any_extraction(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("data.txt", "safe")
        bundle.writestr("../outside.txt", "unsafe")
    metadata = {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size}
    destination = tmp_path / "fresh-repository"
    with pytest.raises(RuntimeError, match="Unsafe archive path"):
        verify_and_extract(archive, metadata, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()
