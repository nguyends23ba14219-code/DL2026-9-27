import json

import numpy as np
import pytest

from experiments.package_artifacts import round_checkpoints
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
