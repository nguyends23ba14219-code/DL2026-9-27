import json
import os
import pytest
from src.data.dataset import load_data
def test_data_integrity():
    train_dataset, val_dataset, test_dataset, split_manifest = load_data({})
    # Kiểm tra kích thước tập train = 54000, val = 6000, test = 10000
    assert len(train_dataset) == 54000
    assert len(val_dataset) == 6000
    assert len(test_dataset) == 10000
    assert split_manifest["train_size"] == 54000
    assert split_manifest["val_size"] == 6000
    assert split_manifest["test_size"] == 10000
    # Kiểm tra không có data leakage: set(train_indices) & set(val_indices) == set()
    train_indices = split_manifest["train_indices"]
    val_indices = split_manifest["val_indices"]
    assert set(train_indices) & set(val_indices) == set()
    # Kiểm tra file split_manifest.json tồn tại, load lên có seed == 2026
    manifest_path = os.path.join("outputs", "splits", "split_manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["seed"] == 2026