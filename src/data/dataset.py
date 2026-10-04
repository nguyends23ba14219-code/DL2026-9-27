from pathlib import Path
import os
import json
import requests
from torchvision.datasets.utils import check_integrity, extract_archive

import numpy as np
import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from src.data.partition import stratified_split
from src.utils.logging import atomic_json

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class HTTPSFashionMNIST(FashionMNIST):
    # torchvision's historical HTTP mirror can fail with 403. Preserve MD5 checks.
    def download(self):
        if self._check_exists():
            return
        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)
        for filename, md5 in self.resources:
            destination = Path(self.raw_folder) / filename
            if not check_integrity(str(destination), md5):
                temporary = destination.with_suffix(destination.suffix + ".part")
                with requests.get(self.mirrors[0] + filename, stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                if not check_integrity(str(temporary), md5):
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError("Dataset checksum mismatch: " + filename)
                os.replace(temporary, destination)
            extract_archive(str(destination), self.raw_folder)

    mirrors = ['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/']


class DatasetResult(tuple):
    """4-tuple (train_dataset, val_dataset, test_dataset, split_manifest) hỗ trợ cả dict access cho các runner cũ."""
    def __new__(cls, train_dataset, val_dataset, test_dataset, split_manifest, **extra):
        obj = super().__new__(cls, (train_dataset, val_dataset, test_dataset, split_manifest))
        obj._extra = extra
        return obj

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._extra[item]
        return super().__getitem__(item)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._extra
        return super().__contains__(item)

    def get(self, key, default=None):
        return self._extra.get(key, default)


def load_data(config=None):
    if config is None:
        config = {}
    root = config.get('data_dir', 'data')
    split_seed = config.get('split_seed', 2026)

    # Tiền xử lý: ToTensor(), Normalize((0.5,), (0.5,))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    official = HTTPSFashionMNIST(root, train=True, download=True, transform=transform)
    test = HTTPSFashionMNIST(root, train=False, download=True, transform=transform)
    labels = official.targets.numpy() if hasattr(official.targets, 'numpy') else np.array(official.targets)

    # Phân chia: Stratified với seed cố định np.random.default_rng(2026)
    rng = np.random.default_rng(split_seed)
    train_indices, val_indices = [], []
    for c in range(10):
        c_indices = rng.permutation(np.flatnonzero(labels == c))
        val_indices.extend(c_indices[:600].tolist())
        train_indices.extend(c_indices[600:].tolist())

    train_indices = [int(i) for i in train_indices]
    val_indices = [int(i) for i in val_indices]

    # Trả về Subset cho train và val
    train_dataset = Subset(official, train_indices)
    val_dataset = Subset(official, val_indices)
    test_dataset = test

    # Tạo dict split_manifest
    split_manifest = {
        'seed': split_seed,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_dataset),
        'train_indices': train_indices,
        'val_indices': val_indices,
    }

    # Lưu dict này ra file outputs/splits/split_manifest.json
    os.makedirs('outputs/splits', exist_ok=True)
    with open('outputs/splits/split_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(split_manifest, f, indent=2)

    # Duy trì các file split seed và manifest cũ cho tương thích
    split_dir = Path(config.get('output_dir', 'outputs')) / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"seed_{split_seed}.npz"
    if not split_path.exists():
        np.savez_compressed(split_path, train=np.array(train_indices), val=np.array(val_indices))

    manifest_path = split_dir / 'manifest.json'
    if not manifest_path.exists():
        atomic_json(manifest_path, {
            'split_seed': split_seed, 'train': len(train_indices), 'validation': len(val_indices),
            'test': len(test), 'train_counts': np.bincount(labels[train_indices]).tolist(),
            'validation_counts': np.bincount(labels[val_indices]).tolist(),
            'normalization': '(uint8 / 255 - 0.5) / 0.5', 'test_source': 'official independent test set',
        })

    def images(raw):
        return raw.unsqueeze(1).float().div_(255).sub_(0.5).div_(0.5)

    extra = {
        'x': images(official.data), 'y': official.targets,
        'test_x': images(test.data), 'test_y': test.targets,
        'train_idx': np.array(train_indices, dtype=np.int64),
        'val_idx': np.array(val_indices, dtype=np.int64),
        'labels': labels,
    }

    return DatasetResult(train_dataset, val_dataset, test_dataset, split_manifest, **extra)
