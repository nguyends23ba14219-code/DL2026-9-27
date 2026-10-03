from pathlib import Path
import os
import requests
from torchvision.datasets.utils import check_integrity, extract_archive

import numpy as np
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


def load_data(config):
    root = config['data_dir']
    official = HTTPSFashionMNIST(root, train=True, download=True)
    test = HTTPSFashionMNIST(root, train=False, download=True)
    labels = official.targets.numpy()
    train_idx, val_idx = stratified_split(labels, config['split_seed'])
    split_dir = Path(config['output_dir']) / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"seed_{config['split_seed']}.npz"
    if split_path.exists():
        saved = np.load(split_path)
        if not np.array_equal(saved['train'], train_idx) or not np.array_equal(saved['val'], val_idx):
            raise ValueError('Persisted split differs from fixed protocol')
    else:
        np.savez_compressed(split_path, train=train_idx, val=val_idx)
    atomic_json(split_dir / 'manifest.json', {
        'split_seed': config['split_seed'], 'train': len(train_idx), 'validation': len(val_idx),
        'test': len(test), 'train_counts': np.bincount(labels[train_idx]).tolist(),
        'validation_counts': np.bincount(labels[val_idx]).tolist(),
        'normalization': '(uint8 / 255 - 0.5) / 0.5', 'test_source': 'official independent test set',
    })
    def images(raw):
        return raw.unsqueeze(1).float().div_(255).sub_(0.5).div_(0.5)
    return {
        'x': images(official.data), 'y': official.targets,
        'test_x': images(test.data), 'test_y': test.targets,
        'train_idx': train_idx, 'val_idx': val_idx, 'labels': labels,
    }
