# Dataset: Fashion-MNIST

## Official source, version and license

- Official dataset: https://github.com/zalandoresearch/fashion-mnist
- Paper: https://arxiv.org/abs/1708.07747
- Version used: the original **2017 Fashion-MNIST IDX distribution**, with 60,000 official training and 10,000 official test images. Upstream does not assign a semantic version to these IDX files; the exact file identities below define the version.
- Download base: https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/
- License: MIT, Zalando Research; see `artifacts/FASHION-MNIST-LICENSE.txt`.
- Images: uint8 grayscale, 28 x 28, one channel. Labels: integers 0 through 9.

| Official file | Official MD5 |
|---|---|
| train-images-idx3-ubyte.gz | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| train-labels-idx1-ubyte.gz | 25c81989df183df01b3e8a0aad5dffbe |
| t10k-images-idx3-ubyte.gz | bef4ecab320f06d8554ea6380940ec79 |
| t10k-labels-idx1-ubyte.gz | bb300cfdad3c16e7a12a480ee83cd310 |

`src/data/dataset.py` uses HTTPS and the official torchvision resource checksums. MD5 here identifies the published dataset files. Downloaded release archives use SHA-256.

## Class mapping

0: T-shirt/top; 1: Trouser; 2: Pullover; 3: Dress; 4: Coat; 5: Sandal; 6: Shirt; 7: Sneaker; 8: Bag; 9: Ankle boot.

## Train, validation and test

`stratified_split` in `src/data/partition.py` permutes indices independently within each class using NumPy default_rng with split seed **2026**. The first 600 indices per class are validation; the remaining 5,400 are training.

| Subset | Images | Images/class | Source |
|---|---:|---:|---|
| Train | 54,000 | 5,400 | Official training split |
| Validation | 6,000 | 600 | Official training split |
| Test | 10,000 | 1,000 | Unchanged official test split |

Index arrays refer to the original IDX file order. The training/validation split is shared across all run seeds and settings. No validation/test index participates in local or centralized gradient updates. Test is evaluated after training and validation checkpoint selection.

## Preprocessing

1. Read the original image tensor and add a channel dimension: N x 1 x 28 x 28.
2. Convert uint8 to float32 and divide by 255.
3. Apply `(x - 0.5) / 0.5`, giving the fixed range [-1, 1].
4. Keep original resolution. No augmentation, resizing, class filtering or fitted statistics.

The exact same procedure applies to every configuration. This is a reproducible normalization of the official dataset, not a separately collected dataset.

## Client partitions

Ten clients each receive 5,400 training images. Run seeds 42, 43 and 44 determine a partition stream through `stream_seed(seed, "partition")`. Each client has two dominant classes in a cyclic random permutation. Proportions are `(1-lambda)/10 + lambda/2` for the two dominant labels, and `(1-lambda)/10` otherwise.

| Lambda | Dominant class/client | Other class/client | TV from uniform |
|---|---:|---:|---:|
| 0.0 | 540 | 540 | 0.00 |
| 0.5 | 1,620 | 270 | 0.40 |
| 0.9 | 2,484 | 54 | 0.72 |

Every class is dominant at two clients. Row and column totals are 5,400. Indices are allocated once without replacement. Strong clients have all ten labels and 92% of their data in two labels. E1R30 and E3R10 use the identical strong partition within each seed.

Local validation partitions use the same class permutation on disjoint validation indices. Each client receives 600 validation images. Strong quotas are 276 for each dominant label and six for each other label. These are distribution-specific evaluations of the global model.

## Reproduce data without training

From the repository root, after installing `requirements-dev.txt`:

```bash
python experiments/prepare_data.py --data-dir data --output-dir outputs/reproduction
```

This downloads/checks official files, reproduces the 54k/6k/10k split and all nine train/validation client partitions (three severities x three seeds). It writes split indices and manifests under `outputs/reproduction/splits/` and `outputs/reproduction/partitions/`. No model training is required. The saved official indices are committed under `outputs/splits/` and `outputs/partitions/` for direct comparison.

To reproduce the experiment data during training, `main.py` and `experiments/run_suite.py` call the same loader and partition functions. Dataset caches under `data/` are ignored by Git.

## Download the complete processed study dataset

[processed-data.zip, release v1.0.0](https://github.com/nguyends23ba14219-code/DL2026-9-27/releases/download/v1.0.0/processed-data.zip) contains the full data used by the experiments, including normalization, train/validation split and all client allocations. This is separate from the smaller inference-only demo bundle.

```bash
python experiments/fetch_artifacts.py --data
```

The command downloads over public HTTPS, verifies SHA-256 and file size against `artifacts/release_manifest.json`, and extracts under `artifacts/processed-data/`. No GitHub login or GitHub CLI is required.

| Archive file, relative to `artifacts/processed-data/` | Contents |
|---|---|
| `official_train.npz` | All 60,000 official training images after preprocessing, plus labels; original IDX row order |
| `official_test.npz` | All 10,000 official test images after preprocessing, plus labels; original IDX row order |
| `splits/seed_2026.npz` | `train` (54,000) and `val` (6,000) row indices into `official_train.npz` |
| `partitions/lambda_{0,0.5,0.9}_seed_{42,43,44}/indices.npz` | Nine partitions, each with `train_0` through `train_9` and `val_0` through `val_9`; indices address the original official training rows |
| `splits/manifest.json`, partition manifests and root `manifest.json` | Counts, normalization, exact official MD5 identities, label mapping and protocol |
| `README.md`, `FASHION-MNIST-LICENSE.txt` | Loading notes and the upstream MIT license |

Images are float32 arrays of shape `N x 1 x 28 x 28`, already in [-1, 1]. Labels are int64. **Do not normalize these arrays again.** Strong E1R30 and E3R10 use the same strong partition. All split and client index arrays are checked against the committed experiment indices before packaging.

Example: inspect the exact data seen by client 0 in the strong setting, seed 42:

```python
from pathlib import Path
import numpy as np

root = Path("artifacts/processed-data")
with np.load(root / "official_train.npz", allow_pickle=False) as data:
    images, labels = data["images"], data["labels"]
with np.load(root / "splits/seed_2026.npz", allow_pickle=False) as split:
    train_idx, val_idx = split["train"], split["val"]
with np.load(root / "partitions/lambda_0.9_seed_42/indices.npz", allow_pickle=False) as clients:
    client_x = images[clients["train_0"]]
    client_y = labels[clients["train_0"]]
assert client_x.shape == (5400, 1, 28, 28)
assert len(train_idx) == 54000 and len(val_idx) == 6000
```

Rebuild the downloadable archive from the official source and verify every index against the recorded study:

```bash
python experiments/package_processed_data.py --data-dir data --release-dir /tmp/project27-data-release
```

The exporter verifies all four original gzip MD5 checksums, uses the same float32 tensor preprocessing as training, regenerates/compares the split and nine partitions, packages the license, and records the archive SHA-256 and byte count in the release manifest. `main.py` still uses the official-source loader; the archive supplies an independently downloadable copy of the same tensors and indices, without changing training or evaluation.

## Downloadable demo subset

The fixed demo subset is downloadable in [demo-artifacts.zip, release v1.0.0](https://github.com/nguyends23ba14219-code/DL2026-9-27/releases/download/v1.0.0/demo-artifacts.zip). It includes `artifacts/demo_examples.npz` with uint8 images, labels and `official_indices`, selected as the first ten official test indices per class (100 total). The demo applies the same normalization at inference. The original MIT license accompanies the data. This subset supports inference only and is not substituted for the full training or evaluation data.

```bash
python experiments/fetch_artifacts.py
```

The repository and release assets are public. Downloads work without a GitHub account. SHA-256 and file sizes for the complete processed dataset, demo bundle and full checkpoints are recorded in `artifacts/release_manifest.json` and the release `SHA256SUMS.txt`.
