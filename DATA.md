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

`src/data/dataset.py` uses HTTPS and the official torchvision resource checksums. MD5 here identifies the published dataset files. Downloaded model releases use SHA-256.

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

## Downloadable processed/demo subset

The fixed processed demo subset is downloadable in [demo-artifacts.zip, release v1.0.0](https://github.com/nguyends23ba14219-code/DL2026-9-27/releases/tag/v1.0.0). It includes `artifacts/demo_examples.npz` with images, labels and `official_indices`, selected as the first ten official test indices per class (100 total). The original MIT license accompanies the data. This subset supports inference only and is not substituted for the full training or evaluation data.

```bash
python experiments/fetch_artifacts.py
```

The repository is private. An authorized GitHub account must have access for `gh release download`. SHA-256 and file sizes are recorded in `artifacts/release_manifest.json`. The full processed training dataset is reconstructed from the official files and committed indices rather than redistributed as a second image archive.
