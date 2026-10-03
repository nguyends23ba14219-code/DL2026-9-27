import hashlib

import numpy as np


def stratified_split(labels, seed=2026, val_per_class=600):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in range(10):
        indices = rng.permutation(np.flatnonzero(labels == c))
        if len(indices) <= val_per_class:
            raise ValueError('Each class must have more than val_per_class samples')
        val.extend(indices[:val_per_class])
        train.extend(indices[val_per_class:])
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64)


def partition(indices, labels, severity, seed):
    """Exact balanced cyclic quotas. No replacement, equal row/column totals."""
    if severity not in (0.0, 0.5, 0.9):
        raise ValueError('Supported lambda values: 0, 0.5, 0.9')
    indices, labels = np.asarray(indices), np.asarray(labels)
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Input indices must be unique')
    totals = np.bincount(labels[indices], minlength=10)
    if not np.all(totals == totals[0]):
        raise ValueError('Partition requires balanced classes')
    size = int(totals[0])
    base = size * (1 - severity) / 10
    extra = size * severity / 2
    if not np.isclose(base, round(base)) or not np.isclose(extra, round(extra)):
        raise ValueError('Dataset size does not support exact integer quotas')
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(10)
    counts = np.full((10, 10), round(base), dtype=np.int64)
    for k in range(10):
        counts[k, permutation[[k, (k + 1) % 10]]] += round(extra)
    clients = [[] for _ in range(10)]
    for c in range(10):
        pool = rng.permutation(indices[labels[indices] == c])
        offset = 0
        for k in range(10):
            n = int(counts[k, c])
            clients[k].extend(pool[offset:offset + n])
            offset += n
    clients = [np.array(x, dtype=np.int64) for x in clients]
    combined = np.concatenate(clients)
    assert np.array_equal(np.sort(combined), np.sort(indices))
    assert np.all(counts.sum(0) == size) and np.all(counts.sum(1) == size)
    manifest = {
        'lambda': severity, 'seed': seed, 'class_permutation': permutation.tolist(),
        'counts': counts.tolist(), 'samples_per_client': size,
        'total_variation': (np.abs(counts / size - 0.1).sum(1) / 2).tolist(),
        'sha256': hashlib.sha256(combined.tobytes()).hexdigest(),
    }
    return clients, counts, manifest
