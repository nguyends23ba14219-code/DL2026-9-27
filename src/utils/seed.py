import hashlib
import os
import random

import numpy as np
import torch


def stream_seed(seed, *keys):
    """Stable independent RNG streams, insensitive to experiment execution order."""
    token = ":".join(map(str, (seed, *keys))).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def seed_all(seed):
    # Required by deterministic CUDA matrix operations (PyTorch 2.8 reproducibility notes).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def state_hash(state):
    """Hash tensor contents in a stable key order for initialization/resume checks."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot(model):
    """Copy weights to CPU so later client updates cannot mutate the broadcast."""
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
