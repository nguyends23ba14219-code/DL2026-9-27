import hashlib
import random

import numpy as np
import torch


def stream_seed(seed, *keys):
    """Stable independent RNG streams, insensitive to experiment execution order."""
    token = ":".join(map(str, (seed, *keys))).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
