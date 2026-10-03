import torch


def batches(x, y, indices, batch_size, seed=None):
    """In-memory tensor loader, same minibatch path for FL and centralized."""
    index = torch.as_tensor(indices, dtype=torch.long)
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        index = index[torch.randperm(len(index), generator=generator)]
    for start in range(0, len(index), batch_size):
        selected = index[start:start + batch_size].to(x.device)
        yield x[selected], y[selected]
