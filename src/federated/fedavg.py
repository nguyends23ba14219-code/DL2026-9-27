import torch


def fedavg(client_states, sample_counts):
    """Aggregate model tensors only. No dataset access on the server."""
    if not client_states or len(client_states) != len(sample_counts):
        raise ValueError("Expected matching nonempty states and sample counts")
    if any(n <= 0 for n in sample_counts):
        raise ValueError("Sample counts must be positive")
    total = sum(sample_counts)
    first = client_states[0]
    output = {}
    for state in client_states:
        if state.keys() != first.keys():
            raise ValueError("State keys differ")
    for key, ref in first.items():
        if not ref.is_floating_point():
            raise ValueError("This CNN protocol aggregates floating tensors only")
        result = torch.zeros_like(ref)
        for state, n in zip(client_states, sample_counts):
            if state[key].shape != ref.shape or state[key].dtype != ref.dtype:
                raise ValueError("Tensor shape or dtype mismatch")
            if not torch.isfinite(state[key]).all():
                raise ValueError("Non-finite client weights")
            result.add_(state[key], alpha=n / total)
        output[key] = result
    return output
