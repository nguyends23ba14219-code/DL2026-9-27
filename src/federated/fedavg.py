import torch


def fedavg(client_states, sample_counts):
    """Return sum(n_k / N * w_k), validating tensors without accessing datasets."""
    if not client_states or len(client_states) != len(sample_counts):
        raise ValueError("Expected matching nonempty states and sample counts")
    if any(n <= 0 for n in sample_counts):
        raise ValueError("Sample counts must be positive")
    total_samples = sum(sample_counts)
    reference_state = client_states[0]
    output = {}
    for state in client_states:
        if state.keys() != reference_state.keys():
            raise ValueError("State keys differ")
    for key, reference in reference_state.items():
        if not reference.is_floating_point():
            raise ValueError("This CNN protocol aggregates floating tensors only")
        result = torch.zeros_like(reference)
        for state, count in zip(client_states, sample_counts):
            tensor = state[key]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError("Tensor shape or dtype mismatch")
            if not torch.isfinite(tensor).all():
                raise ValueError("Non-finite client weights")
            result.add_(tensor, alpha=count / total_samples)
        output[key] = result
    return output
