from src.federated.client import client_update
from src.federated.fedavg import fedavg
from src.utils.seed import snapshot


def federated_round(model, x, y, clients, config, round_index):
    """Broadcast once, train all clients, then replace global weights with FedAvg."""
    global_state = snapshot(model)
    client_states, sample_counts = [], []
    totals = {"loss_sum": 0, "examples_seen": 0, "optimizer_steps": 0}
    for client_id, indices in enumerate(clients):
        state, stats = client_update(model, global_state, x, y, indices, config, round_index, client_id)
        client_states.append(state)
        sample_counts.append(stats["n_samples"])
        for key in totals:
            totals[key] += stats[key]
    model.load_state_dict(fedavg(client_states, sample_counts))
    return totals
