from src.federated.client import client_update
from src.federated.fedavg import fedavg
from src.utils.seed import snapshot


def federated_round(model, x, y, clients, config, round_index):
    global_state = snapshot(model)
    states, counts, stats = [], [], []
    for k, indices in enumerate(clients):
        state, local = client_update(model, global_state, x, y, indices, config, round_index, k)
        states.append(state)
        counts.append(local["n_samples"])
        stats.append(local)
    model.load_state_dict(fedavg(states, counts))
    return {key: sum(s[key] for s in stats) for key in ("loss_sum", "examples_seen", "optimizer_steps")}
