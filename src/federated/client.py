from src.training.local_train import train_local
from src.utils.seed import snapshot


def client_update(model, global_state, x, y, indices, config, round_index, client_id):
    """Train one client from the broadcast weights and return an independent copy."""
    # Reload the same frozen broadcast for EVERY client, never chain client updates.
    model.load_state_dict(global_state)
    stats = train_local(model, x, y, indices, config, round_index, client_id)
    return snapshot(model), stats
