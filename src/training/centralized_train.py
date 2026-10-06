from src.training.local_train import train_local


def centralized_epoch(model, x, y, indices, config, epoch):
    """Use the same SGD loop for one complete pass over the central training pool."""
    return train_local(model, x, y, indices, config, epoch)
