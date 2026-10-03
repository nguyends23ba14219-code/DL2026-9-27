from src.training.local_train import train_local


def centralized_epoch(model, x, y, indices, config, epoch):
    return train_local(model, x, y, indices, config, epoch)
