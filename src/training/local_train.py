import torch
from torch import nn

from src.data.loaders import batches
from src.utils.seed import stream_seed


def train_local(model, x, y, indices, config, round_index, client_id=0):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0, weight_decay=0)
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = torch.zeros((), device=x.device)
    seen = steps = 0
    epochs = 1 if config["mode"] == "centralized" else config["local_epochs"]
    for epoch in range(epochs):
        seed = stream_seed(config["seed"], "loader", round_index, client_id, epoch)
        for bx, by in batches(x, y, indices, config["batch_size"], seed):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(by)
            seen += len(by)
            steps += 1
    value = float(loss_sum.cpu())
    if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError("Non-finite training loss")
    return {"n_samples": len(indices), "loss_sum": value, "examples_seen": seen, "optimizer_steps": steps}
