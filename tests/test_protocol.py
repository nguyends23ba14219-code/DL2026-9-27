import copy
import json

import numpy as np
import pytest
import torch

from src.data.partition import partition, stratified_split
from src.evaluation.metrics import classification_metrics
from src.federated.client import client_update
from src.federated.fedavg import fedavg
from src.models.cnn import SmallCNN
from src.runner import run
from src.training.local_train import train_local
from src.utils.seed import seed_all, snapshot, state_hash


def test_stratified_split_no_leakage():
    labels = np.repeat(np.arange(10), 6000)
    train, val = stratified_split(labels)
    assert len(train) == 54000 and len(val) == 6000
    assert not set(train) & set(val)
    assert len(set(train) | set(val)) == 60000
    assert (np.bincount(labels[train]) == 5400).all()
    assert np.array_equal(train, stratified_split(labels)[0])


@pytest.mark.parametrize("severity,dominant,other", [(0, 540, 540), (0.5, 1620, 270), (0.9, 2484, 54)])
def test_exact_partition(severity, dominant, other):
    labels = np.repeat(np.arange(10), 5400)
    clients, counts, manifest = partition(np.arange(54000), labels, severity, 42)
    assert len(np.unique(np.concatenate(clients))) == 54000
    assert (counts.sum(0) == 5400).all() and (counts.sum(1) == 5400).all()
    for row in counts:
        assert sorted(row) == sorted([dominant] * 2 + [other] * 8)
    assert np.allclose(manifest["total_variation"], 0.8 * severity)
    _, _, repeated = partition(np.arange(54000), labels, severity, 42)
    assert repeated == manifest


def test_local_validation_quota_and_pairing():
    labels = np.repeat(np.arange(10), 600)
    _, counts, manifest = partition(np.arange(6000), labels, 0.9, 42)
    assert sorted(counts[0]) == [6] * 8 + [276] * 2
    train_labels = np.repeat(np.arange(10), 5400)
    _, _, train = partition(np.arange(54000), train_labels, 0.9, 42)
    assert train["class_permutation"] == manifest["class_permutation"]


def test_fedavg_weighted_and_no_aliasing():
    a, b = {"w": torch.tensor([1.0, 3.0])}, {"w": torch.tensor([5.0, 7.0])}
    result = fedavg([a, b], [1, 3])
    assert torch.equal(result["w"], torch.tensor([4.0, 6.0]))
    result["w"].zero_()
    assert torch.equal(a["w"], torch.tensor([1.0, 3.0]))
    with pytest.raises(ValueError):
        fedavg([a], [0])


def test_metrics_known_answer_macro_uses_all_ten_labels():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 0.7)
    assert result["accuracy"] == 0.75
    assert result["macro_precision"] == pytest.approx((1 + 2 / 3) / 10)
    assert result["macro_recall"] == pytest.approx(1.5 / 10)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 10)
    assert result["confusion_matrix"][0][:2] == [1, 1]


def test_cnn_shape_and_parameter_count():
    model = SmallCNN()
    assert sum(p.numel() for p in model.parameters()) == 105866
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_each_client_starts_from_broadcast():
    torch.set_num_threads(1)
    seed_all(42)
    model = SmallCNN()
    original = snapshot(model)
    x, y = torch.randn(20, 1, 28, 28), torch.arange(20) % 10
    cfg = {"seed": 42, "mode": "federated", "batch_size": 10, "local_epochs": 1, "learning_rate": 0.01}
    first, _ = client_update(model, original, x, y, np.arange(20), cfg, 1, 0)
    second, _ = client_update(model, original, x, y, np.arange(20), cfg, 1, 0)
    assert state_hash(first) == state_hash(second)
    assert state_hash(original) != state_hash(first)


def fake_data():
    generator = torch.Generator().manual_seed(99)
    labels = np.repeat(np.arange(10), 200)
    train = np.concatenate([np.arange(c * 200, c * 200 + 100) for c in range(10)])
    val = np.concatenate([np.arange(c * 200 + 100, (c + 1) * 200) for c in range(10)])
    return {
        "x": torch.randn(2000, 1, 28, 28, generator=generator),
        "y": torch.tensor(labels),
        "test_x": torch.randn(100, 1, 28, 28, generator=generator),
        "test_y": torch.arange(100) % 10,
        "train_idx": train,
        "val_idx": val,
        "labels": labels,
    }


@pytest.mark.parametrize("mode,local_epochs", [("centralized", 1), ("federated", 1), ("federated", 3)])
def test_resume_matches_uninterrupted_and_rejects_config_change(tmp_path, monkeypatch, mode, local_epochs):
    import src.runner as runner

    cfg = {
        "seed": 42,
        "split_seed": 2026,
        "mode": mode,
        "setting": "test",
        "lambda": 0.9,
        "rounds": 2,
        "local_epochs": local_epochs,
        "batch_size": 64,
        "learning_rate": 0.01,
        "device": "cpu",
        "cpu_threads": 1,
        "output_dir": str(tmp_path / "full"),
    }
    data = fake_data()
    full = run(cfg, data=data)
    cfg2 = {**cfg, "output_dir": str(tmp_path / "resumed")}
    step_name = "centralized_epoch" if mode == "centralized" else "federated_round"
    original = getattr(runner, step_name)

    def interrupted(*args):
        if args[-1] == 2:
            raise KeyboardInterrupt()
        return original(*args)

    monkeypatch.setattr(runner, step_name, interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(cfg2, data=data)
    monkeypatch.setattr(runner, step_name, original)
    resumed = run(cfg2, resume=True, data=data)
    final1 = torch.load(full / "checkpoints/round_002.pt", weights_only=True)
    final2 = torch.load(resumed / "checkpoints/round_002.pt", weights_only=True)
    assert state_hash(final1["state"]) == state_hash(final2["state"])
    assert (
        json.loads((full / "final_metrics.json").read_text())["final_test"]
        == json.loads((resumed / "final_metrics.json").read_text())["final_test"]
    )
    changed = copy.deepcopy(cfg2)
    changed["learning_rate"] = 0.02
    with pytest.raises(ValueError):
        run(changed, resume=True, data=data)


@pytest.mark.parametrize("setting", ["centralized", "iid"])
def test_test_evaluation_follows_training_and_cannot_select_checkpoint(tmp_path, monkeypatch, setting):
    import src.evaluation.post_training as post_training
    import src.runner as runner

    config = runner.load_config(f"configs/{setting}.yaml", seed=42)
    config.update(rounds=2, device="cpu", output_dir=str(tmp_path))
    events = []
    validation_scores = iter([0.2, 0.8, 0.8])  # Rounds 1 and 2 tie; the earliest must win.
    test_scores = iter([0.95, 0.4, 0.6])  # Test prefers round 0; selection must ignore this.
    step_name = "centralized_epoch" if setting == "centralized" else "federated_round"
    original_step = getattr(runner, step_name)
    original_evaluate = runner.evaluate
    trained_states = {}

    def train_step(*args):
        stats = original_step(*args)
        trained_states[args[-1]] = state_hash(snapshot(args[0]))
        events.append("train")
        return stats

    def validation(*args, **kwargs):
        metrics, labels, predictions = original_evaluate(*args, **kwargs)
        metrics["accuracy"] = next(validation_scores)
        events.append("validation")
        return metrics, labels, predictions

    def after_training(model, x, y, indices=None):
        metrics, labels, predictions = original_evaluate(model, x, y, indices)
        if indices is None:
            metrics["accuracy"] = next(test_scores)
            events.append("test")
        else:
            # Local subsets must use the final global model, not the validation winner.
            assert state_hash(snapshot(model)) == trained_states[2]
            events.append("local_validation")
        return metrics, labels, predictions

    monkeypatch.setattr(runner, step_name, train_step)
    monkeypatch.setattr(runner, "evaluate", validation)
    monkeypatch.setattr(post_training, "evaluate", after_training)
    path = run(config, data=fake_data())
    result = json.loads((path / "final_metrics.json").read_text())
    assert events[:8] == ["validation", "train", "validation", "train", "validation", "test", "test", "test"]
    assert events[8:] == (["local_validation"] * 10 if setting == "iid" else [])
    assert result["best_validation_round"] == 1
    assert result["best_validation_test"]["accuracy"] == 0.4
    assert result["final_test"]["accuracy"] == 0.6


def test_can_overfit_tiny_dataset():
    torch.set_num_threads(1)
    seed_all(4)
    model = SmallCNN()
    x = torch.zeros(20, 1, 28, 28)
    x[10:, :, :, 14:] = 1
    y = torch.tensor([0] * 10 + [1] * 10)
    cfg = {"seed": 4, "mode": "centralized", "batch_size": 20, "learning_rate": 0.1}
    for epoch in range(40):
        train_local(model, x, y, np.arange(20), cfg, epoch)
    assert (model(x).argmax(1) == y).float().mean() > 0.95


def test_evaluation_weights_partial_last_batch():
    from src.evaluation.metrics import evaluate

    logits = torch.arange(50, dtype=torch.float32).reshape(5, 10)
    labels = torch.tensor([9, 9, 9, 9, 0])
    metrics, _, _ = evaluate(torch.nn.Identity(), logits, labels, batch_size=4)
    expected = torch.nn.functional.cross_entropy(logits, labels).item()
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["accuracy"] == 0.8


def test_same_run_has_exclusive_writer_lock(tmp_path):
    from filelock import FileLock

    cfg = {"output_dir": str(tmp_path), "setting": "locked", "seed": 42}
    output = tmp_path / "runs/locked/seed_42"
    output.mkdir(parents=True)
    with FileLock(output / ".run.lock"):
        with pytest.raises(RuntimeError, match="already writing this run"):
            run(cfg)
