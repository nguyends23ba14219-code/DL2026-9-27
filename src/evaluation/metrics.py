import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from src.data.loaders import batches


def classification_metrics(labels, predictions, loss):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    p, r, f, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(10),
        zero_division=0,
    )
    cm = confusion_matrix(labels, predictions, labels=np.arange(10))
    normalized = np.divide(
        cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True) != 0
    )
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "loss": float(loss),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "per_class": [
            {"class": c, "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]), "support": int(support[c])}
            for c in range(10)
        ],
        "n_samples": len(labels),
    }


@torch.inference_mode()
def evaluate(model, x, y, indices=None, batch_size=512):
    model.eval()
    if indices is None:
        indices = np.arange(len(y))
    losses = torch.zeros((), device=x.device)
    labels, predictions = [], []
    for bx, by in batches(x, y, indices, batch_size):
        logits = model(bx)
        losses += nn.functional.cross_entropy(logits, by, reduction="sum")
        labels.append(by)
        predictions.append(logits.argmax(1))
    true = torch.cat(labels).cpu().numpy()
    pred = torch.cat(predictions).cpu().numpy()
    return classification_metrics(true, pred, float(losses.cpu()) / len(indices)), true, pred
