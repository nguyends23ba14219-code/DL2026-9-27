import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from src.data.loaders import batches


def classification_metrics(labels, predictions, loss):
    """Report all ten classes; confusion matrix rows are true labels."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(10),
        zero_division=0,
    )
    counts = confusion_matrix(labels, predictions, labels=np.arange(10))
    row_totals = counts.sum(1, keepdims=True)
    normalized = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals != 0)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "loss": float(loss),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": counts.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "per_class": [
            {
                "class": class_id,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in range(10)
        ],
        "n_samples": len(labels),
    }


@torch.inference_mode()
def evaluate(model, x, y, indices=None, batch_size=512):
    """Inference only; weight the final partial batch by its actual sample count."""
    model.eval()
    if indices is None:
        indices = np.arange(len(y))
    loss_sum = torch.zeros((), device=x.device)
    labels, predictions = [], []
    for images, targets in batches(x, y, indices, batch_size):
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum")
        labels.append(targets)
        predictions.append(logits.argmax(1))
    true = torch.cat(labels).cpu().numpy()
    pred = torch.cat(predictions).cpu().numpy()
    return classification_metrics(true, pred, float(loss_sum.cpu()) / len(indices)), true, pred
