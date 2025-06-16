import torch
from sklearn.metrics import f1_score as sk_f1

def binary_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def f1_score(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return sk_f1(labels.numpy(), preds.numpy())
