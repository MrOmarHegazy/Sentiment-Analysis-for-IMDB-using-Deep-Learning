import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from models.model import SentimentClassifier
from utils.data_loader import IMDBDataset
from torch.utils.data import DataLoader


def main():
    # Ensure output directory exists
    os.makedirs("plots", exist_ok=True)

    # Load training history
    with open("checkpoints/history.pkl", "rb") as f:
        history = pickle.load(f)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Plot 1: Training Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss over Epochs")
    plt.tight_layout()
    plt.savefig("plots/train_loss.png")
    plt.close()
    print("Saved plots/train_loss.png")

    # Plot 2: Validation Accuracy & F1
    plt.figure()
    plt.plot(epochs, history["val_acc"], marker="o", label="Val Accuracy")
    plt.plot(epochs, history["val_f1"], marker="o", label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Validation Accuracy and F1 over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/val_metrics.png")
    plt.close()
    print("Saved plots/val_metrics.png")

    # Plot 3: Smoothed Validation F1
    val_f1_series = pd.Series(history["val_f1"])
    smoothed = val_f1_series.ewm(alpha=0.3).mean()
    plt.figure()
    plt.plot(epochs, history["val_f1"], label="Raw Val F1")
    plt.plot(epochs, smoothed,       label="Smoothed Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Raw vs. Smoothed Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/smoothed_val_f1.png")
    plt.close()
    print("Saved plots/smoothed_val_f1.png")

    # Plot 4: Epoch Duration (if recorded)
    if "epoch_time" in history and len(history["epoch_time"]) > 0:
        time_len = len(history["epoch_time"])
        time_epochs = list(range(1, time_len + 1))
        plt.figure()
        plt.plot(time_epochs, history["epoch_time"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Time (s)")
        plt.title("Epoch Duration")
        plt.tight_layout()
        plt.savefig("plots/epoch_time.png")
        plt.close()
        print("Saved plots/epoch_time.png")

    # Device selection for evaluation
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Evaluating on device: {device}")

    # Load best model
    model = SentimentClassifier().to(device)
    model.load_state_dict(torch.load("checkpoints/best-model.pt", map_location=device))
    model.eval()

    # Prepare test data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_ds = IMDBDataset("test", tokenizer)
    test_dl = DataLoader(
        test_ds,
        batch_size=32,
        num_workers=4,
        pin_memory=(device.type != "cpu"),
    )

    # Collect probabilities and labels
    all_probs, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, lbl in tqdm(test_dl, desc="Inference"):
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            logits = model(input_ids, attn_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(lbl.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Plot 5: ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/roc_curve.png")
    plt.close()
    print("Saved plots/roc_curve.png")

    # Plot 6: Confusion Matrix
    preds = (all_probs >= 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig("plots/confusion_matrix.png")
    plt.close()
    print("Saved plots/confusion_matrix.png")

if __name__ == "__main__":
    main()
