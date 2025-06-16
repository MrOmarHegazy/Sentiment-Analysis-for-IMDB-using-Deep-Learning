import os
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.model import SentimentClassifier
from utils.data_loader import IMDBDataset
from utils.metrics import binary_accuracy, f1_score

RESULTS_PATH = "checkpoints/test_results.pkl"
CKPT_MODEL   = "checkpoints/train_ckpt.pkl"

def evaluate():
    # 1) Try load cached test results
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as f:
            metrics = pickle.load(f)
        print("✔ Loaded cached test results:", metrics)
        return

    # 2) Device selection (MPS → CUDA → CPU)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f">>> Evaluating on device: {DEVICE}")

    # 3) Load model checkpoint
    ckpt = torch.load(CKPT_MODEL, map_location=DEVICE)
    model = SentimentClassifier().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 4) Prepare data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_ds = IMDBDataset("test", tokenizer)
    test_dl = DataLoader(
        test_ds,
        batch_size=32,
        num_workers=4,
        pin_memory=(DEVICE.type != "cpu"),
    )

    # 5) Run inference with a progress bar
    all_logits, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in tqdm(
            test_dl, desc="🔍 Testing", leave=False
        ):
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)

            logits = model(input_ids, attn_mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    # 6) Compute metrics
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = (torch.sigmoid(all_logits) >= 0.5).long()

    acc = binary_accuracy(preds, all_labels)
    f1  = f1_score(preds, all_labels)
    metrics = {"test_acc": acc, "test_f1": f1}

    # 7) Cache & report
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(metrics, f)
    print("💾 Saved test results:", metrics)


if __name__ == "__main__":
    evaluate()
