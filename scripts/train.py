import os, torch, pickle, time
from torch import autocast, compile
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from models.model import SentimentClassifier
from models.loss  import SentimentWeightedLoss
from utils.data_loader import get_dataloaders
from utils.metrics     import binary_accuracy, f1_score

def train():
    # 1) HYPERPARAMS
    MODEL_NAME   = "bert-base-uncased"
    BATCH_SIZE   = 32
    LR           = 2e-5
    EPOCHS       = 5
    CKPT_PATH    = "checkpoints/train_ckpt.pkl"
    os.makedirs("checkpoints", exist_ok=True)
    
    # 2) DEVICE SELECETION
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # 3) Data
    train_dl, val_dl, _ = get_dataloaders(MODEL_NAME, BATCH_SIZE)

    # 4) Model / Opt / Sched / Loss
    model     = SentimentClassifier(MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
    )
    criterion = SentimentWeightedLoss().to(DEVICE)

    # 5) Resume logic
    start_epoch = 0
    TRAIN_START = time.time()
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    if os.path.exists(CKPT_PATH):
        print(f"⏳ Loading checkpoint from {CKPT_PATH} …")
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"✔ Resumed from epoch {start_epoch}")

    # 6) Training loop
    best_val_f1 = 0.0 # Initialize validation

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")
        # ——— Training ———
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc="  Training", leave=False)
        for input_ids, attn_mask, labels in pbar:
            input_ids, attn_mask, labels = (
                input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
            )
            optimizer.zero_grad()

            if DEVICE == "mps":
                with autocast(device_type="mps"):
                    logits = model(input_ids, attn_mask)
                    loss = criterion(logits, labels, attn_mask)
            else:
                logits = model(input_ids, attn_mask)
                loss = criterion(logits, labels, attn_mask)
         
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss / len(train_dl)
        history["train_loss"].append(avg_train)
        print(f"  ✔ Train loss: {avg_train:.4f}")

        # ——— Validation ———
        model.eval()
        all_logits, all_labels = [], []
        pbar = tqdm(val_dl, desc="  Validating", leave=False)
        with torch.no_grad():
            for input_ids, attn_mask, labels in pbar:
                input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
                logits = model(input_ids, attn_mask)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        preds = (torch.sigmoid(all_logits) >= 0.5).long()

        acc = binary_accuracy(preds, all_labels)
        f1  = f1_score(preds, all_labels)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        print(f"  🔍 Val  acc: {acc:.4f} | f1: {f1:.4f}")

        # ——— Save best‐model.pt on improvement ———
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), "checkpoints/best-model.pt")
            print(f"  ⭐ New best model saved (f1: {f1:.4f})")

        # ——— Epoch checkpoint (resume data) ———
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "sched_state" : scheduler.state_dict(),
            "history"     : history,
            "best_val_f1" : best_val_f1
        }
        torch.save(ckpt, CKPT_PATH)
        print(f"  💾 Checkpoint saved at epoch {epoch+1}")

        with open("checkpoints/history.pkl", "wb") as f:
            pickle.dump(history, f)
        print("✔ Saved training history to checkpoints/history.pkl")

         # ─── record epoch duration ───
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        history.setdefault("epoch_time", []).append(epoch_time)
        print(f"  ⏱ Epoch time: {epoch_time:.1f}s")

    print("\n Training complete.")


if __name__ == "__main__":
    train()
