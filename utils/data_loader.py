import os
import pickle, torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

CACHE_DIR = "data"

def load_or_cache_imdb(split: str):
    """
    Returns (texts, labels) for the given split, downloading only once.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"imdb_{split}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            texts, labels = pickle.load(f)
        print(f"✔ Loaded {split} split from cache ({cache_path})")
    else:
        ds = load_dataset("imdb", split=split)
        texts, labels = ds["text"], ds["label"]
        with open(cache_path, "wb") as f:
            pickle.dump((texts, labels), f)
        print(f"→ Downloaded + cached {split} split to {cache_path}")
    return texts, labels


class IMDBDataset(Dataset):
    def __init__(self, split: str, tokenizer, max_length=256):
        texts, labels = load_or_cache_imdb(split)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


def get_dataloaders(model_name, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train = IMDBDataset("train[:80%]", tokenizer)
    val   = IMDBDataset("train[80%:]", tokenizer)
    test  = IMDBDataset("test", tokenizer)

    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val,   batch_size),
        DataLoader(test,  batch_size),
    )
