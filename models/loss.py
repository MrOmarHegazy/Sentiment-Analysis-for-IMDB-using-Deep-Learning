import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentWeightedLoss(nn.Module):
    """
    Custom loss: BCEWithLogits but penalizes
    - long reviews more heavily (length weighting)
    - highly confident wrong predictions more.
    """
    def __init__(self, length_weight=0.5, confidence_weight=2.0):
        super().__init__()
        self.base = nn.BCEWithLogitsLoss(reduction="none")
        self.length_weight = length_weight
        self.conf_weight = confidence_weight

    def forward(self, logits, targets, attention_mask):
        """
        logits: [B], targets: [B], attention_mask:[B, L]
        """
        batch_size = logits.size(0)
        base_loss = self.base(logits, targets)

        # 1) length-based weight: longer reviews get higher weight
        lengths = attention_mask.sum(dim=1).float()  # number of tokens
        len_w = 1 + self.length_weight * (lengths / lengths.mean() - 1)

        # 2) confidence-based weight: wrong + high prob → bigger penalty
        probs = torch.sigmoid(logits)
        conf_margin = torch.abs(probs - targets)  # high when wrong/confident
        conf_w = 1 + self.conf_weight * conf_margin

        weights = len_w * conf_w
        weighted_loss = (base_loss * weights).mean()
        return weighted_loss
