import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dropout_prob=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        pooled  = outputs.pooler_output    # [batch, hidden]
        dropped = self.dropout(pooled)
        logits  = self.classifier(dropped).squeeze(-1)
        return logits
