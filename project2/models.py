"""RoBERTa encoder (random or pretrained weights)"""
import lightning as L
import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from src.models import Classifier

class TransformerClassifier(L.LightningModule):
    def __init__(self, model_name: str, init: str, input_format: str, clf_hidden_dim: int, dropout: float,
                 lr: float, weight_decay: float, warmup_ratio: float):
        super().__init__()
        self.save_hyperparameters()
        # same config (layers, hidden size, heads, vocab, dropout) for both: only the starting weights differ
        config = AutoConfig.from_pretrained(model_name)
        if init == "pretrained":
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        elif init == "random":
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)  # HF init: normal(0, 0.02)
        else:
            raise ValueError(init)
        n_outputs = 2 if input_format == "joint" else 1  # joint: 1 sequence -> 2 logits, else 1 score per sequence
        self.classifier = Classifier(config.hidden_size, clf_hidden_dim, n_outputs, dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch: dict) -> torch.Tensor:
        B, n_seq, L = batch["input_ids"].shape
        out = self.encoder(
            input_ids=batch["input_ids"].view(B * n_seq, L),  # all sequences through the encoder at once
            attention_mask=batch["attention_mask"].view(B * n_seq, L),
        )
        cls = out.last_hidden_state[:, 0]  # vector of the <s> token: [B * n_seq, H]
        return self.classifier(cls).view(B, 2)  # [B, 2] logits (h1, h2)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        logits = self(batch)
        loss = self.loss_fn(logits, batch["target"])
        acc = (logits.argmax(dim=-1) == batch["target"]).float().mean()
        batch_size = len(batch["target"])
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        return torch.softmax(self(batch), dim=-1)  # [B, 2] probabilities for h1, h2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # warmup + linear decay (lecture: start = max/100, end = start/100, warmup = 10% of the steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_ratio,
            div_factor=100,
            final_div_factor=100,
            anneal_strategy="linear",
            cycle_momentum=False,  # keep Adam's betas fixed
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
