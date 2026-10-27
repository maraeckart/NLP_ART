"""Project 1, method 1: averaged word2vec embeddings + MLP classifier.

Each hypothesis is scored separately with the same MLP on [avg(O1); avg(O2); avg(h)];
the two scores are compared with cross-entropy against `target` (0 = h1, 1 = h2).

Run from the repo root:
    python project1/word2vec.py                      # train + validate
    python project1/word2vec.py --lr 3e-4 --test     # also evaluate the best checkpoint on test
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning as L
import nltk
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data import TEXT_COLUMNS, load_ds
from src.embeddings import load_word2vec
from src.predictions import log_predictions
from src.preprocessing import PAD, UNK, build_embedding_matrix, build_vocab, preprocess, tokens_to_ids

WANDB_ENTITY = "mara-eckart-nlp"
WANDB_PROJECT = "art-p1"
MODEL_NAME = "word2vec-mlp"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "project1" / MODEL_NAME

FIELDS = {"o1": "observation_1", "o2": "observation_2", "h1": "hypothesis_1", "h2": "hypothesis_2"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--finetune_embeddings", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test", action="store_true", help="evaluate best checkpoint on the test split")
    return p.parse_args()



class ARTDataset(Dataset):
    def __init__(self, tokenized: dict[str, list[list[str]]], targets: list[int], vocab: dict[str, int]):
        self.ids = {key: [tokens_to_ids(t, vocab) for t in tokenized[col]] for key, col in FIELDS.items()}
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> dict:
        return {key: self.ids[key][i] for key in FIELDS} | {"target": self.targets[i]}


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Pad each sentence field to the longest sentence of that field in the batch."""
        out = {
            key: pad_sequence([torch.tensor(ex[key], dtype=torch.long) for ex in batch],
                              batch_first=True, padding_value=pad_id)
            for key in FIELDS
        }
        out["target"] = torch.tensor([ex["target"] for ex in batch])
        return out
    return collate


class AvgMLP(L.LightningModule):
    def __init__(self, embedding_matrix, pad_id: int, unk_id: int, hidden_dim: int, dropout: float,
                 lr: float, weight_decay: float, finetune_embeddings: bool):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_matrix"])
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix), freeze=not finetune_embeddings, padding_idx=pad_id
        )
        dim = embedding_matrix.shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(3 * dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # one plausibility score per hypothesis
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def sentence_vector(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, L] token ids -> [B, D] mean of word vectors, ignoring <pad> and <unk>."""
        mask = ((ids != self.hparams.pad_id) & (ids != self.hparams.unk_id)).unsqueeze(-1).float()
        summed = (self.embedding(ids) * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1)  # all-<unk> sentence -> zero vector, not NaN

    def forward(self, batch: dict) -> torch.Tensor:
        o1, o2 = self.sentence_vector(batch["o1"]), self.sentence_vector(batch["o2"])
        scores = [self.classifier(torch.cat([o1, o2, self.sentence_vector(batch[h])], dim=-1)) for h in ("h1", "h2")]
        return torch.cat(scores, dim=-1)  # [B, 2] logits

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        logits = self(batch)
        loss = self.loss_fn(logits, batch["target"])
        acc = (logits.argmax(dim=-1) == batch["target"]).float().mean()
        bs = len(batch["target"])
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
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
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


# ---------- 4. Training ----------

def main():
    args = parse_args()
    L.seed_everything(args.seed)
    nltk.download("punkt_tab", quiet=True)

    ds = load_ds()
    wv = load_word2vec()
    tokenized = {split: {col: [preprocess(t) for t in ds[split][col]] for col in TEXT_COLUMNS} for split in ds}
    vocab = build_vocab([toks for col in TEXT_COLUMNS for toks in tokenized["train"][col]], wv)
    embedding_matrix = build_embedding_matrix(vocab, wv)

    collate = make_collate(vocab[PAD])
    loaders = {
        split: DataLoader(ARTDataset(tokenized[split], ds[split]["target"], vocab), batch_size=args.batch_size,
                          shuffle=(split == "train"), collate_fn=collate)
        for split in ds
    }

    run_name = (f"w2v-mlp_h{args.hidden_dim}_lr{args.lr:g}_"
                f"{'finetune' if args.finetune_embeddings else 'frozen'}_s{args.seed}")
    logger = WandbLogger(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=MODEL_NAME, name=run_name)
    logger.log_hyperparams(vars(args) | {"vocab_size": len(vocab)})

    checkpoint = ModelCheckpoint(dirpath=CHECKPOINT_ROOT / logger.experiment.id, filename="best",
                                 monitor="val/loss", mode="min")
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)

    model = AvgMLP(embedding_matrix, pad_id=vocab[PAD], unk_id=vocab[UNK], hidden_dim=args.hidden_dim,
                   dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
                   finetune_embeddings=args.finetune_embeddings)
    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="cpu", logger=logger,
                        callbacks=[checkpoint, early_stopping], deterministic=True)
    trainer.fit(model, loaders["train"], loaders["validation"])
    log_predictions(trainer, model, loaders["validation"], ds["validation"], logger, key="val/predictions")

    if args.test:
        trainer.test(model, loaders["test"], ckpt_path="best")
        log_predictions(trainer, model, loaders["test"], ds["test"], logger, key="test/predictions")
    wandb.finish()


if __name__ == "__main__":
    main()
