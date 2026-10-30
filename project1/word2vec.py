"""
Each hypothesis is scored separately with the same classifier on [avg(O1); avg(O2); avg(h)]
the two scores are compared with cross-entropy on targets (0 = h1, 1 = h2)
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

from src.data import load_ds
from src.embeddings import load_word2vec
from src.predictions import log_predictions
from src.preprocessing import PAD, SPECIAL_TOKENS, UNK, build_embedding_matrix, build_vocab, preprocess, tokens_to_ids

WANDB_ENTITY = "mara-eckart-nlp"
WANDB_PROJECT = "art-p1"
MODEL_NAME = "word2vec-mlp"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "project1" / MODEL_NAME

FIELDS = {"o1": "observation_1", "o2": "observation_2", "h1": "hypothesis_1", "h2": "hypothesis_2"}
PAD_ID = SPECIAL_TOKENS.index(PAD)  # 0
UNK_ID = SPECIAL_TOKENS.index(UNK)  # 1


def parse_args() -> argparse.Namespace:
    """Hyperparameters"""
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--num_workers", type=int, default=0,)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test", action="store_true")
    return p.parse_args()


class ARTDataset(Dataset):
    """one sample = token ids of O1, O2, h1, h2 + target"""

    def __init__(self, tokenized: dict[str, list[list[str]]], targets: list[int], vocab: dict[str, int]):
        self.samples = []
        for i, target in enumerate(targets):
            sample = {key: tokens_to_ids(tokenized[column][i], vocab) for key, column in FIELDS.items()}
            sample["target"] = target
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        return self.samples[i]


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """pad each sentence in the batch to the same length (max of the batch)"""
    out = {}
    for key in FIELDS:
        sentences = [torch.tensor(sample[key], dtype=torch.long) for sample in batch]
        out[key] = pad_sequence(sentences, batch_first=True, padding_value=PAD_ID)  # [B, L_max]
    out["target"] = torch.tensor([sample["target"] for sample in batch], dtype=torch.long)  # [B]
    return out


class ARTDataModule(L.LightningDataModule):
    """loading data -> preprocessing -> build vocab + embedding matrix from train -> datasets + dataloaders"""

    def __init__(self, batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.datasets = {}

    def setup(self, stage: str):
        if self.datasets: 
            return
        nltk.download("punkt_tab", quiet=True)
        self.ds = load_ds()  #raw text
        tokenized = {
            split: {column: [preprocess(text) for text in self.ds[split][column]] for column in FIELDS.values()}
            for split in self.ds
        }
        wv = load_word2vec()
        train_tokens = [tokens for column in FIELDS.values() for tokens in tokenized["train"][column]]
        self.vocab = build_vocab(train_tokens, wv)  # train only: val/test words not in train -> <unk>
        self.embedding_matrix = build_embedding_matrix(self.vocab, wv)  # [vocab size, 300], passed to the model
        for split in self.ds:
            self.datasets[split] = ARTDataset(tokenized[split], self.ds[split]["target"], self.vocab)

    def _dataloader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._dataloader("validation")

    def test_dataloader(self):
        return self._dataloader("test")


class Classifier(nn.Module):
    """2-layer classifier with ReLU"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim] -> [B, output_dim] 
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)


class AvgClassifier(L.LightningModule):
    def __init__(self, embedding_matrix, hidden_dim: int, dropout: float, lr: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_matrix"]) 
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix), freeze=True, padding_idx=PAD_ID  # frozen: train only the classifier
        )
        embedding_dim = embedding_matrix.shape[1]  # 300
        self.classifier = Classifier(3 * embedding_dim, hidden_dim, 1, dropout)  # [o1; o2; h] -> 1 score
        self.loss_fn = nn.CrossEntropyLoss()

    def sentence_vector(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, L] ids -> [B, 300]: average over the words of each sentence (along L), ignoring <pad> and <unk>."""
        mask = ((ids != PAD_ID) & (ids != UNK_ID)).unsqueeze(-1).float()  # [B, L, 1], 1 = real word
        summed = (self.embedding(ids) * mask).sum(dim=1)  # [B, 300]
        n_words = mask.sum(dim=1).clamp(min=1)  
        return summed / n_words

    def forward(self, batch: dict) -> torch.Tensor:
        o1 = self.sentence_vector(batch["o1"])
        o2 = self.sentence_vector(batch["o2"])
        scores = []
        for h in ("h1", "h2"): 
            x = torch.cat([o1, o2, self.sentence_vector(batch[h])], dim=-1)  # [B, 900]
            scores.append(self.classifier(x))  # [B, 1]
        return torch.cat(scores, dim=-1)  # [B, 2] logits

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
        params = [p for p in self.parameters() if p.requires_grad]  # only the classifier
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def train(args: argparse.Namespace):
    L.seed_everything(args.seed)  

    dm = ARTDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=args.accelerator == "gpu")
    dm.setup("fit") 
    model = AvgClassifier(dm.embedding_matrix, hidden_dim=args.hidden_dim, dropout=args.dropout,
                          lr=args.lr, weight_decay=args.weight_decay)

    run_name = f"w2v-mlp_lr{args.lr:g}_h{args.hidden_dim}_d{args.dropout:g}_p{args.patience}"
    logger = WandbLogger(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=MODEL_NAME, name=run_name)
    logger.log_hyperparams(vars(args) | {"vocab_size": len(dm.vocab)})

    # early stopping on val/loss, keep the epoch with the best val/acc
    checkpoint = ModelCheckpoint(dirpath=CHECKPOINT_ROOT / logger.experiment.id, filename="best",
                                 monitor="val/acc", mode="max", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)

    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator=args.accelerator, devices=1, logger=logger,
                        callbacks=[checkpoint, early_stopping], deterministic=True)
    trainer.fit(model, datamodule=dm)
    logger.experiment.summary["best_val_acc"] = checkpoint.best_model_score.item() 
    logger.experiment.summary["best_val_loss"] = early_stopping.best_score.item()  
    log_predictions(trainer, model, dm.val_dataloader(), dm.ds["validation"], logger, key="val/predictions")

    if args.test:  
        trainer.test(model, datamodule=dm, ckpt_path="best")
        log_predictions(trainer, model, dm.test_dataloader(), dm.ds["test"], logger, key="test/predictions")
    wandb.finish()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
