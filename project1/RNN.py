"""Project 1, method 2: word2vec embeddings + LSTM encoder + classifier.

Input formats (--input_format):
    joint:    one sequence  O1 <sep> h1 <sep> h2 <sep> O2, final hidden state -> 2 logits (0 = h1, 1 = h2)
    separate: two sequences O1 <sep> h1 <sep> O2 and O1 <sep> h2 <sep> O2, same LSTM + classifier
              -> one score each, the 2 scores are the logits
Cross-entropy against `target` in both cases.

Run from the repo root:
    python project1/RNN.py                           # train + validate
    python project1/RNN.py --bidirectional --test    # also evaluate the best checkpoint on test
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data import TEXT_COLUMNS, load_ds
from src.embeddings import load_word2vec
from src.predictions import log_predictions
from src.preprocessing import PAD, SEP, build_embedding_matrix, build_vocab, preprocess, tokens_to_ids

WANDB_ENTITY = "mara-eckart-nlp"
WANDB_PROJECT = "art-p1"
MODEL_NAME = "lstm"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "project1" / MODEL_NAME

SEQUENCE_ORDERS = {
    "joint": [["observation_1", "hypothesis_1", "hypothesis_2", "observation_2"]],  # O1 <sep> h1 <sep> h2 <sep> O2
    "separate": [["observation_1", "hypothesis_1", "observation_2"],  # O1 <sep> h1 <sep> O2
                 ["observation_1", "hypothesis_2", "observation_2"]],  # O1 <sep> h2 <sep> O2
    "h_last": [["observation_1", "observation_2", "hypothesis_1"],  # O1 <sep> O2 <sep> h1
               ["observation_1", "observation_2", "hypothesis_2"]],  # O1 <sep> O2 <sep> h2
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_format", choices=list(SEQUENCE_ORDERS), default="joint")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--finetune_embeddings", action="store_true")
    p.add_argument("--keep_punct", action="store_true", help="keep punctuation tokens (removed by default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test", action="store_true", help="evaluate best checkpoint on the test split")
    return p.parse_args()


class ARTSequenceDataset(Dataset):
    def __init__(self, tokenized: dict[str, list[list[str]]], targets: list[int], vocab: dict[str, int],
                 input_format: str):
        sep_id = vocab[SEP]
        self.sequences = []  # per example: list of 1 (joint) or 2 (separate) id sequences
        for i in range(len(targets)):
            example = []
            for order in SEQUENCE_ORDERS[input_format]:
                ids = []
                for j, col in enumerate(order):
                    if j > 0:
                        ids.append(sep_id)
                    ids += tokens_to_ids(tokenized[col][i], vocab)
                example.append(ids)
            self.sequences.append(example)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> dict:
        return {"ids": self.sequences[i], "target": self.targets[i]}


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        """Pad each sequence slot to its longest sequence in the batch; keep the true lengths for packing."""
        ids, lengths = [], []
        for k in range(len(batch[0]["ids"])):  # 1 slot (joint) or 2 slots (separate)
            seqs = [torch.tensor(ex["ids"][k], dtype=torch.long) for ex in batch]
            ids.append(pad_sequence(seqs, batch_first=True, padding_value=pad_id))
            lengths.append(torch.tensor([len(s) for s in seqs]))
        return {"ids": ids, "lengths": lengths, "target": torch.tensor([ex["target"] for ex in batch])}
    return collate



class LSTMClassifier(L.LightningModule):
    def __init__(self, embedding_matrix, pad_id: int, input_format: str, hidden_dim: int, num_layers: int,
                 bidirectional: bool, dropout: float, lr: float, weight_decay: float, finetune_embeddings: bool):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_matrix"])
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix), freeze=not finetune_embeddings, padding_idx=pad_id
        )
        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout between LSTM layers only
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        # joint: 2 logits (h1, h2) from one sequence; separate: 1 score per sequence
        n_outputs = 2 if input_format == "joint" else 1
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, n_outputs))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, ids: list[torch.Tensor], lengths: list[torch.Tensor]) -> torch.Tensor:
        # joint: [B, 2]; separate: [B, 1] score for h1 and [B, 1] score for h2 -> concatenated to [B, 2]
        return torch.cat([self.classifier(self.encode(i, l)) for i, l in zip(ids, lengths)], dim=-1)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """[B, L] token ids -> [B, H] final hidden state of the last LSTM layer."""
        embedded = self.embedding(ids)  # [B, L, D]
        # packing: the LSTM stops at each sequence's true length, so <pad> never changes the final state
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: [num_layers * num_directions, B, H]
        if self.hparams.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # last layer, forward + backward
        else:
            final = h_n[-1]  # last layer
        return final

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        logits = self(batch["ids"], batch["lengths"])
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
        return torch.softmax(self(batch["ids"], batch["lengths"]), dim=-1)  # [B, 2] probabilities for h1, h2

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
    tokenized = {
        split: {col: [preprocess(t, remove_punct=not args.keep_punct) for t in ds[split][col]] for col in TEXT_COLUMNS}
        for split in ds
    }
    vocab = build_vocab([toks for col in TEXT_COLUMNS for toks in tokenized["train"][col]], wv)
    embedding_matrix = build_embedding_matrix(vocab, wv)

    collate = make_collate(vocab[PAD])
    loaders = {
        split: DataLoader(ARTSequenceDataset(tokenized[split], ds[split]["target"], vocab, args.input_format),
                          batch_size=args.batch_size, shuffle=(split == "train"), collate_fn=collate)
        for split in ds
    }

    run_name = (f"{'bi' if args.bidirectional else ''}lstm-{args.input_format}-{args.num_layers}l_h{args.hidden_dim}_"
                f"lr{args.lr:g}_{'finetune' if args.finetune_embeddings else 'frozen'}_s{args.seed}")
    logger = WandbLogger(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=MODEL_NAME, name=run_name)
    logger.log_hyperparams(vars(args) | {"vocab_size": len(vocab)})

    checkpoint = ModelCheckpoint(dirpath=CHECKPOINT_ROOT / logger.experiment.id, filename="best",
                                 monitor="val/loss", mode="min")
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)

    model = LSTMClassifier(embedding_matrix, pad_id=vocab[PAD], input_format=args.input_format,
                           hidden_dim=args.hidden_dim,
                           num_layers=args.num_layers, bidirectional=args.bidirectional, dropout=args.dropout,
                           lr=args.lr, weight_decay=args.weight_decay, finetune_embeddings=args.finetune_embeddings)
    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="cpu", logger=logger,
                        callbacks=[checkpoint, early_stopping], gradient_clip_val=args.grad_clip,
                        deterministic=True)
    trainer.fit(model, loaders["train"], loaders["validation"])
    log_predictions(trainer, model, loaders["validation"], ds["validation"], logger, key="val/predictions")

    if args.test:
        trainer.test(model, loaders["test"], ckpt_path="best")
        log_predictions(trainer, model, loaders["test"], ds["test"], logger, key="test/predictions")
    wandb.finish()


if __name__ == "__main__":
    main()
