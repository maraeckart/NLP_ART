import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset

from src.datamodule import ARTDataModule
from src.models import Classifier
from src.predictions import log_predictions
from src.preprocessing import PAD, SEP, SPECIAL_TOKENS, tokens_to_ids

WANDB_ENTITY = "mara-eckart-nlp"
WANDB_PROJECT = "art-p1"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "project1" 

NUM_LAYERS = 2

PAD_ID = SPECIAL_TOKENS.index(PAD)  # 0
SEP_ID = SPECIAL_TOKENS.index(SEP)  # 2

SEQUENCE_ORDERS = {
    "h_last": [["observation_1", "observation_2", "hypothesis_1"], 
               ["observation_1", "observation_2", "hypothesis_2"]], 
    "separate": [["observation_1", "hypothesis_1", "observation_2"],  
                 ["observation_1", "hypothesis_2", "observation_2"]], 
    "joint": [["observation_1", "hypothesis_1", "hypothesis_2", "observation_2"]], 
    "h_only": [["hypothesis_1","hypothesis_2"]],
}


def parse_args() -> argparse.Namespace:
    """Hyperparameters"""
    p = argparse.ArgumentParser()
    p.add_argument("--input_format", default="h_last", choices=list(SEQUENCE_ORDERS))
    p.add_argument("--cell", default="lstm", choices=["lstm", "gru"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--clf_hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=3)  
    p.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test", action="store_true")
    return p.parse_args()


def build_sequence(tokenized: dict[str, list[list[str]]], columns: list[str], i: int, vocab: dict[str, int]) -> list[int]:
    # add seprator token within a sequence
    ids = []
    for j, column in enumerate(columns):
        if j > 0:
            ids.append(SEP_ID)
        ids += tokens_to_ids(tokenized[column][i], vocab)
    return ids


class ARTSequenceDataset(Dataset):
    def __init__(self, tokenized: dict[str, list[list[str]]], targets: list[int], vocab: dict[str, int],
                 input_format: str):
        self.samples = []
        for i, target in enumerate(targets):
            sequences = [build_sequence(tokenized, columns, i, vocab) for columns in SEQUENCE_ORDERS[input_format]]
            self.samples.append({"sequences": sequences, "target": target})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        return self.samples[i]


def collate_fn(batch: list[dict]) -> dict:
    """Pad each sequence in the batch to the same length (max of the batch)"""
    # padding on the longest sequence of each slot in the batch
    ids, lengths = [], []
    for k in range(len(batch[0]["sequences"])):  # 2 slots (h_last/separate) or 1 (joint)
        sequences = [torch.tensor(sample["sequences"][k], dtype=torch.long) for sample in batch]
        ids.append(pad_sequence(sequences, batch_first=True, padding_value=PAD_ID))  # [B, L_max]
        lengths.append(torch.tensor([len(s) for s in sequences]))  # [B]
    target = torch.tensor([sample["target"] for sample in batch], dtype=torch.long)  # [B]
    return {"ids": ids, "lengths": lengths, "target": target}


class RNNClassifier(L.LightningModule):
    """frozen word2vec -> 2-layer LSTM or GRU -> final hidden state -> 2-layer classifier."""

    def __init__(self, embedding_matrix, input_format: str, hidden_dim: int, clf_hidden_dim: int, dropout: float,
                 bidirectional: bool, lr: float, weight_decay: float, cell: str = "lstm"):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_matrix"])  # matrix is saved as weights, not as config
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix), freeze=True, padding_idx=PAD_ID  # frozen
        )
        rnn_class = nn.LSTM if cell == "lstm" else nn.GRU  # same arguments for both
        self.rnn = rnn_class(
            input_size=embedding_matrix.shape[1],  # 300
            hidden_size=hidden_dim,
            num_layers=NUM_LAYERS,  # 2
            batch_first=True,  # input [B, L, 300] instead of [L, B, 300]
            bidirectional=bidirectional,
            dropout=dropout,  # applied between the 2 RNN layers
        )
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)  
        n_outputs = 2 if input_format == "joint" else 1 
        self.classifier = Classifier(rnn_out_dim, clf_hidden_dim, n_outputs, dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """[B, L] ids + [B] true lengths -> [B, H]: final hidden state of the top RNN layer."""
        embedded = self.embedding(ids)  # [B, L, 300]
        # packing: the RNN stops at each sequence's true length, so <pad> never changes the final state
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)  # LSTM has cell state, GRU has no cell state
        h_n = state[0] if isinstance(state, tuple) else state  # [num_layers * num_directions, B, H]
        if self.hparams.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=-1)  # top layer: forward + backward -> [B, 2H]
        return h_n[-1]  # top layer -> [B, H]

    def forward(self, batch: dict) -> torch.Tensor:
        scores = [self.classifier(self.encode(ids, lengths)) for ids, lengths in zip(batch["ids"], batch["lengths"])]
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
        params = [p for p in self.parameters() if p.requires_grad] 
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def train(args: argparse.Namespace):
    L.seed_everything(args.seed)

    make_dataset = partial(ARTSequenceDataset, input_format=args.input_format)
    dm = ARTDataModule(make_dataset=make_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                       num_workers=args.num_workers, pin_memory=args.accelerator == "gpu")
    dm.setup("fit")  # done manually: the model needs dm.embedding_matrix
    model = RNNClassifier(dm.embedding_matrix, input_format=args.input_format, hidden_dim=args.hidden_dim,
                          clf_hidden_dim=args.clf_hidden_dim, dropout=args.dropout,
                          bidirectional=args.bidirectional, lr=args.lr, weight_decay=args.weight_decay,
                          cell=args.cell)

    prefix = ("bi" if args.bidirectional else "") + args.cell  # lstm, bilstm, gru, bigru
    run_name = (f"{prefix}-{args.input_format}_lr{args.lr:g}_h{args.hidden_dim}_c{args.clf_hidden_dim}"
                f"_d{args.dropout:g}_p{args.patience}")
    logger = WandbLogger(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=args.cell, name=run_name)
    logger.log_hyperparams(vars(args) | {"vocab_size": len(dm.vocab), "num_layers": NUM_LAYERS})

    # early stopping on val/loss, keep the epoch with the best val/acc
    checkpoint = ModelCheckpoint(dirpath=CHECKPOINT_ROOT / args.cell / logger.experiment.id, filename="best",
                                 monitor="val/acc", mode="max", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)

    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator=args.accelerator, devices=1, logger=logger,
                        callbacks=[checkpoint, early_stopping], gradient_clip_val=args.grad_clip,
                        deterministic=True)
    trainer.fit(model, datamodule=dm)
    logger.experiment.summary["best_val_acc"] = checkpoint.best_model_score.item()  # epoch kept in best.ckpt
    logger.experiment.summary["best_val_loss"] = early_stopping.best_score.item()  # lowest val/loss
    log_predictions(trainer, model, dm.val_dataloader(), dm.ds["validation"], logger, key="val/predictions")

    if args.test:  
        trainer.test(model, datamodule=dm, ckpt_path="best")
        log_predictions(trainer, model, dm.test_dataloader(), dm.ds["test"], logger, key="test/predictions")
    wandb.finish()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
