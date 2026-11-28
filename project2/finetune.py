"""Train RoBERTa (random init or pretrained) on ART."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from project2.models import TransformerClassifier
from project2.preprocessing import INPUT_FORMATS, TransformerDataModule
from src.predictions import log_predictions

WANDB_ENTITY = "mara-eckart-nlp"
WANDB_PROJECT = "art-p2"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "project2"
MODEL_NAME = "roberta-base"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, choices=["random", "pretrained"])
    p.add_argument("--input_format", default="separate", choices=INPUT_FORMATS)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--clf_hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)  # classifier only, the encoder keeps its config (0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu", "mps"])
    p.add_argument("--precision", default="32-true", choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace):
    L.seed_everything(args.seed)  # before building the model: makes the random weights reproducible

    dm = TransformerDataModule(MODEL_NAME, input_format=args.input_format, batch_size=args.batch_size,
                               num_workers=args.num_workers, pin_memory=args.accelerator == "gpu")
    model = TransformerClassifier(MODEL_NAME, init=args.init, input_format=args.input_format,
                                  clf_hidden_dim=args.clf_hidden_dim, dropout=args.dropout, lr=args.lr,
                                  weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio)

    run_name = f"{args.init}-{args.input_format}_lr{args.lr:g}_bs{args.batch_size}_wd{args.weight_decay:g}_s{args.seed}"
    logger = WandbLogger(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=args.init, name=run_name)
    logger.log_hyperparams(vars(args) | {"model_name": MODEL_NAME})

    # early stopping on val/loss, keep the epoch with the best val/acc (as in project 1)
    checkpoint = ModelCheckpoint(dirpath=CHECKPOINT_ROOT / args.init / logger.experiment.id, filename="best",
                                 monitor="val/acc", mode="max", save_top_k=1,
                                 save_weights_only=True)  # ~0.5 GB instead of ~1.5 GB with optimizer state
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)

    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator=args.accelerator, devices=1,
                        precision=args.precision, logger=logger, gradient_clip_val=args.grad_clip,
                        callbacks=[checkpoint, early_stopping, LearningRateMonitor(logging_interval="step")],
                        deterministic="warn")  # some GPU ops have no deterministic version
    trainer.fit(model, datamodule=dm)
    logger.experiment.summary["best_val_acc"] = checkpoint.best_model_score.item()
    logger.experiment.summary["best_val_loss"] = early_stopping.best_score.item()
    log_predictions(trainer, model, dm.val_dataloader(), dm.ds["validation"], logger, key="val/predictions")

    if args.test:
        trainer.test(model, datamodule=dm, ckpt_path="best")
        log_predictions(trainer, model, dm.test_dataloader(), dm.ds["test"], logger, key="test/predictions")
    wandb.finish()


if __name__ == "__main__":
    train(parse_args())
