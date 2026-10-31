from typing import Callable

import lightning as L
import nltk
from torch.utils.data import DataLoader, Dataset

from src.data import TEXT_COLUMNS, load_ds
from src.embeddings import load_word2vec
from src.preprocessing import build_embedding_matrix, build_vocab, preprocess


class ARTDataModule(L.LightningDataModule):
    """loading data -> preprocessing -> build vocab + embedding matrix from train -> datasets + dataloaders"""

    def __init__(self, make_dataset: Callable[..., Dataset], collate_fn: Callable, batch_size: int = 64,
                 num_workers: int = 0, pin_memory: bool = False):
        super().__init__()
        self.make_dataset = make_dataset  # called as make_dataset(tokenized, targets, vocab)
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.datasets = {}

    def setup(self, stage: str):
        if self.datasets:  # already set up: Lightning calls setup again for trainer.test
            return
        nltk.download("punkt_tab", quiet=True)
        self.ds = load_ds()  # raw text, kept for the prediction table
        tokenized = {
            split: {column: [preprocess(text) for text in self.ds[split][column]] for column in TEXT_COLUMNS}
            for split in self.ds
        }
        wv = load_word2vec()
        train_tokens = [tokens for column in TEXT_COLUMNS for tokens in tokenized["train"][column]]
        self.vocab = build_vocab(train_tokens, wv)  # train only: val/test words not in train -> <unk>
        self.embedding_matrix = build_embedding_matrix(self.vocab, wv)  # [vocab size, 300], passed to the model
        for split in self.ds:
            self.datasets[split] = self.make_dataset(tokenized[split], self.ds[split]["target"], self.vocab)

    def _dataloader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
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
