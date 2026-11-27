"""Tokenization and batching for the Transformer models"""
import os
from functools import partial

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data import load_ds
from src.preprocessing import clean_text

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  

INPUT_FORMATS = ["separate", "h_last", "h_only", "joint"]


def build_inputs(ex: dict, input_format: str, sep: str) -> list[tuple[str, str | None]]:
    """input formatting"""
    o1, o2 = clean_text(ex["observation_1"]), clean_text(ex["observation_2"])
    h1, h2 = clean_text(ex["hypothesis_1"]), clean_text(ex["hypothesis_2"])
    if input_format == "separate":
        return [(f"{o1} {h1} {o2}", None), (f"{o1} {h2} {o2}", None)]  # <s> O1 h O2 </s>
    if input_format == "h_last":
        return [(f"{o1} {o2}", h1), (f"{o1} {o2}", h2)]  # <s> O1 O2 </s></s> h </s>
    if input_format == "h_only":
        return [(h1, None), (h2, None)]  # <s> h </s>
    if input_format == "joint":
        return [(sep.join([o1, h1, h2, o2]), None)]  # <s> O1 </s></s> h1 </s></s> h2 </s></s> O2 </s>
    raise ValueError(input_format)


def encode_split(split_ds, tokenizer, input_format: str) -> list[dict]:
    """list of token id lists + target"""
    sep = tokenizer.sep_token * 2  # "</s></s>", same separator RoBERTa uses between two segments
    samples = []
    for ex in split_ds:
        input_ids = [tokenizer(text, pair)["input_ids"] for text, pair in build_inputs(ex, input_format, sep)]
        assert max(len(ids) for ids in input_ids) <= tokenizer.model_max_length  # no truncation needed
        samples.append({"input_ids": input_ids, "target": ex["target"]})
    return samples


def collate(batch: list[dict], pad_id: int) -> dict:
    """Pad all sequences of the batch to the longest one"""
    seqs = [torch.tensor(ids) for ex in batch for ids in ex["input_ids"]]
    input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()  # 1 = real token, 0 = padding
    B, n_seq = len(batch), len(batch[0]["input_ids"])
    return {
        "input_ids": input_ids.view(B, n_seq, -1),
        "attention_mask": attention_mask.view(B, n_seq, -1),
        "target": torch.tensor([ex["target"] for ex in batch]),
    }


class TransformerDataModule(L.LightningDataModule):
    def __init__(self, model_name: str = "roberta-base", input_format: str = "separate", batch_size: int = 32,
                 num_workers: int = 0, pin_memory: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.input_format = input_format
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.datasets = {}

    def setup(self, stage: str):
        if self.datasets:
            return
        self.ds = load_ds()  # raw text
        for split in self.ds:
            self.datasets[split] = encode_split(self.ds[split], self.tokenizer, self.input_format)

    def _dataloader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.datasets[split],  # a plain list works as a Dataset
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=partial(collate, pad_id=self.tokenizer.pad_token_id),
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
