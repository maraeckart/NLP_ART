"""Find the largest batch size for the LSTM that fits into GPU memory"""
import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from RNN import ARTSequenceDataset, LSTMClassifier, collate_fn
from src.datamodule import ARTDataModule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_format", default="h_last", choices=["h_last", "separate", "joint"])
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--clf_hidden_dim", type=int, default=256)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--device", default="cuda", help="cuda on GPUHub (cpu only to test the script)")
    p.add_argument("--max_batch_size", type=int, default=None, help="upper limit")
    return p.parse_args()


def fits(model, batch_size: int, n_slots: int, max_len: int, vocab_size: int, device: str) -> bool:
    """One forward + backward pass on a worst-case batch: all sequences have the maximum length."""
    ids = [torch.randint(3, vocab_size, (batch_size, max_len), device=device) for _ in range(n_slots)]
    lengths = [torch.full((batch_size,), max_len) for _ in range(n_slots)]
    target = torch.zeros(batch_size, dtype=torch.long, device=device)
    try:
        loss = model.loss_fn(model({"ids": ids, "lengths": lengths}), target)
        loss.backward()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        model.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    dm = ARTDataModule(make_dataset=partial(ARTSequenceDataset, input_format=args.input_format), collate_fn=collate_fn)
    dm.setup("fit")
    train = dm.datasets["train"]
    max_len = max(len(seq) for sample in train.samples for seq in sample["sequences"])  # longest train sequence
    n_slots = len(train.samples[0]["sequences"])  # 2 (h_last/separate) or 1 (joint)
    limit = args.max_batch_size or len(train)
    print(f"worst case: {n_slots} sequence(s) per sample, each {max_len} tokens | upper limit {limit}")

    model = LSTMClassifier(dm.embedding_matrix, input_format=args.input_format, hidden_dim=args.hidden_dim,
                           clf_hidden_dim=args.clf_hidden_dim, dropout=0.2, bidirectional=args.bidirectional,
                           lr=1e-3, weight_decay=0.01).to(args.device).train()
    check = partial(fits, model, n_slots=n_slots, max_len=max_len, vocab_size=len(dm.vocab), device=args.device)

    # 1. double the batch size until it does not fit (or the upper limit is reached)
    good, bad, size = 0, None, 16
    while size < limit:
        if not check(size):
            bad = size
            break
        print(f"  {size:>6}: fits")
        good, size = size, size * 2
    if bad is None:  # never failed below the limit: try the limit itself
        if check(limit):
            print(f"  {limit:>6}: fits -> the whole upper limit fits into memory")
            good = limit
        else:
            bad = limit
    # 2. binary search between the last size that fit and the first that did not
    while bad is not None and bad - good > 1:
        mid = (good + bad) // 2
        if check(mid):
            good = mid
        else:
            bad = mid
    print(f"largest batch size that fits: {good} -> {-(-len(train) // good)} updates per epoch "
          f"(train set {len(train)} samples)")


if __name__ == "__main__":
    main()
