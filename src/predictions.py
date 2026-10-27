"""Log every example of a split with the model's prediction as a W&B table (for error analysis)."""
import torch

COLUMNS = ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2", "h1_only", "h2_only",
           "label", "pred", "p_h1", "p_h2", "correct"]


def log_predictions(trainer, model, loader, split_ds, logger, key: str) -> None:
    """Run the BEST checkpoint on `loader` (must not be shuffled) and log one table row per example.

    `split_ds` is the matching Hugging Face split, so rows line up with the loader's order.
    h1_only / h2_only: words that appear in only one hypothesis (shows what was swapped).
    """
    probs = torch.cat(trainer.predict(model, loader, ckpt_path="best"))  # [N, 2], via predict_step
    rows = []
    for ex, (p_h1, p_h2) in zip(split_ds, probs.tolist()):
        words_h1, words_h2 = set(ex["hypothesis_1"].lower().split()), set(ex["hypothesis_2"].lower().split())
        pred = 1 if p_h1 >= p_h2 else 2
        rows.append([
            ex["observation_1"], ex["observation_2"], ex["hypothesis_1"], ex["hypothesis_2"],
            " ".join(sorted(words_h1 - words_h2)), " ".join(sorted(words_h2 - words_h1)),
            ex["label"], pred, round(p_h1, 3), round(p_h2, 3), pred == ex["label"],
        ])
    logger.log_table(key=key, columns=COLUMNS, data=rows)
