# NLP_ART

HSLU NLP HS26 course project: **Abductive Commonsense Reasoning (ART)**.
Given two observations O1 and O2, choose the more plausible of two hypotheses.

- Dataset: [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26) (train 16,801 / validation 1,000 / test 1,532)
- Project 1: word embeddings (word2vec) and word embeddings + RNN
- Project 2: Transformers (randomly initialized, pretrained, LLM)
- Experiment tracking: Weights & Biases — _link TBD_

## Repository structure

```
NLP_ART/
├── src/                # shared code: preprocessing, data loading, training
├── project1/notebooks/ # project 1 experiments (word2vec, RNN)
├── data/               # downloaded embeddings/caches (not versioned)
├── checkpoints/        # model checkpoints (not versioned)
├── EXPECTATIONS.md     # expected test scores, written before experiments
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt                    
wandb login
```

## Preprocessing Decisions

- **Format cleaning:** no HTML in the data, but some curly apostrophes, broken characters, line breaks and double spaces → normalized; digits → `#` (GoogleNews format, e.g. `10` → `##`).
- **Tokenization:** NLTK `word_tokenize`, splits punctuation from words (`woods.` → `woods`, `.`).
- **No lowercasing:** Google News word2vec is cased, and capitalization carries information (names).
- **No stemming/lemmatization:** the pretrained vocabulary contains full word forms, stems often don't exist in it.
- **No stopword removal:** small words like *never*/*not* can be the only difference between h1 and h2.
- **Punctuation removed:** almost every sentence ends with `.`, so it carries no signal and has few word2vec vectors (optional flag for the RNN).
- **Unknown words:** mapped to `<unk>` and ignored when averaging. ≈11.8% of train tokens are `<unk>`, mainly *to/a/and/of/'s*, which GoogleNews leaves out.
- **No truncation:** sentences are short (median 8 words, max 30).
- **Feature selection:** all of O1, O2 and h are used, no hand-made features.
- **Input format:** each hypothesis is scored separately with the same model: `[avg(O1); avg(O2); avg(h)]` (word2vec) or `O1 <sep> h <sep> O2` (RNN).
- **Labels:** {1, 2} mapped to class indices {0, 1} (`target`), cross-entropy over the two scores.
- **Splits:** given train/validation/test splits; tuning on validation only, test once at the end. 6 train rows with an empty hypothesis (`{}.`/`.`) removed.
- **Vocabulary/embedding:** built from train only (words with a word2vec vector + `<pad>`, `<unk>`, `<sep>`), pretrained word2vec-google-news-300.
- **Batching/padding:** pad to the longest sentence per batch with `<pad>`.

## Tools used

- Claude Code (Anthropic) was used as a coding assistant for project setup and code; all code was reviewed and is understood by the author.

## References

- Bhagavatula et al. (2020). Abductive Commonsense Reasoning. ICLR. https://arxiv.org/abs/1908.05739
