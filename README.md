# NLP_ART

HSLU NLP HS26 course project: **Abductive Commonsense Reasoning (ART)**.
Given two observations O1 and O2, choose the more plausible of two hypotheses.

- Dataset: [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)
- Project 1: word embeddings (word2vec) and word embeddings + RNN
- Project 2: Transformers (randomly initialized, pretrained, LLM)
- Experiment tracking: [W&B view "Project 1 – final results"](https://wandb.ai/mara-eckart-nlp/art-p1?nw=nn8kaxbqkgu)


## Dataset

ART, course version `amarfurt/art-nlp-hs26`: two observations O1, O2 and two hypotheses h1, h2, label = the more plausible hypothesis.
Train 16,801 / validation 1,000 / test 1,532. 
## Repository structure

```
NLP_ART/
├── src/                 # shared code: preprocessing, datamodule, classifier, prediction tables
├── project1/
│   ├── word2vec.py      # word2vec + 2-layer ReLU classifier
│   ├── RNN.py           # word2vec + 2-layer LSTM/GRU + 2-layer ReLU classifier
│   ├── find_batch_size.py
│   ├── sweeps/          # W&B sweep configs
│   └── notebooks/       # data exploration, preprocessing check
├── data/                # downloaded embeddings/caches 
├── checkpoints/         # model checkpoints 
├── EXPECTATIONS.md      # expected test scores, written before experiments
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt                    
wandb login
```

## Project 1

```bash
python project1/word2vec.py --lr 0.001 --hidden_dim 512 --dropout 0.2 --patience 6 --seed 42 --test
python project1/RNN.py --input_format h_last --lr 0.01 --hidden_dim 128 --dropout 0.5 --batch_size 512 --seed 42 --test
```
Add `--accelerator gpu --num_workers 4` on GPUHub. `--input_format h_only` gives the hypothesis-only baseline.

| model | seeds | val acc | test acc |
|---|---|---|---|
| word2vec + MLP | 42 | 0.711 | 0.524 |
| LSTM (h_last) | 42, 43, 44 | 0.704 ± 0.018 | 0.544 ± 0.007 |
| LSTM h_only | 42 | 0.697 | 0.522 |

- Hyperparameters chosen on validation only, test evaluated once per seed (best val/acc checkpoint)
- The LSTM is slightly better than h_only → it mostly uses the hypothesis, not the observations

## Preprocessing Decisions

- **Format cleaning:** no HTML in the data, but some curly apostrophes, broken characters, line breaks and double spaces → normalized; missing apostrophes restored (didnt → didn't, keeps negation); digits → `#` (`10` → `##`).
- **Tokenization:** NLTK `word_tokenize`, splits punctuation from words (`word.` → `word`, `.`).
- **No lowercasing:** Google News word2vec is cased, and capitalization carries information (names).
- **No stemming/lemmatization:** the pretrained vocabulary contains full word forms, stems often don't exist in it.
- **No stopword removal:** small words like *never*/*not* can be the only difference between h1 and h2.
- **Punctuation removed:** almost every sentence ends with `.`, so it carries no signal and has few word2vec vectors (optional flag for the RNN).
- **Unknown words:** mapped to `<unk>` and ignored when averaging. GoogleNews lacks lowercase *to/a/and/of* but has *To/A/And/Of*, so lookup tries exact → lowercase → Capitalized (train `<unk>` rate 11.6% → 1.0%; remaining: *'s*, names, typos, rare words).
- **No truncation:** sentences are short (median 8 words, max 30).
- **Labels:** {1, 2} mapped to class indices {0, 1} (`target`), cross-entropy over the two scores.
- **Splits:** given train/validation/test splits; tuning on validation only, test once at the end. 6 train rows with an empty hypothesis (`{}.`/`.`) removed.
- **Vocabulary/embedding:** built from train only (words with a word2vec vector + `<pad>`, `<unk>`, `<sep>`), pretrained word2vec-google-news-300.
- **Batching/padding:** pad to the longest sentence per batch with `<pad>`


## References

- Bhagavatula et al. (2020). Abductive Commonsense Reasoning. ICLR. https://arxiv.org/abs/1908.05739
