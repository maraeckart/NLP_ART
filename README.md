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

- Labels {1, 2} mapped to class indices {0, 1}(`target`)

## Tools used

- Claude Code (Anthropic) was used as a coding assistant for project setup and code; all code was reviewed and is understood by the author.

## References

- Bhagavatula et al. (2020). Abductive Commonsense Reasoning. ICLR. https://arxiv.org/abs/1908.05739
