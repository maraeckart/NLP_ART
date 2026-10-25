# per sentence
import re
import string
import unicodedata  

import numpy as np
from nltk.tokenize import word_tokenize  

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
SPECIAL_TOKENS = [PAD, UNK, SEP]  




def clean_text(text: str) -> str:
    """Fix formatting issues found in the data (no HTML present)."""
    text = text.replace("’", "'").replace("´", "'").replace("`", "'")
    text = re.sub("\ufffd+", "'", text)
    text = re.sub(r"\s+", " ", text)  
    return text.strip()

def remove_punctuation(tokens: list[str]) -> list[str]:
    """Drop tokens that are just punctuation characters"""
    return [t for t in tokens if not all(ch in string.punctuation or unicodedata.category(ch).startswith("P") for ch in t)]

def replace_digits(tokens: list[str]) -> list[str]:
    """GoogleNews word2vec writes digits as '#' (10 -> ##)"""
    return [re.sub(r"\d", "#", t) for t in tokens]

def tokenize(text: str) -> list[str]:
    """Split into words and punctuation ("woods." -> ["woods", "."])"""
    return word_tokenize(text)


def preprocess(text: str, remove_punct: bool = True) -> list[str]:
    """Full pipeline for one sentence: clean -> tokenize -> remove punctuation -> digits to #
    (digits last: '#' counts as punctuation, so '2003' -> '####' would otherwise be dropped)"""
    tokens = tokenize(clean_text(text))
    if remove_punct:
        tokens = remove_punctuation(tokens)
    return replace_digits(tokens)


def build_vocab(token_lists: list[list[str]], wv) -> dict[str, int]:
    """Vocabulary from the TRAIN split: special tokens + every word that has a word2vec vector"""
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tokens in token_lists:
        for tok in tokens:
            if tok not in vocab and tok in wv:
                vocab[tok] = len(vocab)
    return vocab


def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """Map tokens to ids; words not in the vocabulary -> <unk>."""
    unk_id = vocab[UNK]
    return [vocab.get(tok, unk_id) for tok in tokens]


def build_embedding_matrix(vocab: dict[str, int], wv, seed: int = 42) -> np.ndarray:
    """Rows = word2vec vectors in vocab order. <pad>/<unk> = zeros, <sep> = small random vector."""
    matrix = np.zeros((len(vocab), wv.vector_size), dtype=np.float32)
    for tok, idx in vocab.items():
        if tok in wv:
            matrix[idx] = wv[tok]
    matrix[vocab[SEP]] = np.random.default_rng(seed).normal(0, 0.1, wv.vector_size)
    return matrix

