# per sentence
import re
import string
import unicodedata  

import numpy as np
from nltk.tokenize import word_tokenize  

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
SPECIAL_TOKENS = [PAD, UNK, SEP]  


MISSING_APOSTROPHE = re.compile(r"\b(did|does|do|was|were|is|are|could|would|should|have|has|had)nt\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Fix formatting issues found in the data """
    text = text.replace("’", "'").replace("´", "'").replace("`", "'")
    text = re.sub("\ufffd+", "'", text)
    text = MISSING_APOSTROPHE.sub(r"\1n't", text)  # didnt -> didn't (keeps the negation; not cant/wont: real words)
    text = re.sub(r"\s+", " ", text)  
    return text.strip()

def remove_punctuation(tokens: list[str]) -> list[str]:
    """Drop tokens that are just punctuation characters"""
    return [t for t in tokens if not all(ch in string.punctuation or unicodedata.category(ch).startswith("P") for ch in t)]

def replace_digits(tokens: list[str]) -> list[str]:
    """Replace digits as '#' (10 -> ##)"""
    return [re.sub(r"\d", "#", t) for t in tokens]

def tokenize(text: str) -> list[str]:
    """Split into words and punctuation"""
    return word_tokenize(text)

def preprocess(text: str, remove_punct: bool = True) -> list[str]:
    """Full pipeline per sentence: clean -> tokenize -> remove punctuation -> digits to #"""
    tokens = tokenize(clean_text(text))
    if remove_punct:
        tokens = remove_punctuation(tokens)
    return replace_digits(tokens)

def word2vec_key(tok: str, wv) -> str | None:
    """Lower/uppercase per token"""
    for candidate in (tok, tok.lower(), tok.capitalize()):
        if candidate in wv:
            return candidate
    return None

def build_vocab(token_lists: list[list[str]], wv) -> dict[str, int]:
    """special tokens"""
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tokens in token_lists:
        for tok in tokens:
            if tok not in vocab and word2vec_key(tok, wv) is not None:
                vocab[tok] = len(vocab)
    return vocab

def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """Mapping tokens to ids"""
    unk_id = vocab[UNK]
    return [vocab.get(tok, unk_id) for tok in tokens]

def build_embedding_matrix(vocab: dict[str, int], wv, seed: int = 42) -> np.ndarray:
    matrix = np.zeros((len(vocab), wv.vector_size), dtype=np.float32)
    for tok, idx in vocab.items():
        key = word2vec_key(tok, wv)
        if key is not None:
            matrix[idx] = wv[key]
    matrix[vocab[SEP]] = np.random.default_rng(seed).normal(0, 0.1, wv.vector_size)
    return matrix

