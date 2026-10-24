# per sentence
def clean_text(text: str) -> str:
    pass
def tokenize(text: str) -> list[str]:
    pass
def lowercase(tokens: list[str]) -> list[str]:
    pass
def remove_punctuation(tokens: list[str]) -> list[str]:
    pass
def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    pass
def stem(tokens: list[str]) -> list[str]:
    pass
def lemmatize(tokens: list[str]) -> list[str]:
    pass
def preprocess(text: str, config: dict) -> list[str]:
    pass

# dataset level 
def build_vocab(token_lists: list[list[str]], min_freq: int, specials: list[str]) -> dict[str, int]:
    pass
def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    pass

# batch level
def truncate(ids: list[int], max_len: int) -> list[int]:
    pass