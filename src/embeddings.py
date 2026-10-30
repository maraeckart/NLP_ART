from pathlib import Path

from gensim.models import KeyedVectors

GENSIM_DIR = Path(__file__).resolve().parents[1] / "data" / "gensim"
W2V_GZ = GENSIM_DIR / "word2vec-google-news-300" / "word2vec-google-news-300.gz"
W2V_KV = GENSIM_DIR / "word2vec-google-news-300.kv"


def load_word2vec() -> KeyedVectors:
    if not W2V_KV.exists():
        if not W2V_GZ.exists():
            import os
            os.environ["GENSIM_DATA_DIR"] = str(GENSIM_DIR)
            import gensim.downloader as api
            api.load("word2vec-google-news-300", return_path=True) 
        KeyedVectors.load_word2vec_format(W2V_GZ, binary=True).save(str(W2V_KV))
    return KeyedVectors.load(str(W2V_KV), mmap="r")
