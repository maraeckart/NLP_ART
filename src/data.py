
from datasets import DatasetDict, load_dataset

DATASET_NAME = "amarfurt/art-nlp-hs26"
TEXT_COLUMNS = ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]


def load_ds() -> DatasetDict:
    ds = load_dataset(DATASET_NAME)
    return ds.map(lambda ex: {"target": ex["label"] - 1})
