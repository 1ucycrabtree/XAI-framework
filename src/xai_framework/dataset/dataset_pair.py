from dataclasses import dataclass

from dataset.dataset import Dataset


@dataclass
class DatasetPair:
    train: Dataset
    test: Dataset
