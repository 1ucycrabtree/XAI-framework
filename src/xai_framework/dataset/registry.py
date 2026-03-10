from pathlib import Path

from dataset.dataset_pair import DatasetPair
from load_config import DatasetConfig
from utils.registry import Registry

DATASET_TYPES = Registry("dataset_types")


def get_dataset(cfg: DatasetConfig) -> DatasetPair:
    train_suffix = Path(cfg.train_file_path).suffix.lower()
    test_suffix = Path(cfg.test_file_path).suffix.lower()

    if train_suffix != test_suffix:
        raise ValueError(
            f"Train and test files must have the same format. "
            f"Got train='{train_suffix}' and test='{test_suffix}'."
        )

    data_cls = DATASET_TYPES.get(test_suffix)
    loader = data_cls(cfg)
    return loader.datasets
