from pathlib import Path

import pandas as pd

from dataset.base_loader import BaseDataLoader
from dataset.registry import DATASET_TYPES

file_suffix = ".csv"


@DATASET_TYPES.register_module(file_suffix)
class CsvDataLoader(BaseDataLoader):
    SUPPORTED_SUFFIX = file_suffix

    def _load_file(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)
