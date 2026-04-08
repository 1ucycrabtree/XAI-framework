from pathlib import Path

import pandas as pd
from dataset.base_loader import BaseDataLoader
from dataset.registry import DATASET_TYPES

file_suffix = ".parquet"


@DATASET_TYPES.register_module(file_suffix)
class ParquetDataLoader(BaseDataLoader):
    SUPPORTED_SUFFIX = file_suffix

    def _load_file(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)
