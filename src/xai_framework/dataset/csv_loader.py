from pathlib import Path

import pandas as pd

from dataset.data_loader import DataLoader, register_loader

file_suffix = ".csv"


@register_loader(file_suffix)
class CsvDataLoader(DataLoader):
    SUPPORTED_SUFFIX = file_suffix

    def _load_file(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)
