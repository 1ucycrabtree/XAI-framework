import os

import pandas as pd

from dataset.data_loader import DataLoader
from dataset.dataset import Dataset


class ParquetDataLoader(DataLoader):
    def __init__(self):
        pass

    def load(
        self,
        x_path: str,
        y_path: str | None = None,
        target_label: str | None = None,
        metadata: dict | None = None,
    ) -> Dataset:
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"File not found: {x_path}")

        X = pd.read_parquet(x_path)

        y = None
        if y_path:
            if not os.path.exists(y_path):
                raise FileNotFoundError(f"File not found: {y_path}")

            y = pd.read_parquet(y_path)

            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                y = y.squeeze()

        return Dataset(
            X=X,
            y=y,
            target_label=target_label,
            feature_names=list(X.columns),
            metadata={"source": "parquet", **(metadata or {})},
        )
