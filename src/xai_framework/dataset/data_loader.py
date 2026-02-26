import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from dataset.dataset import Dataset

_REGISTRY: dict[str, type["DataLoader"]] = {}


# Decorator and registry for dataset loaders based on file extension!
def register_loader(suffix: str):
    """Decorator to register a loader for a file extension."""

    def decorator(cls: type["DataLoader"]):
        _REGISTRY[suffix.lower()] = cls
        return cls

    return decorator


def get_dataset_loader(path: str) -> "DataLoader":
    suffix = Path(path).suffix.lower()
    loader_cls = _REGISTRY.get(suffix)
    if not loader_cls:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: {list(_REGISTRY.keys())}"
        )
    return loader_cls()


class DataLoader(ABC):
    SUPPORTED_SUFFIX: str = ""

    def load(
        self,
        path: str | Path,
        target_label: str,
        drop_columns: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Dataset:
        path = Path(path)

        logging.info(f"Loading dataset from '{path}' using {self.__class__.__name__}.")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != self.SUPPORTED_SUFFIX:
            raise ValueError(
                f"Expected a {self.SUPPORTED_SUFFIX} file, got: {path.suffix}"
            )

        df = self._load_file(path)

        if target_label not in df.columns:
            raise ValueError(
                f"Target '{target_label}' not found in columns: {list(df.columns)}"
            )

        y = df[target_label].copy()
        X = df.drop(columns=[target_label])
        logging.info(f"Split dataset by target label '{target_label}'.")

        if drop_columns:
            missing_cols = [col for col in drop_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Columns to drop not found in dataset: {missing_cols}.\n"
                    f"Available columns: {list(df.columns)}."
                )
            logging.info(f"Dropping columns: {drop_columns}.")
            X = X.drop(columns=drop_columns)

        logging.info(
            f"Dataset loaded with {X.shape[0]} rows and {X.shape[1]} features."
        )  # noqa: E501

        return Dataset(
            X=X,
            y=y,
            target_label=target_label,
            feature_names=list(X.columns),
            metadata={"source": self.__class__.__name__, **(metadata or {})},
        )

    @abstractmethod
    def _load_file(self, path: Path) -> pd.DataFrame:
        pass
