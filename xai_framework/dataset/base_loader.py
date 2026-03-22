import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from dataset.dataset import Dataset
from dataset.dataset_pair import DatasetPair
from load_config import DatasetConfig


class BaseDataLoader(ABC):
    SUPPORTED_SUFFIX: str = ""

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.datasets = self._load_datasets()

    @property
    def train(self) -> Dataset:
        return self.datasets.train

    @property
    def test(self) -> Dataset:
        return self.datasets.test

    @abstractmethod
    def _load_file(self, path: Path) -> pd.DataFrame:
        pass

    def _load_datasets(self) -> DatasetPair:
        train_dataset = self._load_single_dataset(
            self.cfg.train_file_path, split="train"
        )
        test_dataset = self._load_single_dataset(self.cfg.test_file_path, split="test")
        self._validate_pair(train_dataset, test_dataset)

        return DatasetPair(train=train_dataset, test=test_dataset)

    def _load_single_dataset(self, file_path: str, split: str) -> Dataset:
        path = Path(file_path)

        logging.info(
            f"Loading {split} dataset from '{path}' using {self.__class__.__name__}."
        )

        if not path.exists():
            raise FileNotFoundError(f"{split.capitalize()} file not found: {path}")
        if path.suffix.lower() != self.SUPPORTED_SUFFIX:
            raise ValueError(
                f"Expected a {self.SUPPORTED_SUFFIX} file for {split} set, got: {path.suffix}"  # noqa: E501
            )

        df = self._load_file(path)

        if self.cfg.target_label not in df.columns:
            raise ValueError(
                f"Target label '{self.cfg.target_label}' not found in {split} dataset. "
                f"This framework expects ground truth labels on both splits for "
                f"model evaluation and explanation validation. "
            )

        y = df[self.cfg.target_label].copy()
        X = df.drop(columns=[self.cfg.target_label])
        logging.info(
            f"Split {split} dataset by target label '{self.cfg.target_label}'."
        )

        logging.info(
            f"{split.capitalize()} dataset: {X.shape[0]} samples, "
            f"{X.shape[1]} features, "
            f"class distribution: {y.value_counts().to_dict()}"
        )

        return Dataset(
            X=X,
            y=y,
            target_label=self.cfg.target_label,
            feature_names=list(X.columns),
            exclude_columns=self.cfg.drop_columns,
            categorical_features=self.cfg.categorical_features,
            integer_features=self.cfg.integer_features,
            non_negative_features=self.cfg.non_negative_features,
            non_negative_prefixes=self.cfg.non_negative_prefixes,
            metadata={
                "source": self.__class__.__name__,
                "split": split,
                **(self.cfg.metadata or {}),
            },
        )

    def _validate_pair(self, train: Dataset, test: Dataset) -> None:
        self._validate_columns_match(train, test)
        self._validate_dtypes_match(train, test)
        self._validate_target_coverage(train, test)

    def _validate_columns_match(self, train: Dataset, test: Dataset) -> None:
        train_cols = list(train.X_full.columns)
        test_cols = list(test.X_full.columns)

        if train_cols == test_cols:
            return

        train_set, test_set = set(train_cols), set(test_cols)
        only_in_train = train_set - test_set
        only_in_test = test_set - train_set

        if only_in_train or only_in_test:
            raise ValueError(
                f"Feature mismatch between train and test splits.\n"
                f"  Only in train: {sorted(only_in_train) or 'none'}\n"
                f"  Only in test:  {sorted(only_in_test) or 'none'}"
            )

        raise ValueError(
            f"Train and test have the same columns but in different order.\n"
            f"  Train: {train_cols}\n"
            f"  Test:  {test_cols}"
        )

    def _validate_dtypes_match(self, train: Dataset, test: Dataset) -> None:
        mismatches = {
            col: (str(train.X_full[col].dtype), str(test.X_full[col].dtype))
            for col in train.X_full.columns
            if train.X_full[col].dtype != test.X_full[col].dtype
        }
        if mismatches:
            detail = "\n".join(
                f"  '{col}': train={dtypes[0]}, test={dtypes[1]}"
                for col, dtypes in mismatches.items()
            )
            raise ValueError(f"Dtype mismatches between train and test:\n{detail}")

    def _validate_target_coverage(self, train: Dataset, test: Dataset) -> None:
        train_classes = set(train.y.unique())
        test_classes = set(test.y.unique())
        unseen = test_classes - train_classes
        if unseen:
            logging.warning(
                f"Test set contains classes not present in train: {unseen}. "
                f"Model predictions for these classes will be unreliable."
            )
