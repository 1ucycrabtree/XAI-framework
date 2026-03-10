import logging

import pandas as pd


class Dataset:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_label: str,
        feature_names: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        metadata: dict | None = None,
    ):
        self.__X = X  # Use a private attribute to prevent accidental usage of original X df  # noqa: E501
        self.y = y
        self.target_label = target_label
        self.feature_names = feature_names or list(self.__X.columns)
        self.exclude_columns = exclude_columns or []
        self.metadata = metadata or {}

        self._validate()

    def _validate(self):
        if not isinstance(self.__X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if self.__X.shape[0] == 0:
            raise ValueError("Dataset X cannot be empty")
        if not isinstance(self.y, pd.Series):
            raise ValueError("y must be a pandas Series")
        if len(self.y) != len(self.__X):
            raise ValueError("Length of y must match the number of rows in X")
        if not self.__X.index.equals(self.y.index):
            raise ValueError("Index of X and y must match")
        if not isinstance(self.target_label, str):
            raise ValueError("target_label must be a string")
        if not isinstance(self.feature_names, list) or not all(
            isinstance(f, str) for f in self.feature_names
        ):
            raise ValueError(
                "Something has gone wrong parsing the X columns to features.\n"
                "Feature names must be a list of strings."
            )
        if set(self.feature_names) != set(self.__X.columns):
            raise ValueError("Feature names must match the columns in X")
        if self.exclude_columns:
            missing = [c for c in self.exclude_columns if c not in self.__X.columns]
            if missing:
                logging.warning(
                    f"Exclude columns not found in dataset (already removed?): {missing}. "  # noqa: E501
                    f"These will be ignored."
                )
        if self.metadata and not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")

    @property
    def X_model(self) -> pd.DataFrame:
        """Features used for model explanation and perturbation. Excludes any columns specified in exclude_columns."""  # noqa: E501
        return self.__X.drop(columns=self.exclude_columns, errors="ignore")

    @property
    def X_full(self) -> pd.DataFrame:
        """All features including those in exclude_columns. Useful for model evaluation."""  # noqa: E501
        return self.__X

    @property
    def model_feature_names(self) -> list[str]:
        """Feature names used for model explanation and perturbation. Excludes any columns specified in exclude_columns."""  # noqa: E501
        return list(self.X_model.columns)
