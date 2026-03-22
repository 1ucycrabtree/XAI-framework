import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from load_config import ModelConfig
from model.base_model import BaseModel
from model.registry import MODELS


@MODELS.register_module("CatBoost")
class CatBoostFraudModel(BaseModel):
    def __init__(self, cfg: ModelConfig):
        self.decision_threshold: float | None = None
        metadata = cfg.metadata if isinstance(cfg.metadata, dict) else {}
        threshold = metadata.get("decision_threshold")
        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "model.metadata.decision_threshold must be numeric if provided."
                ) from exc
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("model.metadata.decision_threshold must be in [0, 1].")
            self.decision_threshold = threshold
        super().__init__(cfg)

    @property
    def model(self):
        return self._model

    def _load_model(self) -> None:
        logging.info(f"Loading model from {self.path}.")

        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {self.path}.")

        try:
            self._model = CatBoostClassifier()
            self._model.load_model(self.path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model from {self.path}: {e}.")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._prepare_catboost_input(X)
        if self.decision_threshold is None:
            return self.model.predict(X_prepared)
        probs = self.model.predict_proba(X_prepared)[:, 1]
        return (probs >= self.decision_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._prepare_catboost_input(X)
        return self.model.predict_proba(X_prepared)

    def _prepare_catboost_input(self, X: pd.DataFrame) -> pd.DataFrame:
        feature_names = list(self.model.feature_names_ or [])
        if isinstance(X, pd.DataFrame):
            X_prepared = X.copy()
        else:
            array_input = np.asarray(X)
            if array_input.ndim == 1:
                array_input = array_input.reshape(1, -1)
            if array_input.ndim != 2:
                return X
            if feature_names and array_input.shape[1] == len(feature_names):
                X_prepared = pd.DataFrame(array_input, columns=feature_names)
            else:
                return X

        cat_feature_indices = self.model.get_cat_feature_indices()

        for idx in cat_feature_indices:
            if idx < 0 or idx >= len(X_prepared.columns):
                continue
            col = X_prepared.columns[idx]
            X_prepared[col] = X_prepared[col].fillna("missing").astype(str)

        return X_prepared
