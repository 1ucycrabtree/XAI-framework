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
        if self.decision_threshold is None:
            return self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.decision_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
