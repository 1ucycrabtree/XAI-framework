import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from model.model import BaseModel, register_model


@register_model("catboost")
class CatBoostFraudModel(BaseModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._model = CatBoostClassifier()
        self._load_model()

    @property
    def model(self):
        return self._model

    def _load_model(self) -> None:
        logging.info(f"Loading model from {self.path}.")

        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {self.path}.")

        try:
            self._model.load_model(self.path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model from {self.path}: {e}.")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def validate_features(self, feature_names: list[str]) -> bool:
        if self.model.feature_names_ is None:
            raise ValueError(
                "Model does not have feature names - is the model loaded correctly?"
            )

        expected = set(self.model.feature_names_)
        provided = set(feature_names)

        if expected != provided:
            missing = expected - provided
            extra = provided - expected

            if missing:
                raise ValueError(f"Missing features: {missing}")

            if extra:
                raise ValueError(f"Unexpected features: {extra}")

        return True

    @property
    def feature_names(self) -> list[str]:
        if self.model.feature_names_ is None:
            raise ValueError(
                "Model does not have feature names - is the model loaded correctly?"
            )
        return list(self.model.feature_names_)
