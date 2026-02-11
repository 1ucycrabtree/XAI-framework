from catboost import CatBoostClassifier
from pathlib import Path
import logging
import numpy as np
import pandas as pd

from model.model import BaseModel


class CatBoostFraudModel(BaseModel):
    def __init__(self, model_path: str):
        self.path = Path(model_path)
        self.model = CatBoostClassifier()
        self._load_model()

    def _load_model(self):
        logging.info(f"Loading model from {self.path}")

        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {self.path}")

        try:
            self.model.load_model(self.path)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model from {self.path}: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def validate_features(self, data_columns: list[str]) -> bool:
        expected = set(self.model.feature_names_)
        provided = set(data_columns)
        
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
        return list(self.model.feature_names_)