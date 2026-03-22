from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from load_config import ModelConfig


class BaseModel(ABC):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.path = Path(cfg.file_path)
        self._load_model()

    @property
    @abstractmethod
    def model(self) -> Any:
        """Return the underlying model object (e.g., CatBoostClassifier)."""
        pass

    @abstractmethod
    def _load_model(self) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def validate_features(self, feature_names: list[str]) -> bool:
        if (
            not hasattr(self.model, "feature_names_")
            or self.model.feature_names_ is None
        ):
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
        if (
            not hasattr(self.model, "feature_names_")
            or self.model.feature_names_ is None
        ):
            raise ValueError(
                "Model does not have feature names - is the model loaded correctly?"
            )
        return list(self.model.feature_names_)
