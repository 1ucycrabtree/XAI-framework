from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REGISTRY: dict[str, type["BaseModel"]] = {}


# Decorator and registry for models keyed by architecture name.
def register_model(architecture: str):
    """Decorator to register a model for an architecture."""

    def decorator(cls: type["BaseModel"]):
        _REGISTRY[architecture.lower()] = cls
        return cls

    return decorator


def get_model(architecture: str, model_path: str) -> "BaseModel":
    key = architecture.lower()
    model_cls = _REGISTRY.get(key)
    if not model_cls:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Supported: {list(_REGISTRY.keys())}"
        )

    return model_cls(model_path)


class BaseModel(ABC):
    def __init__(self, model_path: str):
        self.path = Path(model_path)

    @property
    @abstractmethod
    def model(self) -> Any:
        """Return the underlying model object (e.g., CatBoostClassifier)."""
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def validate_features(self, feature_names: list[str]) -> bool:
        pass

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        pass
