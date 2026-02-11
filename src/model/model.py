from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    
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