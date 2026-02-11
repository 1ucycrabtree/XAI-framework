from abc import ABC, abstractmethod
import pandas as pd

class BasePerturbation(ABC):
    
    @abstractmethod
    def perturb(self, X: pd.DataFrame) -> pd.DataFrame:
        pass