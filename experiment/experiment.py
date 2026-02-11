from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseExperiment(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, X: pd.DataFrame) -> Any:
        pass
