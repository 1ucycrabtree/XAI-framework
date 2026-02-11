from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseExperiment(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, X: pd.DataFrame) -> Any:
        pass
