from abc import ABC, abstractmethod

from dataset.dataset import Dataset


class DataLoader(ABC):
    @abstractmethod
    def load(self, *args, **kwargs) -> Dataset:
        pass
