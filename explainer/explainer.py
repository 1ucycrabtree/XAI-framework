from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult


class BaseExplainer(ABC):
    """
    Abstract base class for model explainers. Defines the interface
    for generating feature attributions.
    """

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> ExplanationResult:
        pass

    def explanation_result(
        self,
        values: np.ndarray,
        base_value: float,
        index: pd.Index,
        feature_names: list[str],
    ) -> ExplanationResult:
        explanations = []
        for idx, val in enumerate(values):
            explanations.append(
                Explanation(instance_id=index[idx], values=val, base_value=base_value)
            )

        return ExplanationResult(
            explainer_name=self.__class__.__name__,
            instances=explanations,
            base_value=base_value,
            feature_names=feature_names,
            instance_ids=list(index),
        )

    @abstractmethod
    def plot_local_explanation(self, X):
        pass

    @abstractmethod
    def plot_global_explanation(self, X):
        pass
