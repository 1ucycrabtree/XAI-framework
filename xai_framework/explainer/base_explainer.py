from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from dataset.dataset import Dataset
from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult
from load_config import ExplainerConfig
from model.base_model import BaseModel


class BaseExplainer(ABC):
    """
    Abstract base class for model explainers. Defines the interface
    for generating feature attributions.
    """

    def __init__(
        self,
        cfg: ExplainerConfig,
        model_wrapper: BaseModel,
    ):
        self.cfg = cfg
        self.model_wrapper = model_wrapper
        self._initialise_explainer()

    @abstractmethod
    def _initialise_explainer(self) -> None:
        pass

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> ExplanationResult:
        pass

    @abstractmethod
    def plot_local_explanation(self, X):
        pass

    def explanation_result(
        self,
        values: np.ndarray,
        base_value: float,
        index: pd.Index,
        feature_names: list[str],
        predictions: np.ndarray,
    ) -> ExplanationResult:
        if len(predictions) != len(values):
            raise ValueError(
                f"Predictions length ({len(predictions)}) does not match explanations length ({len(values)})."  # noqa: E501
            )

        explanations = []
        for idx, val in enumerate(values):
            prediction = predictions[idx]
            explanations.append(
                Explanation(
                    instance_id=index[idx],
                    values=val,
                    base_value=base_value,
                    prediction=prediction,
                )
            )

        return ExplanationResult(
            explainer_name=self.__class__.__name__,
            instances=explanations,
            base_value=base_value,
            feature_names=feature_names,
            instance_ids=list(index),
        )

    def plot_global_explanation(self, X):
        pass


class BaseDatasetExplainer(BaseExplainer):
    def __init__(
        self,
        cfg: ExplainerConfig,
        model_wrapper: BaseModel,
        train_dataset: Optional[Dataset] = None,
    ):
        if train_dataset is None:
            raise ValueError(f"{self.__class__.__name__} requires a training dataset.")
        self.train_dataset = train_dataset
        super().__init__(cfg, model_wrapper)
