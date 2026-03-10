from abc import ABC, abstractmethod

from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult
from load_config import MetricConfig


class BaseLocalMetric(ABC):
    """
    Base class for local metrics comparing two explanations
    (baseline vs perturbed) for the same instance.
    """

    def __init__(self, cfg: MetricConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.params = cfg.params or {}
        self.validate_params()

    @abstractmethod
    def validate_params(self):
        """Validate metric params from self.cfg.params."""
        pass

    @abstractmethod
    def evaluate(
        self,
        baseline_explanation: Explanation,
        perturbed_explanation: Explanation,
        **kwargs,
    ) -> float:
        """
        Compute a local metric value comparing baseline and perturbed explanations.
        """
        pass


class BaseGlobalMetric(ABC):
    """
    Base class for global metrics computed over a collection of explanations.
    """

    def __init__(self, cfg: MetricConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.params = cfg.params or {}
        self.validate_params()

    @abstractmethod
    def validate_params(self):
        """Validate metric params from self.cfg.params."""
        pass

    @abstractmethod
    def evaluate(self, explanations: ExplanationResult, **kwargs) -> float:
        """Compute a dataset-level metric from explanations."""
        pass
