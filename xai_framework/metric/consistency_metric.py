from typing import List, Tuple

import numpy as np
import pandas as pd

from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult
from metric.base_metric import BaseGlobalMetric
from metric.registry import METRICS


@METRICS.register_module("GlobalConsistencyMetric")
class GlobalConsistencyMetric(BaseGlobalMetric):
    """
    Evaluates Global Consistency (m_c)
    across an entire dataset of baseline explanations based on Dasgupta et al. (2022).
    """

    def validate_params(self):
        self.k = self.params.get("top_k_depth", self.params.get("k", 5))
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(
                f"GlobalConsistencyMetric requires positive integer top_k_depth (or k). Got {self.k}."  # noqa: E501
            )

    def _get_property_pi(
        self, explanation: Explanation, feature_names: List[str]
    ) -> Tuple[str, ...]:
        importances = np.abs(np.ravel(explanation.values))

        top_k_indices = np.argsort(-importances, kind="mergesort")[: self.k]
        top_k_features = tuple(feature_names[i] for i in top_k_indices)
        return top_k_features

    def evaluate(self, explanations: ExplanationResult, **kwargs) -> float:
        """
        Calculates Global Consistency (m_c) using Equation 3.6.
        Measures the extent to which instances sharing the same explanation
        property pi also share the same predicted label.
        """
        instances = explanations.instances
        n = len(instances)
        if n == 0:
            return 0.0

        feature_names = explanations.feature_names

        data = []
        for exp in instances:
            if exp.prediction is None:
                raise ValueError(
                    f"Explanation for instance {exp.instance_id} is missing a prediction. "  # noqa: E501
                    "Global Consistency requires model predictions to evaluate faithfulness."  # noqa: E501
                )

            # ensure prediction is a discrete label
            y = (
                int(exp.prediction)
                if isinstance(exp.prediction, (int, float, np.number))
                else exp.prediction
            )

            pi = self._get_property_pi(exp, feature_names)
            data.append({"pi": pi, "y": y})

        df = pd.DataFrame(data)

        # Total instances with exact same property pi
        pi_counts = df.groupby("pi").size().to_dict()

        # Total instances with same property pi and same label y
        pi_y_counts = df.groupby(["pi", "y"]).size().to_dict()

        consistency_sum = 0.0

        # Evaluate for every instance i
        for _, row in df.iterrows():
            pi_i = row["pi"]
            y_i = row["y"]

            N_pi = pi_counts[pi_i]
            N_pi_y = pi_y_counts.get((pi_i, y_i), 0)

            # Only consider properties that are shared by more than one instance
            if N_pi > 1:
                term = (N_pi_y - 1) / (N_pi - 1)
                consistency_sum += term
        m_c = consistency_sum / n
        return float(m_c)
