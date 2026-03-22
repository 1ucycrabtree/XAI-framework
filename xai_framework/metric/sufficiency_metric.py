from typing import List, Tuple

import numpy as np
import pandas as pd
from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult
from metric.base_metric import BaseGlobalMetric
from metric.registry import METRICS


@METRICS.register_module("GlobalSufficiencyMetric")
class GlobalSufficiencyMetric(BaseGlobalMetric):
    def validate_params(self):
        # We discretise continuous explanations into property pi (Top-K feature names)
        self.k = self.params.get("top_k_depth", self.params.get("k", 5))
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(
                f"GlobalSufficiencyMetric requires positive integer top_k_depth (or k). Got {self.k}."  # noqa: E501
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
        Calculates Global Sufficiency (m_s).
        Tests if the property pi (Top-K features) is sufficient to predict the
        model's classification across the entire distribution.
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
                    "Global Sufficiency requires model predictions to evaluate faithfulness"  # noqa: E501
                )

            y = (
                int(exp.prediction)
                if isinstance(exp.prediction, (int, float, np.number))
                else exp.prediction
            )

            pi = self._get_property_pi(exp, feature_names)
            data.append({"pi": pi, "y": y})

        df = pd.DataFrame(data)

        # Total instances with the exact same property pi
        pi_counts = df.groupby("pi").size().to_dict()

        # For each property pi, find the count of its most frequent model prediction
        pi_max_y_counts = (
            df.groupby("pi")["y"].apply(lambda x: x.value_counts().max()).to_dict()
        )

        sufficiency_sum = 0.0

        # Sufficiency probability for every instance i
        for _, row in df.iterrows():
            pi_i = row["pi"]

            # P(y | pi): probability that sharing this Top-5 tuple guarantees the dominant label  # noqa: E501
            p_y_given_pi = pi_max_y_counts[pi_i] / pi_counts[pi_i]

            sufficiency_sum += p_y_given_pi

        m_s = sufficiency_sum / n
        return float(m_s)
