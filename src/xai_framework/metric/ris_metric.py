import logging

import numpy as np
import pandas as pd

from explainer.explanation import Explanation
from metric.base_metric import BaseLocalMetric
from metric.registry import METRICS


@METRICS.register_module("RelativeInputStability")
class RelativeInputStabilityMetric(BaseLocalMetric):
    """
    Relative Input Stability (RIS).

    Quantifies the sensitivity of the explanation to input perturbations by
    normalising the explanation distance (L2 norm) against the input distance (Gower).
    High RIS indicates that a small change in input caused a disproportionately
    large change in the explanation (instability).

    Formula:
        RIS(x) = D_exp(E_x, E_x') / D_inp(x, x')

    Params:
        epsilon (float, optional): A small constant added to the denominator to prevent division by zero when input distance is very small. Default is 1e-6.

    References:
        Agarwal et al. (2022). Rethinking Stability for Attribution-based Explanations.
    """  # noqa: E501

    def validate_params(self):
        self.epsilon = self.params.get("epsilon", 1e-6)
        if not isinstance(self.epsilon, (int, float)) or self.epsilon < 0:
            raise ValueError(
                f"Invalid epsilon value for RelativeInputStabilityMetric: {self.epsilon}. "  # noqa: E501
                f"Must be a non-negative number."
            )

    def _compute_gower_distance(
        self, base_input: pd.Series, perturbed_input: pd.Series, continuous_ranges: dict
    ) -> float:
        """
        Compute the Gower distance between two input vectors, normalizing continuous features
        by their specified ranges.

        Args:
            base_input: Original input features.
            perturbed_input: Perturbed input features.
            continuous_ranges: Dictionary of {feature_name: (min, max)} for normalisation.

        Returns:
            float: The Gower distance between the two inputs.
        """  # noqa: E501
        if base_input.shape != perturbed_input.shape:
            raise ValueError("Base and perturbed inputs have different shapes.")

        total_features = len(base_input)
        distance_sum = 0.0

        # only calculate distance for changed features to avoid unnecessary computation
        changed_cols = base_input.index[base_input != perturbed_input]

        for col in changed_cols:
            val_base = base_input[col]
            val_perturbed = perturbed_input[col]

            if col in continuous_ranges:
                min_val, max_val = continuous_ranges[col]
                rng = max_val - min_val
                if rng > 0:
                    distance_sum += abs(val_base - val_perturbed) / rng
            else:
                distance_sum += 1.0  # categorical features strict mismatch

        return float(distance_sum / total_features)

    def evaluate(
        self,
        baseline_explanation: Explanation,
        perturbed_explanation: Explanation,
        **kwargs,
    ) -> float:
        # get input features from kwargs
        base_input = kwargs.get("baseline_input")
        perturbed_input = kwargs.get("perturbed_input")
        continuous_ranges = kwargs.get(
            "continuous_ranges", self.params.get("continuous_ranges")
        )

        if base_input is None or perturbed_input is None:
            raise ValueError(
                "RelativeInputStabilityMetric requires 'baseline_input' and 'perturbed_input' in kwargs."  # noqa: E501
            )

        if not isinstance(base_input, pd.Series) or not isinstance(
            perturbed_input, pd.Series
        ):
            raise TypeError("RIS inputs must be provided as pandas Series.")

        if continuous_ranges is None:
            continuous_ranges = {}

        if not isinstance(continuous_ranges, dict):
            raise ValueError(
                "RelativeInputStabilityMetric expects 'continuous_ranges' to be a dictionary if provided."  # noqa: E501
            )

        base_vals = np.ravel(baseline_explanation.values)
        perturbed_vals = np.ravel(perturbed_explanation.values)

        d_exp = np.linalg.norm(base_vals - perturbed_vals, ord=2)

        d_inp = self._compute_gower_distance(
            base_input, perturbed_input, continuous_ranges
        )

        if d_inp < self.epsilon:
            logging.warning(
                f"Input distance (D_inp) is very small ({d_inp:.6f}). Returning 0.0 to avoid instability in RIS calculation."  # noqa: E501
            )
            return 0.0

        ris_score = d_exp / d_inp
        return float(ris_score)
