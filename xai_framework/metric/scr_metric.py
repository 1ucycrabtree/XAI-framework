import numpy as np
from explainer.explanation import Explanation
from metric.base_metric import BaseLocalMetric
from metric.registry import METRICS


@METRICS.register_module("SignConsistencyRate")
class SignConsistencyRate(BaseLocalMetric):
    """
    Sign Consistency Rate (SCR).

    Measures the proportion of features that maintain the same direction of contribution
    (positive/negative) after perturbation.

    Formula:
        SCR = (1/d) * sum(I(sgn(E_x) == sgn(E_x')))

    where sgn(v) returns 1, -1, or 0.

    Params:
        Epsilon: float, optional, default=1e-6
            A small threshold to treat near-zero contributions as zero, to avoid noise
            affecting sign consistency.

    Returns:
        float: The proportion of features with consistent signs between baseline and
        perturbed explanations.
    """

    def validate_params(self):
        self.epsilon = self.params.get("epsilon", 1e-6)
        if not isinstance(self.epsilon, (int, float)) or self.epsilon < 0:
            raise ValueError(
                f"Invalid epsilon value for SignConsistencyRate: {self.epsilon}."
                f" Must be a non-negative number."
            )

    def evaluate(
        self,
        baseline_explanation: Explanation,
        perturbed_explanation: Explanation,
        **kwargs,
    ) -> float:
        base_vals = np.ravel(baseline_explanation.values)
        perturbed_vals = np.ravel(perturbed_explanation.values)

        if base_vals.shape != perturbed_vals.shape:
            raise ValueError(
                "Baseline and perturbed explanations must have the same shape."
            )

        base_signs = self._get_signs(base_vals)
        perturbed_signs = self._get_signs(perturbed_vals)

        matches = base_signs == perturbed_signs

        # Return the proportion of features that have the same sign
        return float(np.mean(matches))

    def _get_signs(self, values: np.ndarray) -> np.ndarray:
        """
        Returns +1 for positive, -1 for negative, and 0 for zero.
        Applies the epsilon threshold to treat micro-fluctuations as zero.
        """
        thresholded_values = np.where(np.abs(values) < self.epsilon, 0.0, values)
        return np.sign(thresholded_values)
