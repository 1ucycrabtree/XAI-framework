import numpy as np
from scipy.optimize import root_scalar

from explainer.explanation import Explanation
from metric.base_metric import BaseLocalMetric
from metric.registry import METRICS


@METRICS.register_module("RankBiasedOverlap")
class RankBiasedOverlapMetric(BaseLocalMetric):
    """
    Rank Biased Overlap (RBO).

    A top-weighted metric that compares the overlap of two ranked lists.
    It handles non-conjoint lists and weights changes at the top of the ranking
    higher than changes at the bottom, mimicking analyst behaviour.

    Params:
        W (float, optional, default=0.9): The weight parameter that controls the
            steepness of the weighting.
        d (int, optional, default=5): The depth up to which to compare the rankings.
        p (float, optional): Persistence parameter. If not provided, it is dynamically
            derived given W and depth d.

    Returns:
        float: RBO score between 0 and 1, where 1 means identical rankings.

    References:
        Webber et al. (2010). A Similarity Measure for Indefinite Rankings.
    """

    def validate_params(self):
        self.W = self.params.get("W", 0.9)
        self.d = self.params.get("d", 5)
        self.p = self.params.get("p", None)

        if not isinstance(self.W, (int, float)) or not (0 < self.W < 1):
            raise ValueError(
                f"RBO 'W' parameter must be strictly between 0 and 1. Got {self.W}."
            )
        if not isinstance(self.d, int) or self.d <= 0:
            raise ValueError(
                f"RBO 'd' parameter must be a positive integer. Got {self.d}."
            )

        if self.p is not None:
            if not isinstance(self.p, (int, float)) or not (0 < self.p < 1):
                raise ValueError(
                    f"RBO 'p' parameter must be strictly between 0 and 1. Got {self.p}."
                )
        else:
            self.p = self._calculate_persistence(float(self.W), int(self.d))

    def _calculate_persistence(self, W: float, d: int) -> float:
        """
        Numerically solves for the RBO parameter 'p' given a target depth 'd'
        and a target cumulative weight 'W', based on Equation in Webber et al. (2010):
            W_{1:d} = 1 - p^(d-1) + ((1-p)/p) * d * ln(1/(1-p))
        """

        def rbo_weight_equation(p):
            if p <= 0 or p >= 1:
                return -1  # Out of bounds, root_scalar will ignore this value
            term1 = 1 - p ** (d - 1)
            term2 = ((1 - p) / p) * d * np.log(1 / (1 - p))
            return term1 + term2 - W

        # Find root in the valid range (0,1) using a robust method
        result = root_scalar(
            rbo_weight_equation, bracket=(1e-6, 0.999999), method="brentq"
        )

        if not result.converged:
            raise ValueError(
                f"Failed to derive RBO persistence parameter 'p' for W={W} and d={d}."
            )
        return float(result.root)

    def _get_ranked_indices(self, values: np.ndarray) -> np.ndarray:
        """
        Ranks features by their absolute attribution magnitude.
        Returns a list of feature indices sorted from most to least important.
        """
        importances = np.abs(values)
        ranked_indices = np.argsort(-importances)

        return ranked_indices.tolist()

    def evaluate(
        self,
        baseline_explanation: Explanation,
        perturbed_explanation: Explanation,
        **kwargs,
    ) -> float:
        """
        Compute the Rank Biased Overlap (RBO) between the baseline and perturbed
        explanations.
            RBO = (1-p) * sum_{d=1}^D (p^(d-1) * (A_d / d))
        where A_d is the overlap between the top-d ranked features of both lists.
        """

        if self.p is None:
            raise ValueError(
                "RBO persistence parameter 'p' is not set. Check validate_params."
            )

        base_vals = np.ravel(baseline_explanation.values)
        perturbed_vals = np.ravel(perturbed_explanation.values)

        if base_vals.shape != perturbed_vals.shape:
            raise ValueError(
                "Baseline and perturbed explanations must have the same shape."
            )

        base_ranking = self._get_ranked_indices(base_vals)
        perturbed_ranking = self._get_ranked_indices(perturbed_vals)

        max_depth = max(len(base_ranking), len(perturbed_ranking))
        rbo_sum = 0.0

        set_base = set()
        set_perturbed = set()

        for depth in range(1, max_depth + 1):
            idx = depth - 1

            if idx < len(base_ranking):
                set_base.add(base_ranking[idx])
            if idx < len(perturbed_ranking):
                set_perturbed.add(perturbed_ranking[idx])

            A_d = len(set_base.intersection(set_perturbed))

            # (p^(d-1)) * (A_d / d)
            term = (self.p ** (depth - 1)) * A_d / depth
            rbo_sum += term

        rbo_score = (1 - self.p) * rbo_sum
        return float(rbo_score)
