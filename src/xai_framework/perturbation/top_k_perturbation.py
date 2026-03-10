import logging

import numpy as np
import pandas as pd

from explainer.base_explainer import ExplanationResult
from perturbation.base_perturbation import (
    BasePerturbation,
    CategoricalPerturbationMixin,
)
from perturbation.registry import PERTURBATIONS


@PERTURBATIONS.register_module("TopKFeatures")
class TopKPerturbation(CategoricalPerturbationMixin, BasePerturbation):
    """Simulates a white-box adversarial attack by perturbing only the top-K
    most influential features as identified by an explainer.

    Top-K features are selected by absolute explanation value per instance.
    Continuous top-K features are perturbed with Gaussian noise scaled by
    lambda_param * MAD. Categorical top-K features are flipped to a different
    valid category sampled proportionally to its training frequency.

    This stress-tests the worst-case vulnerability of the XAI narrative: if an
    adversary can mask fraud by slightly altering only the most influential
    features, the system is highly vulnerable to fairwashing.

    Required params:
        k (int): Number of top features to perturb per instance.
        lambda (float): Noise scale for continuous features.
    """

    def validate_params(self) -> None:
        top_k = self.cfg.params.get("k")
        lambda_param = self.cfg.params.get("lambda")

        if top_k is None:
            raise ValueError(
                f"Perturbation '{self.cfg.name}' is missing required param 'top_k'."
            )
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"Perturbation '{self.cfg.name}': 'top_k' must be a positive "
                f"integer, got {top_k!r}."
            )

        self.top_k = top_k

        if lambda_param is None:
            raise ValueError(
                f"Perturbation '{self.cfg.name}' is missing required param 'lambda'."
            )
        if not isinstance(lambda_param, (int, float)) or lambda_param <= 0:
            raise ValueError(
                f"Perturbation '{self.cfg.name}': 'lambda' must be a "
                f"positive number, got {lambda_param!r}."
            )
        self.lambda_param = float(lambda_param)

    def _perturb_instance(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Perturb only the top-K features per instance by absolute explanation value.

        Args:
            X: DataFrame of instances to perturb.
            explanation_result: ExplanationResult containing one Explanation
                per row in X, aligned by instance_id.

        Raises:
            ValueError: If explanation_result is not provided or an Explanation
                is missing for any instance in X.
        """
        explanation_result: ExplanationResult | None = kwargs.get("explanation_result")
        if explanation_result is None:
            raise ValueError(
                f"Perturbation '{self.cfg.name}' (type: 'targeted') requires "
                "'explanation_result' to be passed as a kwarg."
            )

        feature_names: list[str] = explanation_result.feature_names

        explanation_lookup = {
            exp.instance_id: exp for exp in explanation_result.instances
        }

        X_perturbed = X.copy()

        for idx in X_perturbed.index:
            if idx not in explanation_lookup:
                raise ValueError(
                    f"Perturbation '{self.cfg.name}': no Explanation found for "
                    f"instance_id '{idx}'. Ensure explanation_result covers all "
                    "instances in X."
                )

            explanation = explanation_lookup[idx]

            # Select top-K features by absolute explanation value
            abs_values = np.abs(explanation.values.flatten())
            top_k_indices = np.argsort(abs_values)[::-1][: self.top_k]
            top_k_features = [feature_names[i] for i in top_k_indices]

            for col in top_k_features:
                if col not in X_perturbed.columns:
                    logging.warning(
                        "Top-K feature '%s' from explanation is not present in X. "
                        "Skipping.",
                        col,
                    )
                    continue

                if col in self.continuous_features:
                    mad = self.feature_stats[col]["mad"]
                    noise = self.rng.normal(scale=self.lambda_param * mad)
                    X_perturbed.at[idx, col] = X_perturbed.at[idx, col] + noise

                elif col in self.categorical_features:
                    X_perturbed = self._flip_categorical_feature(X_perturbed, col)

                else:
                    logging.info(
                        "Top-K feature '%s' is immutable or unrecognised. Skipping.",
                        col,
                    )

        return X_perturbed
