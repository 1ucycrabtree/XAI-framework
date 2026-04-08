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
        k (int): Maximum number of perturbable features to perturb per instance.
        lambda (float): Noise scale for continuous features.
    """

    def validate_params(self) -> None:
        top_k = self.cfg.params.get("k")
        lambda_param = self.cfg.params.get("lambda")

        if top_k is None:
            raise ValueError(
                f"Perturbation '{self.cfg.name}' is missing required param 'k'."
            )
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError(
                f"Perturbation '{self.cfg.name}': 'k' must be a non-negative "
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

        continuous_cols_in_X = [
            col
            for col in self.perturbable_continuous_features
            if col in X_perturbed.columns
        ]
        # Ensure continuous columns are float for noise addition, but only if they exist in X  # noqa: E501
        if continuous_cols_in_X:
            X_perturbed[continuous_cols_in_X] = X_perturbed[
                continuous_cols_in_X
            ].astype(float)

        for idx in X_perturbed.index:
            if idx not in explanation_lookup:
                raise ValueError(
                    f"Perturbation '{self.cfg.name}': no Explanation found for "
                    f"instance_id '{idx}'. Ensure explanation_result covers all "
                    "instances in X."
                )

            explanation = explanation_lookup[idx]

            # Rank all features, keep perturbable subset, then take first K.
            abs_values = np.abs(explanation.values.flatten())
            ranked_indices = np.argsort(abs_values)[::-1]
            ranked_features = [feature_names[i] for i in ranked_indices]

            eligible = []
            for col in ranked_features:
                if col not in X_perturbed.columns:
                    continue
                if col in self.immutable_features:
                    continue
                if (
                    col in self.perturbable_continuous_features
                    or col in self.perturbable_categorical_features
                ):
                    eligible.append(col)

            realised_k = min(self.top_k, len(eligible))
            selected_features = eligible[:realised_k]
            n_perturbed = 0

            if realised_k == 0:
                self._record_instance_log(
                    {
                        "instance_id": idx,
                        "strategy": self.cfg.name,
                        "requested_k": int(self.top_k),
                        "eligible_perturbable": int(len(eligible)),
                        "realised_k": 0,
                        "selected_features": [],
                        "perturbed_features_count": 0,
                        "skipped_no_eligible_features": True,
                    }
                )
                continue

            for col in selected_features:
                if col in self.perturbable_continuous_features:
                    mad = self.feature_stats[col]["mad"]
                    if pd.isna(mad) or mad == 0:
                        continue
                    noise = self.rng.normal(scale=self.lambda_param * mad)
                    val = pd.to_numeric(X_perturbed.at[idx, col], errors="coerce")
                    if pd.isna(val):
                        continue
                    X_perturbed.at[idx, col] = val + noise
                    n_perturbed += 1

                elif col in self.perturbable_categorical_features:
                    X_perturbed = self._flip_categorical_feature(
                        X_perturbed, col, idx=idx
                    )
                    n_perturbed += 1

            self._record_instance_log(
                {
                    "instance_id": idx,
                    "strategy": self.cfg.name,
                    "requested_k": int(self.top_k),
                    "eligible_perturbable": int(len(eligible)),
                    "realised_k": int(realised_k),
                    "selected_features": selected_features,
                    "perturbed_features_count": int(n_perturbed),
                    "skipped_no_eligible_features": False,
                }
            )

        X_perturbed = self._round_integer_features_for(X_perturbed)
        X_perturbed = self._enforce_non_negative_for(X_perturbed)

        return X_perturbed
