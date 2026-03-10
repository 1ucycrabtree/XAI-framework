import pandas as pd

from perturbation.base_perturbation import BasePerturbation
from perturbation.registry import PERTURBATIONS


@PERTURBATIONS.register_module("DirectionalDrift")
class DirectionalDriftPerturbation(BasePerturbation):
    """Perturbs a specified set of continuous features by shifting each value
    directionally toward the opposite tail of its training distribution.

    Direction is determined per-instance:
        - If value > median: shift toward q01 (lower tail)
        - If value < median: shift toward q99 (upper tail)
        - If value == median: direction is chosen randomly

    The shift magnitude is controlled by drift_factor ∈ (0, 1]:
        new_value = value + drift_factor * (target_tail - value)

    A drift_factor of 1.0 moves the value all the way to the tail boundary.
    A drift_factor of 0.5 moves it halfway.

    Required params:
        drift_factor (float): Must be in (0, 1].
        target_features (list[str]): Features to apply drift to. Must be
            continuous and present in the dataset.
    """

    def validate_params(self) -> None:
        """Parameters validation for DirectionalDriftPerturbation.
        Checks for presence, type, and value constraints.
        """

        drift_factor = self.cfg.params.get("drift_factor")
        target_features = self.cfg.params.get("target_features")

        if drift_factor is None:
            raise ValueError(
                f"Perturbation '{self.cfg.name}' is missing "
                "required param 'drift_factor'."
            )
        if not isinstance(drift_factor, (int, float)) or not (0 < drift_factor <= 1):
            raise ValueError(
                f"Perturbation '{self.cfg.name}': 'drift_factor' must be a "
                f"float in (0, 1], got {drift_factor!r}."
            )
        self.drift_factor = float(drift_factor)

        if target_features is None:
            raise ValueError(
                f"Perturbation '{self.cfg.name}' is missing "
                "required param 'target_features'."
            )
        if not isinstance(target_features, list) or len(target_features) == 0:
            raise ValueError(
                f"Perturbation '{self.cfg.name}': 'target_features' must be a "
                "non-empty list of feature names."
            )

        # Check each target feature is continuous and exists in the dataset
        invalid = [f for f in target_features if f not in self.continuous_features]
        if invalid:
            raise ValueError(
                f"Perturbation '{self.cfg.name}': the following 'target_features' "
                f"are either not present in the dataset or are not continuous "
                f"features: {invalid}. "
                f"Available continuous features: {self.continuous_features}."
            )
        self.target_features = target_features

    def _perturb_instance(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        drift_factor: float = self.cfg.params["drift_factor"]
        target_features: list[str] = self.cfg.params["target_features"]

        X_pert = X.copy()

        for col in target_features:
            median = self.feature_stats[col]["median"]
            q01 = self.feature_stats[col]["q01"]
            q99 = self.feature_stats[col]["q99"]

            for idx in X_pert.index:
                value = X_pert.at[idx, col]

                if value > median:
                    target_tail = q01
                elif value < median:
                    target_tail = q99
                else:
                    # Median tie, randomly pick a direction
                    target_tail = self.rng.choice([q01, q99])

                new_value = float(value + drift_factor * (target_tail - value))
                X_pert.at[idx, col] = new_value

        # Non-target continuous and all categorical features are unchanged
        return X_pert
