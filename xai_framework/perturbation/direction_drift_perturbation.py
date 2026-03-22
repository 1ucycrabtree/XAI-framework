import pandas as pd

from perturbation.base_perturbation import BasePerturbation
from perturbation.registry import PERTURBATIONS


@PERTURBATIONS.register_module("DirectionalDrift")
class DirectionalDriftPerturbation(BasePerturbation):
    """Perturbs a specified set of continuous features by shifting each value
    directionally toward the opposite tail of its training distribution.

    Direction is determined per-instance:
        - If value > median: shift toward q05 (lower tail)
        - If value < median: shift toward q95 (upper tail)
        - If value == median: direction is chosen randomly

    The shift magnitude is controlled by drift_factor ∈ (0, 1]:
        new_value = value + drift_factor * (target_tail - value)

    A drift_factor of 1.0 moves the value all the way to the tail boundary.
    A drift_factor of 0.5 moves it halfway.

    Required params:
        drift_factor (float): Must be in (0, 1].
        drift_factor_range (list[float], optional): If provided, a [min, max]
            range to sample a per-perturbation drift_factor. Both bounds must
            be in (0, 1] and min <= max.
        target_features (list[str]): Features to apply drift to. Must be
            continuous and present in the dataset.
    """

    def validate_params(self) -> None:
        """Parameters validation for DirectionalDriftPerturbation.
        Checks for presence, type, and value constraints.
        """

        drift_factor = self.cfg.params.get("drift_factor")
        drift_factor_range = self.cfg.params.get("drift_factor_range")
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
        self.drift_factor_range = None
        if drift_factor_range is not None:
            if (
                not isinstance(drift_factor_range, (list, tuple))
                or len(drift_factor_range) != 2
            ):
                raise ValueError(
                    f"Perturbation '{self.cfg.name}': 'drift_factor_range' must be "
                    "a list/tuple of two floats, e.g. [0.1, 0.4]."
                )
            low, high = drift_factor_range
            if not all(isinstance(v, (int, float)) for v in (low, high)):
                raise ValueError(
                    f"Perturbation '{self.cfg.name}': 'drift_factor_range' bounds "
                    f"must be numeric, got {drift_factor_range!r}."
                )
            low = float(low)
            high = float(high)
            if not (0 < low <= high <= 1):
                raise ValueError(
                    f"Perturbation '{self.cfg.name}': 'drift_factor_range' must be "
                    f"within (0, 1] and min <= max, got {drift_factor_range!r}."
                )
            self.drift_factor_range = (low, high)

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

        # Check each target feature is perturbable continuous and exists in dataset
        invalid = [
            f for f in target_features if f not in self.perturbable_continuous_features
        ]
        if invalid:
            raise ValueError(
                f"Perturbation '{self.cfg.name}': the following 'target_features' "
                f"are either not present in the dataset or are not perturbable continuous"  # noqa: E501
                f"features: {invalid}. "
                f"Available perturbable continuous features: {self.perturbable_continuous_features}."  # noqa: E501
            )
        self.target_features = target_features

    def _perturb_instance(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        drift_factor: float = self.cfg.params["drift_factor"]
        if self.drift_factor_range is not None:
            drift_factor = float(
                self.rng.uniform(self.drift_factor_range[0], self.drift_factor_range[1])
            )
        target_features: list[str] = self.cfg.params["target_features"]

        X_perturbed = X.copy()

        for col in target_features:
            median = self.feature_stats[col]["median"]
            q05 = self.feature_stats[col]["q05"]
            q95 = self.feature_stats[col]["q95"]
            min_val = self.feature_stats[col]["min"]
            max_val = self.feature_stats[col]["max"]

            if q95 == q05:
                continue

            values = X_perturbed[col]

            # Determine direction
            is_gt = values > median
            is_lt = values < median
            is_eq = ~(is_gt | is_lt)

            # If median randomly pick a direction
            random_targets = pd.Series(
                self.rng.choice([q05, q95], size=is_eq.sum()), index=values[is_eq].index
            )

            # Tails
            targets = pd.Series(index=values.index, dtype=float)
            targets[is_gt] = q05
            targets[is_lt] = q95
            targets[is_eq] = random_targets

            # Limits shift distance (less extreme q90 -> q05)
            max_shift = drift_factor * (q95 - q05)
            shift = drift_factor * (targets - values)
            shift = shift.clip(lower=-max_shift, upper=max_shift)  # clip to bounds

            X_perturbed[col] = (values + shift).clip(lower=min_val, upper=max_val)

        # Non-target continuous and all categorical features are unchanged
        X_perturbed = self._round_integer_features_for(X_perturbed)
        X_perturbed = self._enforce_non_negative_for(X_perturbed)
        return X_perturbed
