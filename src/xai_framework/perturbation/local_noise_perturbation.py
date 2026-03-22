import pandas as pd

from perturbation.base_perturbation import BasePerturbation
from perturbation.registry import PERTURBATIONS


@PERTURBATIONS.register_module("LocalGaussianNoise")
class LocalGaussianNoisePerturbation(BasePerturbation):
    """Perturbs continuous features by adding local Gaussian noise scaled by
    the feature's Median Absolute Deviation (MAD).
    MAD is used rather than standard deviation because financial data such as
    TransactionAmt typically exhibits extreme positive skewness. MAD provides
    a robust measure of local scale that is less sensitive to outliers,
    ensuring perturbations remain strictly local.

    Noise is drawn from: ε ~ N(0, (λ · MAD)²)

    Required params:
        lambda (float): Scale factor controlling noise magnitude.
    """

    def validate_params(self) -> None:
        """Parameters validation for LocalGaussianNoisePerturbation.
        Checks for presence, type, and value constraints.
        """

        lambda_param = self.cfg.params.get("lambda")
        if lambda_param is None:
            raise ValueError(
                f"Perturbation '{self.cfg.name}' is missing "
                f"the required parameter 'lambda'."
            )
        if not isinstance(lambda_param, (int, float)) or lambda_param < 0:
            raise ValueError(
                f"Perturbation '{self.cfg.name}': 'lambda' must be a "
                f"positive number, got {lambda_param!r}."
            )
        self.lambda_param = float(lambda_param)

    def _perturb_instance(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        already_ood: set[str] = kwargs.get("already_ood", set())
        X_perturbed = X.copy()

        for feature in self.perturbable_continuous_features:
            if feature in already_ood:
                continue

            mad = self.feature_stats[feature]["mad"]
            if pd.isna(mad) or mad == 0:
                continue

            noise = self.rng.normal(
                scale=self.lambda_param * mad, size=len(X_perturbed)
            )
            X_perturbed[feature] = X_perturbed[feature].astype(float) + noise

        X_perturbed = self._clip_to_distribution_for(X_perturbed, exclude=already_ood)
        X_perturbed = self._round_integer_features_for(X_perturbed, exclude=already_ood)
        X_perturbed = self._enforce_non_negative_for(X_perturbed, exclude=already_ood)

        return X_perturbed
