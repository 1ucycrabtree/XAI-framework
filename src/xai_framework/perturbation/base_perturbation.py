import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from load_config import PerturbationConfig

MAX_OOD_RETRIES = 5


class BasePerturbation(ABC):
    def __init__(
        self,
        cfg: PerturbationConfig,
        training_data: pd.DataFrame,
        immutable_features: list[str] | None = None,
    ):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

        self._derive_feature_types(training_data, immutable_features)
        self._calculate_continuous_feature_stats(training_data)
        self._calculate_valid_categorical_values(training_data)

        self.validate_params()

    @abstractmethod
    def validate_params(self):
        """Validate that all required params for the perturbation strategy
        are present and correctly formatted in the config (self.cfg.params).

        Raise ValueError with a clear message if not."""
        pass

    @abstractmethod
    def _perturb_instance(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate a single perturbation of X. Called internally by perturb()."""
        pass

    def perturb(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """For each instance in X, generate n_perturbations perturbed copies.

        Each perturbation is checked against the OOD gate. If OOD, retries up
        to MAX_OOD_RETRIES times before falling back to clipping.

        Returns a DataFrame with len(X) * n_perturbations rows, preserving the
        original index so each perturbed row can be traced back to its source.
        """
        all_perturbed = []

        total_instances = len(X) * self.cfg.n_perturbations
        start_time = time.time()
        perturbation_count = 0

        logging.info(
            f"Starting '{self.cfg.name}' perturbation for {len(X)} instances "
            f"generating {total_instances} perturbations..."
        )

        for idx in X.index:
            stop_event = getattr(self, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError(
                    f"Perturbation '{self.cfg.name}' interrupted by stop request."
                )
            row = X.loc[[idx]]
            already_ood = self._get_already_ood_features(row)

            for _ in range(self.cfg.n_perturbations):
                perturbation_count += 1
                logging.debug(
                    f"Perturbing instance {idx} with strategy '{self.cfg.name}'."
                )
                X_pert = self._perturb_instance(row, already_ood=already_ood, **kwargs)

                for attempt in range(1, MAX_OOD_RETRIES + 1):
                    if self._is_in_distribution(X_pert, excluded_features=already_ood):
                        break
                    logging.debug(
                        f"OOD perturbation for '{self.cfg.name}' instance {idx} "
                        f"(attempt {attempt}/{MAX_OOD_RETRIES}). Retrying..."
                    )
                    X_pert = self._perturb_instance(
                        row, already_ood=already_ood, **kwargs
                    )
                else:
                    logging.warning(
                        f"Max OOD retries reached for '{self.cfg.name}' "
                        f"instance {idx}. Applying clipping."
                    )
                    X_pert = self._clip_to_distribution(X_pert)
                all_perturbed.append(X_pert)

            if perturbation_count % 10 == 0 or perturbation_count == total_instances:
                elapsed = time.time() - start_time
                avg_time_per_inst = elapsed / perturbation_count
                eta = avg_time_per_inst * (total_instances - perturbation_count)

                logging.info(
                    f"[{self.cfg.name}] Progress: {perturbation_count}/{total_instances} | "  # noqa: E501
                    f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                )

        total_time = time.time() - start_time
        logging.info(
            f"Generated {len(all_perturbed)} perturbations for '{self.cfg.name}' "
            f"in {total_time:.1f}s."
        )
        return pd.concat(all_perturbed, ignore_index=False)

    def _get_already_ood_features(self, X: pd.DataFrame) -> set[str]:
        """Identify features in X that are already outside the training distribution."""
        already_ood = set()

        for col in self.continuous_features:
            low = self.feature_stats[col]["q01"]
            high = self.feature_stats[col]["q99"]
            col_values = X[col].dropna()
            if len(col_values) == 0:
                continue
            if not ((col_values >= low) & (col_values <= high)).all():
                already_ood.add(col)

        for col in self.categorical_features:
            invalid = ~X[col].isin(self.valid_categories[col])
            if invalid.any():
                already_ood.add(col)

        return already_ood

    def _is_in_distribution(
        self, X: pd.DataFrame, excluded_features: set[str] | None = None
    ) -> bool:
        """Return True if all values in X are within the valid data manifold."""
        return self._is_numerically_valid(
            X, excluded_features=excluded_features
        ) and self._is_categorically_valid(X, excluded_features=excluded_features)

    def _is_numerically_valid(
        self, X: pd.DataFrame, excluded_features: set[str] | None = None
    ) -> bool:
        """Check continuous features are within 1st and 99th percentile of training data"""  # noqa: E501

        excluded_features = excluded_features or set()
        for col in self.continuous_features:
            if col in excluded_features:
                continue
            low = self.feature_stats[col]["q01"]
            high = self.feature_stats[col]["q99"]
            col_values = X[col].dropna()
            if len(col_values) == 0:
                continue
            if not ((col_values >= low) & (col_values <= high)).all():
                return False
        return True

    def _is_categorically_valid(
        self, X: pd.DataFrame, excluded_features: set[str] | None = None
    ) -> bool:
        """Check categorical features only take on values seen in training data"""  # noqa: E501

        excluded_features = excluded_features or set()
        for col in self.categorical_features:
            if col in excluded_features:
                continue
            invalid = ~X[col].isin(self.valid_categories[col])
            if invalid.any():
                return False
        return True

    def _clip_to_distribution(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fallback: clip continuous features to [q01, q99] bounds.
        Categorical features are left unchanged — invalid categories should
        not arise from well-implemented subclasses.

        # TODO: Consider multivariate manifold enforcement in future work.
        """
        X_clipped = X.copy()
        for col in self.continuous_features:
            low = self.feature_stats[col]["q01"]
            high = self.feature_stats[col]["q99"]
            X_clipped[col] = X_clipped[col].clip(lower=low, upper=high)
        return X_clipped

    def _clip_to_distribution_for(
        self, X: pd.DataFrame, exclude: set[str] | None = None
    ) -> pd.DataFrame:
        """Clip continuous features to [q01, q99] bounds, excluding any in the 'exclude' set."""  # noqa: E501
        exclude = exclude or set()
        X_clipped = X.copy()
        for col in self.continuous_features:
            if col in exclude:
                continue
            low = self.feature_stats[col]["q01"]
            high = self.feature_stats[col]["q99"]
            X_clipped[col] = X_clipped[col].clip(lower=low, upper=high)
        return X_clipped

    def _derive_feature_types(
        self, training_data: pd.DataFrame, immutable_features: list[str] | None = None
    ):
        immutable = immutable_features or []
        continuous = []
        categorical = []

        for col in training_data.columns:
            if pd.api.types.is_numeric_dtype(training_data[col]):
                continuous.append(col)
            else:
                categorical.append(col)

        self.immutable_features = immutable
        self.continuous_features = [col for col in continuous if col not in immutable]
        self.categorical_features = [col for col in categorical if col not in immutable]

    def _calculate_continuous_feature_stats(self, training_data: pd.DataFrame):
        self.feature_stats = {}
        for col in self.continuous_features:
            self.feature_stats[col] = {
                "q01": training_data[col].quantile(0.01),
                "q99": training_data[col].quantile(0.99),
                "median": training_data[col].median(),
                "mad": (training_data[col] - training_data[col].median())
                .abs()
                .median(),  # Median Absolute Deviation
            }

    def _calculate_valid_categorical_values(self, training_data: pd.DataFrame):
        """Pre-calculate valid categories and their training frequencies.
        Frequencies are used as sampling probabilities during categorical flipping.
        Only categories with P > 0 in training data are considered valid.
        """

        self.valid_categories = {}
        self.categorical_frequencies = {}

        for col in self.categorical_features:
            value_counts = training_data[col].value_counts()
            self.valid_categories[col] = value_counts.index.tolist()
            self.categorical_frequencies[col] = (
                value_counts / value_counts.sum()
            ).to_dict()


class CategoricalPerturbationMixin:
    """Mixin for perturbation strategies that need to flip categorical features."""

    def _flip_categorical_feature(
        self: "BasePerturbation",  # type: ignore
        X_perturbed: pd.DataFrame,
        col: str,
    ) -> pd.DataFrame:
        """Flip a categorical feature to a different valid category,
        sampled proportionally to its training distribution."""
        X_flipped = X_perturbed.copy()
        valid_values = self.valid_categories[col]
        frequencies = self.categorical_frequencies[col]

        for idx in X_flipped.index:
            current_value = X_flipped.at[idx, col]
            other_values = [v for v in valid_values if v != current_value]

            if not other_values:
                continue

            probs = np.array([frequencies[v] for v in other_values], dtype=float)
            probs /= probs.sum()  # Re-normalise after excluding current value

            X_flipped.at[idx, col] = self.rng.choice(other_values, p=probs)

        return X_flipped
