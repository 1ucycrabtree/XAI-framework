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
        categorical_features: list[str] | None = None,
        perturbable_categorical_features: list[str] | None = None,
        perturbable_numerical_features: list[str] | None = None,
        integer_features: list[str] | None = None,
        non_negative_features: list[str] | None = None,
        non_negative_prefixes: list[str] | None = None,
    ):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.integer_features = set(integer_features or [])
        self.non_negative_features = set(non_negative_features or [])
        self.non_negative_prefixes = list(non_negative_prefixes or [])

        self._derive_feature_types(
            training_data,
            immutable_features=immutable_features,
            categorical_overrides=categorical_features,
            perturbable_categorical=perturbable_categorical_features,
            perturbable_numerical=perturbable_numerical_features,
        )
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
        self._last_instance_logs: list[dict] = []

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

    def get_last_instance_logs(self) -> list[dict]:
        return list(getattr(self, "_last_instance_logs", []))

    def _record_instance_log(self, payload: dict) -> None:
        if not hasattr(self, "_last_instance_logs"):
            self._last_instance_logs = []
        self._last_instance_logs.append(payload)

    def _get_already_ood_features(self, X: pd.DataFrame) -> set[str]:
        """Identify features in X that are already outside the training distribution."""
        already_ood = set()

        for col in self.continuous_features:
            low = self.feature_stats[col]["q05"]
            high = self.feature_stats[col]["q95"]
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
        """Check continuous features are within 5th and 95th percentile of training data"""  # noqa: E501

        excluded_features = excluded_features or set()
        for col in self.continuous_features:
            if col in excluded_features:
                continue
            low = self.feature_stats[col]["q05"]
            high = self.feature_stats[col]["q95"]
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
        """Fallback: clip continuous features to [q05, q95] bounds.
        Categorical features are left unchanged — invalid categories should
        not arise from well-implemented subclasses.

        # TODO: Consider multivariate manifold enforcement in future work.
        """
        X_clipped = X.copy()
        for col in self.continuous_features:
            low = self.feature_stats[col]["q05"]
            high = self.feature_stats[col]["q95"]
            X_clipped[col] = X_clipped[col].clip(lower=low, upper=high)
        return X_clipped

    def _clip_to_distribution_for(
        self, X: pd.DataFrame, exclude: set[str] | None = None
    ) -> pd.DataFrame:
        """Clip continuous features to [q05, q95] bounds, excluding any in the 'exclude' set."""  # noqa: E501
        exclude = exclude or set()
        X_clipped = X.copy()
        for col in self.continuous_features:
            if col in exclude:
                continue
            low = self.feature_stats[col]["q05"]
            high = self.feature_stats[col]["q95"]
            X_clipped[col] = X_clipped[col].clip(lower=low, upper=high)
        return X_clipped

    def _round_integer_features_for(
        self, X: pd.DataFrame, exclude: set[str] | None = None
    ) -> pd.DataFrame:
        """Round integer-like features to preserve discrete semantics."""
        exclude = exclude or set()
        if not self.integer_features:
            return X
        X_rounded = X.copy()
        for col in self.integer_features:
            if col in exclude or col not in X_rounded.columns:
                continue
            X_rounded[col] = X_rounded[col].round()
        return X_rounded

    def _enforce_non_negative_for(
        self, X: pd.DataFrame, exclude: set[str] | None = None
    ) -> pd.DataFrame:
        """Clamp configured features (or prefixes) to be non-negative."""
        exclude = exclude or set()
        if not self.non_negative_features and not self.non_negative_prefixes:
            return X
        X_clamped = X.copy()

        def matches_prefix(col: str, prefix: str) -> bool:
            if not col.startswith(prefix):
                return False
            if len(col) == len(prefix):
                return True
            next_char = col[len(prefix)]
            # Require a non-letter boundary (digit, underscore, etc.) after prefix
            return not next_char.isalpha()

        cols = set(self.non_negative_features)
        if self.non_negative_prefixes:
            for col in X_clamped.columns:
                if any(matches_prefix(col, p) for p in self.non_negative_prefixes):
                    cols.add(col)
        for col in cols:
            if col in exclude or col not in X_clamped.columns:
                continue
            if not pd.api.types.is_numeric_dtype(X_clamped[col]):
                logging.warning(
                    "Non-negative clamp skipped for non-numeric feature '%s'.",
                    col,
                )
                continue
            X_clamped[col] = X_clamped[col].clip(lower=0)
        return X_clamped

    def _derive_feature_types(
        self,
        training_data: pd.DataFrame,
        immutable_features: list[str] | None = None,
        categorical_overrides: list[str] | None = None,
        perturbable_categorical: list[str] | None = None,
        perturbable_numerical: list[str] | None = None,
    ):
        immutable = set(immutable_features or [])
        categorical_override = set(categorical_overrides or [])
        perturbable_cat = set(perturbable_categorical or [])
        perturbable_num = set(perturbable_numerical or [])
        explicit_perturbable_overrides = bool(perturbable_cat or perturbable_num)

        missing_overrides = categorical_override - set(training_data.columns)
        if missing_overrides:
            logging.warning(
                "Categorical overrides not found in training data and will be ignored: %s",  # noqa: E501
                sorted(missing_overrides),
            )
            categorical_override -= missing_overrides

        missing_perturbable_cat = perturbable_cat - set(training_data.columns)
        if missing_perturbable_cat:
            logging.warning(
                "Perturbable categorical features not found in training data and will be ignored: %s",  # noqa: E501
                sorted(missing_perturbable_cat),
            )
            perturbable_cat -= missing_perturbable_cat

        missing_perturbable_num = perturbable_num - set(training_data.columns)
        if missing_perturbable_num:
            logging.warning(
                "Perturbable numerical features not found in training data and will be ignored: %s",  # noqa: E501
                sorted(missing_perturbable_num),
            )
            perturbable_num -= missing_perturbable_num

        continuous = []
        categorical = []

        for col in training_data.columns:
            if col in categorical_override:
                categorical.append(col)
                continue
            if pd.api.types.is_numeric_dtype(training_data[col]):
                continuous.append(col)
            else:
                categorical.append(col)

        self.immutable_features = list(immutable)
        self.continuous_features = [col for col in continuous if col not in immutable]
        self.categorical_features = [col for col in categorical if col not in immutable]

        if explicit_perturbable_overrides:
            invalid_cat_type = perturbable_cat - set(self.categorical_features)
            if invalid_cat_type:
                logging.warning(
                    "Perturbable categorical overrides include non-categorical/immutable features and will be ignored: %s",  # noqa: E501
                    sorted(invalid_cat_type),
                )
            self.perturbable_categorical_features = sorted(
                perturbable_cat & set(self.categorical_features)
            )
        else:
            self.perturbable_categorical_features = list(self.categorical_features)

        if explicit_perturbable_overrides:
            invalid_num_type = perturbable_num - set(self.continuous_features)
            if invalid_num_type:
                logging.warning(
                    "Perturbable numerical overrides include non-numeric/immutable features and will be ignored: %s",  # noqa: E501
                    sorted(invalid_num_type),
                )
            self.perturbable_continuous_features = sorted(
                perturbable_num & set(self.continuous_features)
            )
        else:
            self.perturbable_continuous_features = list(self.continuous_features)

        # Guard against overlap or accidental immutable leakage.
        overlap = set(self.perturbable_categorical_features) & set(
            self.perturbable_continuous_features
        )
        if overlap:
            logging.warning(
                "Features listed as both perturbable categorical and numerical; numerical assignment will be kept: %s",  # noqa: E501
                sorted(overlap),
            )
            self.perturbable_categorical_features = [
                c for c in self.perturbable_categorical_features if c not in overlap
            ]

        leaked_immutable = set(self.immutable_features) & (
            set(self.perturbable_categorical_features)
            | set(self.perturbable_continuous_features)
        )
        if leaked_immutable:
            logging.warning(
                "Immutable features were requested as perturbable and have been removed: %s",  # noqa: E501
                sorted(leaked_immutable),
            )
            self.perturbable_categorical_features = [
                c
                for c in self.perturbable_categorical_features
                if c not in leaked_immutable
            ]
            self.perturbable_continuous_features = [
                c
                for c in self.perturbable_continuous_features
                if c not in leaked_immutable
            ]

    def _calculate_continuous_feature_stats(self, training_data: pd.DataFrame):
        self.feature_stats = {}
        for col in self.continuous_features:
            self.feature_stats[col] = {
                "min": training_data[col].min(),
                "max": training_data[col].max(),
                "q05": training_data[col].quantile(0.05),
                "q95": training_data[col].quantile(0.95),
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
        idx: int | str | None = None,
    ) -> pd.DataFrame:
        """Flip a categorical feature to a different valid category,
        sampled proportionally to its training distribution.

        If idx is provided, only that row is flipped; otherwise all rows
        in X_perturbed are processed (backwards compatible).
        """
        X_flipped = X_perturbed.copy()
        valid_values = self.valid_categories[col]
        frequencies = self.categorical_frequencies[col]

        indices = [idx] if idx is not None else list(X_flipped.index)
        for row_idx in indices:
            if row_idx not in X_flipped.index:
                continue
            current_value = X_flipped.at[row_idx, col]
            other_values = [v for v in valid_values if v != current_value]

            if not other_values:
                continue

            probs = np.array([frequencies[v] for v in other_values], dtype=float)
            probs /= probs.sum()  # Re-normalise after excluding current value

            X_flipped.at[row_idx, col] = self.rng.choice(other_values, p=probs)

        return X_flipped
