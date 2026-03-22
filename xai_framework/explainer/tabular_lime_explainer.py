import hashlib
import logging
import time

import numpy as np
import pandas as pd
from explainer.base_explainer import BaseDatasetExplainer
from explainer.explanation_result import ExplanationResult
from explainer.impute_utils import check_impute_strategy, get_impute_fn
from explainer.registry import EXPLAINERS
from lime import lime_tabular
from sklearn.preprocessing import LabelEncoder


@EXPLAINERS.register_module("TabularLIME")
class TabularLimeWrapper(BaseDatasetExplainer):
    def _initialise_explainer(self) -> None:
        parameters = self.cfg.params or {}
        params = _validate_params(parameters)

        self.col_order = list(self.train_dataset.X_model.columns)
        self.random_seed = params["random_seed"]
        self.num_samples = params["num_samples"]
        if "categorical_features" in parameters:
            logging.warning(
                "TabularLIME 'categorical_features' in explainer params is ignored. "
                "Using dataset.categorical_features instead."
            )
        self.cat_cols = list(self.train_dataset.categorical_features or [])
        self.strategy_name = params["impute_strategy"]
        self.strategy = get_impute_fn(self.strategy_name)

        if params["auto_detect_binary"]:
            logging.info("Auto-detection of binary features enabled.")
            for col in self.col_order:
                unique_vals = self.train_dataset.X_model[col].dropna().unique()

                # If exactly 2 unique values treat as categorical
                if len(unique_vals) == 2:
                    logging.info(
                        f"Auto-detecting '{col}' as binary categorical (values: {unique_vals})."  # noqa: E501
                    )
                    self.cat_cols.append(col)

        unknown = [c for c in self.cat_cols if c not in self.col_order]
        if unknown:
            raise ValueError(
                f"Categorical features specified in params not found in dataset columns: {unknown}"  # noqa: E501
            )

        self.encoders = {}
        encoded_x = self.train_dataset.X_model.copy()
        categorical_features_indices = []
        categorical_names = {}

        for col in self.cat_cols:
            col_idx = self.col_order.index(col)
            categorical_features_indices.append(col_idx)
            le = LabelEncoder()
            nan_count = encoded_x[col].isna().sum()
            if nan_count > 0:
                if self.strategy_name != "none":
                    logging.warning(
                        f"Column '{col}' has {nan_count} NaN values - encoding as '__missing__'"  # noqa: E501
                    )
                    encoded_x[col] = encoded_x[col].fillna("__missing__").astype(str)
                else:
                    raise ValueError(
                        f"Column '{col}' contains NaNs but impute_strategy is 'none'. Please either fill NaNs in the dataset or choose a different impute_strategy."  # noqa: E501
                    )
            encoded_x[col] = pd.Series(
                le.fit_transform(encoded_x[col]), index=encoded_x.index
            )
            categorical_names[col_idx] = le.classes_.tolist()
            self.encoders[col] = le

        self._num_fill_values = {}  # store fill values for consistent NaN filling
        num_cols = [c for c in self.col_order if c not in self.encoders]
        for col in num_cols:
            if encoded_x[col].isna().any():
                if self.strategy is None:
                    raise ValueError(
                        f"Invalid impute_strategy '{self.strategy_name}' in params."  # noqa: E501
                    )
                fill_val = self.strategy(encoded_x[col])
                self._num_fill_values[col] = fill_val
                logging.warning(
                    f"Column '{col}' contains NaNs — filling with {self.strategy_name} for LIME background."  # noqa: E501
                )
                encoded_x[col] = encoded_x[col].fillna(fill_val)

        if encoded_x.isna().any().any():
            bad_cols = encoded_x.columns[encoded_x.isna().any()].tolist()
            raise ValueError(
                f"NaNs remain in encoded training data after filling. Columns: {bad_cols}"  # noqa: E501
            )

        # Compute base value as mean predicted probability on a background sample
        background = self.train_dataset.X_model.sample(
            n=min(200, len(self.train_dataset.X_model)), random_state=self.random_seed
        )
        self._base_value = float(
            self.model_wrapper.predict_proba(background)[:, 1].mean()
        )

        self.cat_cols = tuple(self.cat_cols)

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=encoded_x.values.astype(float),
            mode="classification",
            feature_names=self.col_order,
            categorical_features=categorical_features_indices,
            categorical_names=categorical_names,
            class_names=params["class_names"],
            kernel_width=params["kernel_width"],
            discretize_continuous=params["discretize_continuous"],
            random_state=self.random_seed,
        )

    def _encode_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode a dataframe for LIME input — shared by explain and plot methods."""
        encoded = X[self.col_order].copy()
        for col, le in self.encoders.items():
            known = set(le.classes_)
            fallback_value = "__missing__" if "__missing__" in known else le.classes_[0]
            encoded[col] = encoded[col].apply(
                lambda v: (
                    fallback_value if pd.isna(v) or str(v) not in known else str(v)
                )
            )
            encoded[col] = le.transform(encoded[col])
        num_cols = [c for c in self.col_order if c not in self.encoders]
        for col in num_cols:
            if encoded[col].isna().any():
                if col not in self._num_fill_values:
                    if self.strategy is None:
                        raise ValueError(
                            f"Invalid impute_strategy '{self.strategy_name}' in params for column '{col}'."  # noqa: E501
                        )
                    fill_val = self.strategy(encoded[col])
                    self._num_fill_values[col] = fill_val
                    logging.warning(
                        f"Column '{col}' has NaNs at explain-time but had none in training."  # noqa: E501
                        f"Filling with {self.strategy_name}."  # noqa: E501
                    )
                else:
                    fill_val = self._num_fill_values[col]
                encoded[col] = encoded[col].fillna(fill_val)
        return encoded

    def _make_predict_fn(self):
        """Decode LIME's encoded array back to raw before calling the black-box model."""  # noqa: E501
        num_cols = [c for c in self.col_order if c not in self.encoders]

        def predict_fn(X_lime: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(X_lime, columns=self.col_order)
            for col, le in self.encoders.items():
                decoded = le.inverse_transform(df[col].astype(int))
                missing_sentinel = (
                    "__missing__" if "__missing__" in le.classes_ else None
                )
                df[col] = [
                    np.nan if (missing_sentinel and v == missing_sentinel) else v
                    for v in decoded
                ]
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return self.model_wrapper.predict_proba(df)

        return predict_fn

    def _reseed_for_instance(self, idx) -> None:
        """Reseed the explainer's random state for reproducibility across runs."""
        idx_str = str(idx).encode("utf-8")
        idx_hash = hashlib.md5(idx_str).hexdigest()
        seed_offset = int(idx_hash[:8], 16)
        instance_seed = (self.random_seed + seed_offset) % (2**32)
        self.explainer.random_state.seed(instance_seed)

    def explain(self, X: pd.DataFrame) -> ExplanationResult:
        original_index = X.index
        predictions = self.model_wrapper.predict(X)
        X_temp = X.copy().reset_index(drop=True)

        # Map positional idx back to original ID for stable seeding
        idx_to_original = {i: original_index[i] for i in range(len(original_index))}

        encoded_sample = self._encode_dataframe(X_temp)

        self._validated_encoded(encoded_sample)

        n_features = len(X_temp.columns)
        predict_fn = self._make_predict_fn()
        all_values = []
        np.random.seed(self.random_seed)  # Ensure reproducibility for sampling in LIME

        explanations = 0
        start_time = time.time()
        logging.info(f"Generating LIME explanations for {len(X)} instances")
        for idx in X_temp.index:
            stop_event = getattr(self, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError(
                    f"Explanation generation interrupted for '{self.cfg.method}'."
                )
            explanations += 1
            self._reseed_for_instance(idx_to_original[idx])

            data_row = encoded_sample.loc[idx].values.astype(float)
            exp = self.explainer.explain_instance(
                data_row=data_row,
                predict_fn=predict_fn,
                num_features=n_features,
                num_samples=self.num_samples,
            )

            available_classes = list(exp.local_exp.keys())
            class_idx = max(available_classes)
            lime_weights = exp.local_exp[class_idx]

            row_weights = np.zeros(n_features)
            for feature_idx, weight in lime_weights:
                row_weights[feature_idx] = weight

            # Warn on flat explanations, indicates model is near-constant for this instance  # noqa: E501
            if np.all(row_weights == 0) or np.all(row_weights == row_weights[0]):
                logging.warning(
                    f"Instance {idx}: LIME returned a constant explanation vector. "
                    "This may indicate the model output is near-constant in this region."  # noqa: E501
                )

            all_values.append(row_weights)
            if explanations % 10 == 0 or explanations == len(X):
                elapsed = time.time() - start_time
                avg_time_per_inst = elapsed / explanations
                eta = avg_time_per_inst * (len(X) - explanations)

                logging.info(
                    f"[{self.cfg.method}] Progress: {explanations}/{len(X)} | "
                    f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                )

        total_time = time.time() - start_time
        logging.info(
            f"Generated {len(X)} explanations for '{self.cfg.method}' "
            f"in {total_time:.1f}s."
        )

        return self.explanation_result(
            values=np.array(all_values),
            base_value=self._base_value,
            index=original_index,
            feature_names=X.columns.tolist(),
            predictions=predictions,
        )

    def plot_local_explanation(self, X):
        encoded_sample = self._encode_dataframe(X)
        self._validated_encoded(encoded_sample)

        predict_fn = self._make_predict_fn()
        self._reseed_for_instance(X.index[0])  # Ensure reproducibility for the plot
        exp = self.explainer.explain_instance(
            data_row=encoded_sample.iloc[0].values.astype(float),
            predict_fn=predict_fn,
            num_features=len(X.columns),
            num_samples=self.num_samples,
        )
        exp.as_pyplot_figure()

    def _validated_encoded(self, encoded_sample: pd.DataFrame) -> None:
        if encoded_sample.isna().any().any():
            bad = encoded_sample.columns[encoded_sample.isna().any()].tolist()
            raise ValueError(f"NaNs remain in encoded sample after filling: {bad}")
        if np.isinf(encoded_sample.values.astype(float)).any():
            raise ValueError("Inf values detected in encoded sample before LIME.")


_OPTIONAL_PARAMS_DEFAULTS = {
    "kernel_width": 0.75,
    "discretize_continuous": False,
    "random_seed": 42,
    "class_names": ["0", "1"],
    "num_samples": 5000,
    "auto_detect_binary": False,
    "impute_strategy": "none",
}


def _validate_params(params: dict) -> dict:
    """
    Validate and resolve params for TabularLIME explainer.
    Raises ValueError if required params is missing or invalid.
    Falls back to defaults for optional params with a warning if not provided.
    """

    resolved = {}

    for key, default in _OPTIONAL_PARAMS_DEFAULTS.items():
        if key not in params or params[key] is None:
            logging.warning(
                f"'{key}' not found in explainer.params. Defaulting to {default}."
            )
            resolved[key] = default
        else:
            if key == "impute_strategy":
                check_impute_strategy(params[key])
            resolved[key] = params[key]

    return resolved
