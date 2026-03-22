import logging

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dataset.dataset import Dataset
from explainer.base_explainer import BaseDatasetExplainer
from explainer.explanation_result import ExplanationResult
from explainer.impute_utils import check_impute_strategy, get_impute_fn
from explainer.registry import EXPLAINERS


@EXPLAINERS.register_module("KernelSHAP")
class KernelShapWrapper(BaseDatasetExplainer):
    def _initialise_explainer(self) -> None:
        params = self.cfg.params or {}

        self.nsamples = params.get("n_samples")
        if self.nsamples is None:
            logging.warning(
                "n_samples not found in params, using default value: 'auto'."
            )
            self.nsamples = "auto"

        background_samples = params.get("background_samples")
        if background_samples is None:
            logging.warning(
                "background_samples not found in params, using default value: 20."
            )
            background_samples = 20

        random_seed = params.get("random_seed")
        if random_seed is None:
            logging.warning("random_seed not found in params, using default value: 42.")
            random_seed = 42

        background_data = self._get_stratified_background_data(
            self.train_dataset, background_samples, random_seed
        )
        background_strategy = params.get("background_strategy", "stratified")
        if background_strategy == "kmeans":
            n_init = params.get("kmeans_n_init", 10)
            max_iter = params.get("kmeans_max_iter", 300)
            impute_strategy = params.get("kmeans_impute_strategy", "median")
            check_impute_strategy(impute_strategy)
            background_data = self._get_kmeans_background_data(
                self.train_dataset,
                background_samples,
                random_seed,
                n_init=n_init,
                max_iter=max_iter,
                impute_strategy=impute_strategy,
            )
        elif background_strategy != "stratified":
            logging.warning(
                "Unknown background_strategy '%s'. Falling back to 'stratified'.",
                background_strategy,
            )

        self.explainer = shap.KernelExplainer(
            self.model_wrapper.predict_proba, background_data
        )

    def explain(self, X: pd.DataFrame) -> ExplanationResult:
        shap_values = self.explainer.shap_values(X, nsamples=self.nsamples)
        base_value = self.explainer.expected_value
        predictions = np.array(self.model_wrapper.predict(X)).ravel()

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        if isinstance(base_value, (list, tuple, np.ndarray)):
            base_value = float(np.array(base_value).flat[1])  # minority class
        else:
            base_value = float(base_value)

        return self.explanation_result(
            shap_values,
            base_value,
            X.index,
            list(X.columns),
            predictions=predictions,
        )

    def plot_local_explanation(self, X):
        shap_values = self.explainer(X)
        shap.plots.bar(shap_values[0])

    def plot_global_explanation(self, X):
        shap_values = self.explainer(X)
        shap.plots.bar(shap_values)

    def _get_stratified_background_data(
        self, dataset: Dataset, n_samples: int, random_seed: int
    ) -> pd.DataFrame:
        np.random.seed(random_seed)
        X = dataset.X_model.copy()
        y = dataset.y.copy()

        majority_class = X[y == 0]
        minority_class = X[y == 1]

        n_minority = min(len(minority_class), n_samples // 2)
        n_majority = n_samples - n_minority

        logging.info(
            f"Stratifying background data: {n_minority} Fraud, {n_majority} Non-Fraud."
        )

        sampled_majority = majority_class.sample(n=n_majority, random_state=random_seed)
        sampled_minority = minority_class.sample(n=n_minority, random_state=random_seed)

        background_data = pd.concat([sampled_minority, sampled_majority]).sample(
            frac=1, random_state=random_seed
        )
        return background_data.reset_index(drop=True)

    def _get_kmeans_background_data(
        self,
        dataset: Dataset,
        n_samples: int,
        random_seed: int,
        n_init: int = 10,
        max_iter: int = 300,
        impute_strategy: str = "median",
    ) -> pd.DataFrame:
        np.random.seed(random_seed)
        X = dataset.X_model.copy()
        y = dataset.y.copy()

        categorical = set(dataset.categorical_features or [])
        numeric_cols = [
            c
            for c in X.columns
            if c not in categorical and pd.api.types.is_numeric_dtype(X[c])
        ]
        if not numeric_cols:
            logging.warning(
                "No numeric columns available for KMeans background. Falling back to stratified sampling."  # noqa: E501
            )
            return self._get_stratified_background_data(dataset, n_samples, random_seed)

        majority_class = X[y == 0]
        minority_class = X[y == 1]

        n_minority = min(len(minority_class), n_samples // 2)
        n_majority = n_samples - n_minority

        logging.info(
            "KMeans background: %s Fraud centroids, %s Non-Fraud centroids.",
            n_minority,
            n_majority,
        )

        selected_rows = []

        def _impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
            X_num = df[numeric_cols].copy()
            if impute_strategy == "none":
                if X_num.isna().any().any():
                    raise ValueError(
                        "kmeans_impute_strategy is 'none' but NaNs were found in numeric columns."  # noqa: E501
                    )
                return X_num
            fill_fn = get_impute_fn(impute_strategy)
            fill_values = X_num.apply(fill_fn)
            return X_num.fillna(fill_values)

        def select_medoids(class_df: pd.DataFrame, k: int, seed: int) -> None:
            if k <= 0 or class_df.empty:
                return
            k = min(k, len(class_df))
            X_num = _impute_numeric(class_df)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_num.values)
            kmeans = KMeans(
                n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter
            )
            labels = kmeans.fit_predict(X_scaled)
            centers = kmeans.cluster_centers_

            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                if not cluster_mask.any():
                    continue
                cluster_points = X_scaled[cluster_mask]
                center = centers[cluster_id]
                dists = np.linalg.norm(cluster_points - center, axis=1)
                local_idx = int(np.argmin(dists))
                row_idx = class_df.index[cluster_mask][local_idx]
                selected_rows.append(X.loc[row_idx])

        select_medoids(minority_class, n_minority, random_seed)
        select_medoids(majority_class, n_majority, random_seed + 1)

        background_data = pd.DataFrame(selected_rows)
        background_data = background_data.sample(frac=1, random_state=random_seed)
        return background_data.reset_index(drop=True)
