import logging

import numpy as np
import pandas as pd
import shap

from dataset.dataset import Dataset
from explainer.base_explainer import BaseDatasetExplainer
from explainer.explanation_result import ExplanationResult
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
