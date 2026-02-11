import shap
import pandas as pd
import numpy as np
from explainer.explainer import BaseExplainer
from explainer.explanation_result import ExplanationResult


class KernelShapWrapper(BaseExplainer):
    def __init__(self, predict_func, background_data=None, nsamples="auto"):
        self.explainer = shap.KernelExplainer(
            predict_func, background_data, link="logit"
        )
        self.nsamples = nsamples

    def explain(self, X: pd.DataFrame) -> ExplanationResult:
        shap_values = self.explainer.shap_values(X, nsamples=self.nsamples)
        base_value = self.explainer.expected_value

        if isinstance(shap_values, list):
            shap_values = shap_values[1]


        # For binary classification, select the base value for the positive class (index 1)
        if isinstance(base_value, (list, tuple, np.ndarray)):
            if len(base_value) == 2:
                base_value = float(base_value[1])
            else:
                base_value = float(base_value[0])
        else:
            base_value = float(base_value)

        return self.explanation_result(
            shap_values, base_value, X.index, list(X.columns)
        )

    def plot_local_explanation(self, X):
        shap_values = self.explainer(X)
        shap.plots.bar(shap_values[0])

    def plot_global_explanation(self, X):
        shap_values = self.explainer(X)
        shap.plots.bar(shap_values)
