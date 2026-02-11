import shap
import pandas as pd
from explainer.explanation_result import ExplanationResult
from explainer.explainer import BaseExplainer


class TreeShapWrapper(BaseExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame) -> ExplanationResult:
        shap_values = self.explainer.shap_values(X)
        base_value = self.explainer.expected_value

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if isinstance(base_value, (list, tuple)):
            base_value = base_value[1]

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
