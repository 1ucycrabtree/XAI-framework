import pandas as pd
import shap
from explainer.base_explainer import BaseExplainer
from explainer.explanation_result import ExplanationResult
from explainer.registry import EXPLAINERS


@EXPLAINERS.register_module("TreeSHAP")
class TreeShapWrapper(BaseExplainer):
    def _initialise_explainer(self) -> None:
        self.explainer = shap.TreeExplainer(self.model_wrapper.model)

    def explain(self, X: pd.DataFrame) -> ExplanationResult:
        shap_values = self.explainer.shap_values(X)
        base_value = self.explainer.expected_value
        predictions = self.model_wrapper.predict(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if isinstance(base_value, (list, tuple)):
            base_value = base_value[1]

        base_value = float(base_value)  # type: ignore

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
