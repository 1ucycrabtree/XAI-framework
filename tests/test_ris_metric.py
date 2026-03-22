import numpy as np
import pandas as pd
import pytest

from explainer.explanation import Explanation
from load_config import MetricConfig
from metric.ris_metric import RelativeInputStabilityMetric


def _make_explanation(instance_id, values, prediction):
    return Explanation(
        instance_id=instance_id,
        values=np.array(values, dtype=float),
        base_value=0.0,
        prediction=prediction,
    )


def test_ris_metric_basic():
    metric = RelativeInputStabilityMetric(MetricConfig(name="RelativeInputStability"))
    baseline = _make_explanation("a", [1.0, 2.0], prediction=1)
    perturbed = _make_explanation("a", [2.0, 4.0], prediction=1)

    base_input = pd.Series({"f1": 0.0, "f2": 0.0, "cat": "A"})
    pert_input = pd.Series({"f1": 1.0, "f2": 0.0, "cat": "B"})
    ranges = {"f1": (0.0, 10.0), "f2": (0.0, 10.0)}

    score = metric.evaluate(
        baseline,
        perturbed,
        baseline_input=base_input,
        perturbed_input=pert_input,
        continuous_ranges=ranges,
    )

    d_exp = np.linalg.norm(np.array([1.0, 2.0]) - np.array([2.0, 4.0]), ord=2)
    d_inp = (abs(0.0 - 1.0) / 10.0 + 1.0) / 3.0
    expected = d_exp / d_inp
    assert score == pytest.approx(expected, rel=1e-6)


def test_ris_metric_small_input_distance_returns_zero():
    metric = RelativeInputStabilityMetric(
        MetricConfig(name="RelativeInputStability", params={"epsilon": 0.5})
    )
    baseline = _make_explanation("a", [0.0, 0.0], prediction=1)
    perturbed = _make_explanation("a", [1.0, 1.0], prediction=1)

    base_input = pd.Series({"f1": 0.0})
    pert_input = pd.Series({"f1": 0.1})
    ranges = {"f1": (0.0, 10.0)}

    score = metric.evaluate(
        baseline,
        perturbed,
        baseline_input=base_input,
        perturbed_input=pert_input,
        continuous_ranges=ranges,
    )
    assert score == 0.0


def test_ris_metric_invalid_epsilon():
    with pytest.raises(ValueError):
        RelativeInputStabilityMetric(
            MetricConfig(name="RelativeInputStability", params={"epsilon": -1.0})
        )
