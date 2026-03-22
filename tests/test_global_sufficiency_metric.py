import numpy as np
import pytest
from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult
from load_config import MetricConfig
from metric.sufficiency_metric import GlobalSufficiencyMetric


def _make_explanation(instance_id, values, prediction):
    return Explanation(
        instance_id=instance_id,
        values=np.array(values, dtype=float),
        base_value=0.0,
        prediction=prediction,
    )


def test_global_sufficiency_metric_value():
    feature_names = ["a", "b", "c"]
    instances = [
        _make_explanation("i1", [5.0, 1.0, 0.0], prediction=1),
        _make_explanation("i2", [4.0, 1.0, 0.0], prediction=1),
        _make_explanation("i3", [0.0, 5.0, 1.0], prediction=0),
        _make_explanation("i4", [0.0, 4.0, 1.0], prediction=1),
    ]
    result = ExplanationResult(
        explainer_name="unit",
        instances=instances,
        base_value=0.0,
        feature_names=feature_names,
        instance_ids=[exp.instance_id for exp in instances],
    )

    sufficiency = GlobalSufficiencyMetric(
        MetricConfig(name="GlobalSufficiencyMetric", params={"k": 1})
    )
    m_s = sufficiency.evaluate(result)
    assert m_s == pytest.approx(0.75, rel=1e-6)


def test_global_sufficiency_empty_result_is_zero():
    result = ExplanationResult(
        explainer_name="unit",
        instances=[],
        base_value=0.0,
        feature_names=[],
        instance_ids=[],
    )
    sufficiency = GlobalSufficiencyMetric(MetricConfig(name="GlobalSufficiencyMetric"))
    assert sufficiency.evaluate(result) == 0.0


def test_global_sufficiency_missing_prediction_raises():
    result = ExplanationResult(
        explainer_name="unit",
        instances=[_make_explanation("i1", [1.0, 0.0], prediction=None)],
        base_value=0.0,
        feature_names=["a", "b"],
        instance_ids=["i1"],
    )
    sufficiency = GlobalSufficiencyMetric(MetricConfig(name="GlobalSufficiencyMetric"))
    with pytest.raises(ValueError):
        sufficiency.evaluate(result)


def test_global_sufficiency_invalid_k():
    with pytest.raises(ValueError):
        GlobalSufficiencyMetric(
            MetricConfig(name="GlobalSufficiencyMetric", params={"k": -1})
        )
