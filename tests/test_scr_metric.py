import numpy as np
import pytest
from explainer.explanation import Explanation
from load_config import MetricConfig
from metric.scr_metric import SignConsistencyRate


def _make_explanation(instance_id, values, prediction):
    return Explanation(
        instance_id=instance_id,
        values=np.array(values, dtype=float),
        base_value=0.0,
        prediction=prediction,
    )


def test_scr_metric_sign_consistency():
    metric = SignConsistencyRate(
        MetricConfig(name="SignConsistencyRate", params={"epsilon": 1e-6})
    )
    baseline = _make_explanation("a", [0.0, 1e-8, -2.0, 3.0], prediction=1)
    perturbed = _make_explanation("a", [0.0, -1e-8, -2.0, -3.0], prediction=1)

    score = metric.evaluate(baseline, perturbed)
    assert score == pytest.approx(0.75, rel=1e-6)


def test_scr_metric_invalid_epsilon():
    with pytest.raises(ValueError):
        SignConsistencyRate(
            MetricConfig(name="SignConsistencyRate", params={"epsilon": -1.0})
        )
