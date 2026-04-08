import numpy as np
import pytest
from explainer.explanation import Explanation
from load_config import MetricConfig
from metric.rbo_metric import RankBiasedOverlapMetric


def _make_explanation(instance_id, values, prediction):
    return Explanation(
        instance_id=instance_id,
        values=np.array(values, dtype=float),
        base_value=0.0,
        prediction=prediction,
    )


def test_rbo_metric_expected_value():
    metric = RankBiasedOverlapMetric(
        MetricConfig(name="RankBiasedOverlap", params={"p": 0.5})
    )
    baseline = _make_explanation("a", [3.0, 2.0, 1.0], prediction=1)
    perturbed = _make_explanation("a", [3.0, 2.0, 1.0], prediction=1)

    score = metric.evaluate(baseline, perturbed)
    expected = 1 - (0.5**3)
    assert score == pytest.approx(expected, rel=1e-6)


def test_rbo_metric_invalid_params():
    with pytest.raises(ValueError):
        RankBiasedOverlapMetric(
            MetricConfig(name="RankBiasedOverlap", params={"W": 1.5})
        )
    with pytest.raises(ValueError):
        RankBiasedOverlapMetric(
            MetricConfig(name="RankBiasedOverlap", params={"d": 0})
        )
    with pytest.raises(ValueError):
        RankBiasedOverlapMetric(
            MetricConfig(name="RankBiasedOverlap", params={"p": 2.0})
        )


def test_rbo_metric_shape_mismatch():
    metric = RankBiasedOverlapMetric(
        MetricConfig(name="RankBiasedOverlap", params={"p": 0.5})
    )
    baseline = _make_explanation("a", [1.0, 2.0], prediction=1)
    perturbed = _make_explanation("a", [1.0, 2.0, 3.0], prediction=1)

    with pytest.raises(ValueError):
        metric.evaluate(baseline, perturbed)
