import pytest

from load_config import MetricConfig
from metric.base_metric import BaseGlobalMetric, BaseLocalMetric


def test_base_local_metric_is_abstract():
    with pytest.raises(TypeError):
        BaseLocalMetric(MetricConfig(name="BaseLocalMetric")) # type: ignore


def test_base_global_metric_is_abstract():
    with pytest.raises(TypeError):
        BaseGlobalMetric(MetricConfig(name="BaseGlobalMetric")) # type: ignore
