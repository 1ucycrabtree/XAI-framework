from typing import Union

from load_config import MetricConfig
from metric.base_metric import BaseGlobalMetric, BaseLocalMetric
from utils.registry import Registry

METRICS = Registry("metric")


def get_metric(cfg: MetricConfig) -> Union[BaseLocalMetric, BaseGlobalMetric]:
    metric_cls = METRICS.get(cfg.name)
    return metric_cls(cfg)
