from experiment.base_experiment import BaseExperiment
from load_config import ExperimentConfig
from utils.registry import Registry

EXPERIMENTS = Registry("experiment")


def get_experiment(cfg: ExperimentConfig, **kwargs) -> BaseExperiment:
    experiment_cls = EXPERIMENTS.get(cfg.type)
    return experiment_cls(cfg=cfg, **kwargs)
