import pandas as pd

from load_config import PerturbationConfig
from perturbation.base_perturbation import BasePerturbation
from utils.registry import Registry

PERTURBATIONS = Registry("perturbation")


def get_perturbation(
    cfg: PerturbationConfig,
    training_data: pd.DataFrame,
    immutable_features: list[str] | None = None,
) -> BasePerturbation:
    perturbation_cls = PERTURBATIONS.get(cfg.name)
    return perturbation_cls(cfg, training_data, immutable_features)
