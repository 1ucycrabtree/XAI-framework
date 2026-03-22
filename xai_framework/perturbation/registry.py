import pandas as pd
from load_config import PerturbationConfig
from perturbation.base_perturbation import BasePerturbation
from utils.registry import Registry

PERTURBATIONS = Registry("perturbation")


def get_perturbation(
    cfg: PerturbationConfig,
    training_data: pd.DataFrame,
    immutable_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    perturbable_categorical_features: list[str] | None = None,
    perturbable_numerical_features: list[str] | None = None,
    integer_features: list[str] | None = None,
    non_negative_features: list[str] | None = None,
    non_negative_prefixes: list[str] | None = None,
) -> BasePerturbation:
    perturbation_cls = PERTURBATIONS.get(cfg.name)
    return perturbation_cls(
        cfg,
        training_data,
        immutable_features=immutable_features,
        categorical_features=categorical_features,
        perturbable_categorical_features=perturbable_categorical_features,
        perturbable_numerical_features=perturbable_numerical_features,
        integer_features=integer_features,
        non_negative_features=non_negative_features,
        non_negative_prefixes=non_negative_prefixes,
    )
