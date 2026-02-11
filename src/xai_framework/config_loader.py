from dataclasses import dataclass

import yaml


@dataclass
class PathsConfig:
    x_test: str
    y_test: str
    model: str


@dataclass
class KernelShapConfig:
    background_samples: int
    random_seed: int


@dataclass
class ExperimentConfig:
    n_perturbations: int
    perturbation_magnitude: float


@dataclass
class AppConfig:
    paths: PathsConfig
    experiment: ExperimentConfig
    kernel_shap: KernelShapConfig


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return AppConfig(
        paths=PathsConfig(**data["paths"]),
        experiment=ExperimentConfig(
            n_perturbations=data["experiment"]["n_perturbations"],
            perturbation_magnitude=data["experiment"]["perturbation_magnitude"],
        ),
        kernel_shap=KernelShapConfig(
            background_samples=data["kernel_shap"]["background_samples"],
            random_seed=data["kernel_shap"]["random_seed"],
        ),
    )
