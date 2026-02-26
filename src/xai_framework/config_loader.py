from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    file_path: str
    target_label: str
    drop_columns: list[str] = field(default_factory=list)
    metadata: dict | None = None


@dataclass
class ModelConfig:
    architecture: str
    file_path: str
    metadata: dict | None = None


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
    dataset: DatasetConfig
    model: ModelConfig
    experiment: ExperimentConfig
    kernel_shap: KernelShapConfig


def load_config(config_name: str = "default.yaml") -> AppConfig:
    config_path = "config/" + config_name
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    _validate_paths(data)

    return AppConfig(
        dataset=DatasetConfig(**data["dataset"]),
        model=ModelConfig(**data["model"]),
        experiment=ExperimentConfig(**data["experiment"]),
        kernel_shap=KernelShapConfig(**data["kernel_shap"]),
    )


def _validate_paths(data: dict) -> None:
    dataset_path = data["dataset"]["file_path"]
    model_path = data["model"]["file_path"]

    if not dataset_path:
        raise ValueError("dataset.file_path must be set in the configuration file.")
    if not model_path:
        raise ValueError("model.file_path must be set in the configuration file.")
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}.")
