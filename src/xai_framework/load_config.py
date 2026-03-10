import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    train_file_path: str
    test_file_path: str
    target_label: str
    drop_columns: list[str] = field(default_factory=list)
    immutable_features: list[str] = field(default_factory=list)
    metadata: dict | None = None


@dataclass
class ModelConfig:
    architecture: str
    file_path: str
    metadata: dict | None = None


@dataclass
class ExplainerConfig:
    method: str
    params: dict | None = None


@dataclass
class PerturbationConfig:
    name: str
    n_perturbations: int
    random_seed: int | None = None
    params: dict = field(default_factory=dict)


@dataclass
class MetricConfig:
    name: str
    params: dict | None = None


@dataclass
class ExperimentConfig:
    name: str
    type: str
    sample_size: int
    random_seed: int
    sample_group: str = "TP"
    chunk_size: int = 50
    max_workers: int = 1
    resume: bool = True
    checkpoint_dir: str = "results/checkpoints"
    results_dir: str = "results"
    run_id: str | None = None


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    explainer: ExplainerConfig
    perturbation: list[PerturbationConfig]
    metrics: list[MetricConfig]
    experiment: ExperimentConfig


def load_config(config_name: str = "default.yaml") -> Config:
    config_path = "config/" + config_name
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    _validate_paths(data)
    _validate_experiment_config(data)

    return Config(
        dataset=DatasetConfig(**data["dataset"]),
        model=ModelConfig(**data["model"]),
        explainer=ExplainerConfig(**data["explainer"]),
        perturbation=[PerturbationConfig(**p) for p in data["perturbations"]],
        metrics=[MetricConfig(**m) for m in data["metrics"]],
        experiment=ExperimentConfig(**data["experiment"]),
    )


def _validate_paths(data: dict) -> None:
    train_dataset_path = data["dataset"]["train_file_path"]
    test_dataset_path = data["dataset"]["test_file_path"]
    model_path = data["model"]["file_path"]

    if not train_dataset_path:
        raise ValueError(
            "dataset.train_file_path must be set in the configuration file."
        )
    if not test_dataset_path:
        raise ValueError(
            "dataset.test_file_path must be set in the configuration file."
        )
    if not model_path:
        raise ValueError("model.file_path must be set in the configuration file.")
    if not Path(train_dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {train_dataset_path}.")
    if not Path(test_dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {test_dataset_path}.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}.")


def _validate_experiment_config(data: dict) -> None:
    experiment_cfg = data.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        raise ValueError("experiment configuration block must be a dictionary.")

    defaults = {
        "sample_group": "TP",
        "chunk_size": 50,
        "max_workers": 1,
        "resume": True,
        "checkpoint_dir": "results/checkpoints",
        "results_dir": "results",
        "run_id": None,
    }
    for key, default_value in defaults.items():
        if key not in experiment_cfg:
            experiment_cfg[key] = default_value
            logging.info("experiment.%s not set; defaulting to %r", key, default_value)

    experiment_name = experiment_cfg.get("name")

    if not experiment_name:
        raise ValueError("experiment.name must be set in the configuration file.")

    experiment_type = experiment_cfg.get("type")

    if not experiment_type:
        raise ValueError("experiment.type must be set in the configuration file.")

    sample_size = experiment_cfg.get("sample_size")
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError(
            f"experiment.sample_size must be a positive integer. Got: {sample_size!r}."
        )

    random_seed = experiment_cfg.get("random_seed")
    if not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError(
            f"experiment.random_seed must be a non-negative integer. Got: {random_seed!r}."  # noqa: E501
        )

    sample_group = str(experiment_cfg.get("sample_group")).upper()
    if sample_group not in {"TP", "TN", "FP", "FN"}:
        raise ValueError(
            "experiment.sample_group must be one of: TP, TN, FP, FN. "
            f"Got: {sample_group!r}."
        )
    experiment_cfg["sample_group"] = sample_group

    chunk_size = experiment_cfg.get("chunk_size", 50)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(
            f"experiment.chunk_size must be a positive integer. Got: {chunk_size!r}."
        )

    max_workers = experiment_cfg.get("max_workers", 1)
    if not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError(
            f"experiment.max_workers must be a positive integer. Got: {max_workers!r}."
        )

    resume = experiment_cfg.get("resume")
    if not isinstance(resume, bool):
        raise ValueError(
            f"experiment.resume must be a boolean (true/false). Got: {resume!r}."
        )

    checkpoint_dir = experiment_cfg.get("checkpoint_dir", "results/checkpoints")
    results_dir = experiment_cfg.get("results_dir", "results")
    if not isinstance(checkpoint_dir, str) or not checkpoint_dir.strip():
        raise ValueError(
            f"experiment.checkpoint_dir must be a non-empty string. Got: {checkpoint_dir!r}."  # noqa: E501
        )

    if not isinstance(results_dir, str) or not results_dir.strip():
        raise ValueError(
            f"experiment.results_dir must be a non-empty string. Got: {results_dir!r}."  # noqa: E501
        )

    run_id = experiment_cfg.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError(
            f"experiment.run_id must be a non-empty string if provided. Got: {run_id!r}."  # noqa: E501
        )
