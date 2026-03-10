import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone

from dataset.registry import get_dataset
from experiment.registry import get_experiment
from explainer.registry import get_explainer
from load_config import load_config
from metric.registry import get_metric
from model.registry import get_model
from perturbation.registry import get_perturbation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s| %(levelname)s| %(threadName)s: %(message)s",
    datefmt="%H:%M:%S",
)

logging.getLogger("shap").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="XAI Robustness Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Name of config YAML file",
    )
    return parser.parse_args()


def _default_run_id(experiment_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", experiment_name).strip("_")
    if not safe_name:
        safe_name = "experiment"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_run_{timestamp}"


def main():
    logging.info("Starting XAI Robustness Framework...")
    args = parse_args()
    logging.info("Using config file: %s", args.config)
    cfg = load_config(args.config)
    if not cfg.experiment.run_id:
        cfg.experiment.run_id = _default_run_id(cfg.experiment.name)
    logging.info(
        "Resolved run_id=%s (resume=%s, sample_group=%s, chunk_size=%s, max_workers=%s)",  # noqa: E501
        cfg.experiment.run_id,
        cfg.experiment.resume,
        cfg.experiment.sample_group,
        cfg.experiment.chunk_size,
        cfg.experiment.max_workers,
    )

    datasets = get_dataset(cfg.dataset)
    training_data, test_data = datasets.train, datasets.test

    fraud_model = get_model(cfg.model)
    fraud_model.validate_features(test_data.model_feature_names)

    if cfg.explainer.params is None:
        cfg.explainer.params = {}
    if cfg.explainer.params.get("random_seed") is None:
        cfg.explainer.params["random_seed"] = cfg.experiment.random_seed

    explanation_method = (
        get_explainer(cfg.explainer, fraud_model, test_data)
        if cfg.experiment.max_workers == 1
        else None
    )
    metrics = [get_metric(metric_cfg) for metric_cfg in cfg.metrics]
    cfg_hash = hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    for perturb_cfg in cfg.perturbation:
        if perturb_cfg.random_seed is None:
            perturb_cfg.random_seed = cfg.experiment.random_seed

        logging.info(f"Running perturbation experiment: {perturb_cfg.name}...")
        perturbation = get_perturbation(
            perturb_cfg, training_data.X_model, cfg.dataset.immutable_features
        )

        def build_explainer_for_chunk(chunk_id: int):
            if cfg.experiment.max_workers == 1:
                return explanation_method

            explainer_params = dict(cfg.explainer.params or {})
            explainer_cfg = replace(cfg.explainer, params=explainer_params)
            return get_explainer(explainer_cfg, fraud_model, test_data)

        def build_perturbation_for_chunk(chunk_id: int):
            if cfg.experiment.max_workers == 1:
                return perturbation

            base_seed = (
                perturb_cfg.random_seed
                if perturb_cfg.random_seed is not None
                else cfg.experiment.random_seed
            )
            perturb_cfg_for_chunk = replace(
                perturb_cfg,
                params=dict(perturb_cfg.params),
                random_seed=base_seed + chunk_id,
            )
            return get_perturbation(
                perturb_cfg_for_chunk,
                training_data.X_model,
                cfg.dataset.immutable_features,
            )

        experiment = get_experiment(
            cfg.experiment,
            dataset=test_data,
            model=fraud_model,
            explainer_method_name=cfg.explainer.method,
            perturbation_name=perturb_cfg.name,
            metrics=metrics,
            sample_size=cfg.experiment.sample_size,
            random_seed=cfg.experiment.random_seed,
            config_hash=cfg_hash,
            explainer_factory=build_explainer_for_chunk,
            perturbation_factory=build_perturbation_for_chunk,
        )

        result = experiment.run()
        result.summary()

    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Execution interrupted by user.")
        sys.exit(130)
