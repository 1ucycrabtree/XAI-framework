import argparse
import logging

from config_loader import load_config
from dataset.data_loader import get_dataset_loader
from example.noise_perturbation import GaussianNoisePerturbation
from example.perturbation_experiment import PerturbationExperiment
from example.tree_shap_explainer import TreeShapWrapper
from model.model import get_model

# from explainer.kernel_shap_explainer import KernelShapWrapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(description="XAI Robustness Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Name of config YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    loader = get_dataset_loader(cfg.dataset.file_path)

    dataset = loader.load(
        path=cfg.dataset.file_path,
        target_label=cfg.dataset.target_label,
        drop_columns=cfg.dataset.drop_columns,
        metadata=cfg.dataset.metadata,
    )

    fraud_model = get_model(cfg.model.architecture, cfg.model.file_path)

    fraud_model.validate_features(dataset.feature_names)

    tree_shap_wrapper = TreeShapWrapper(fraud_model.model)

    # background_summary = dataset.X.sample(
    #     cfg.kernel_shap.background_samples, random_state=cfg.kernel_shap.random_seed
    # )
    # kernel_shap_wrapper = KernelShapWrapper(
    #    fraud_model.model.predict_proba, background_data=background_summary
    # )

    perturbation_experiment = PerturbationExperiment(
        dataset=dataset,
        model=fraud_model,
        explainer=tree_shap_wrapper,
        perturbation_strategy=GaussianNoisePerturbation(noise_std=0.001),
        sample_size=100,
        random_seed=42,
    )

    pe_result = perturbation_experiment.run()
    pe_result.summary()

    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
