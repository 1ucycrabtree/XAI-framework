import logging

from explainer.tree_shap_explainer import TreeShapWrapper
# from explainer.kernel_shap_explainer import KernelShapWrapper
from model.catboost_model import CatBoostFraudModel
from dataset.parquet_loader import ParquetDataLoader
from experiment.perturbation_experiment import PerturbationExperiment
from perturbation.noise_perturbation import GaussianNoisePerturbation
from config_loader import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s"
)


def main():
    cfg = load_config()
    parquet_loader = ParquetDataLoader()
    metadata = {
        "dataset": "IEEE-CIS-Fraud-Detection",
        "type": "test",
        "state": "feature_engineered",
    }

    dataset = parquet_loader.load(
        cfg.paths.x_test, cfg.paths.y_test, target_label="is_fraud", metadata=metadata
    )

    fraud_model = CatBoostFraudModel(cfg.paths.model)

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
