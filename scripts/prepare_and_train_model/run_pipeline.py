import logging
from pathlib import Path

# from catboost import CatBoostClassifier

from generate_metadata import MetadataGenerator
from apply_preprocessing import DataPreprocessor
from train_catboost import CatBoostTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_experiment_pipeline():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    METADATA_DIR = Path(__file__).resolve().parent / "metadata"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "data" / "models"

    logging.info(
        "Step 1: Identifying data quality issues and generating cleaning rules..."
    )
    gen = MetadataGenerator()
    gen.run(RAW_DATA_DIR, METADATA_DIR)

    metadata_files = sorted(METADATA_DIR.glob("metadata_*.json"))
    if not metadata_files:
        raise FileNotFoundError("Metadata generation failed to produce a JSON file.")
    latest_metadata = metadata_files[-1]
    logging.info(f"Using latest metadata: {latest_metadata.name}")

    logging.info("Step 2: Applying preprocessing rules and preparing datasets...")
    preprocessor = DataPreprocessor(str(latest_metadata))
    preprocessor.process(RAW_DATA_DIR, PROCESSED_DIR)

    logging.info("Step 3: Training and evaluating the CatBoost model...")
    trainer = CatBoostTrainer(
        train_val_path=PROCESSED_DIR / "train_val_final.parquet",
        test_path=PROCESSED_DIR / "test_final.parquet",
        metadata_path=PROCESSED_DIR / "processed_metadata.json",
    )

    trainer.train(MODELS_DIR)

    # Can load a pre-trained model for evaluation instead of training from scratch:
    # model_path = MODELS_DIR / "catboost_amd_depth8_lr0.01.cbm"
    # trainer.model = CatBoostClassifier().load_model(model_path)

    trainer.evaluate()

    logging.info(
        "Model training and evaluation complete. All outputs saved to data/processed and data/models."
    )


if __name__ == "__main__":
    run_experiment_pipeline()
