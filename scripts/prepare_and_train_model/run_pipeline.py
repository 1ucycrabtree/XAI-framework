import argparse
import logging
from pathlib import Path

# from catboost import CatBoostClassifier
from apply_preprocessing import DataPreprocessor
from generate_metadata import MetadataGenerator
from train_catboost import CatBoostTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing + CatBoost training pipeline"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter grid tuning. If omitted, uses lr=0.01 depth=10.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help=(
            "Skip metadata generation and preprocessing, and train directly from "
            "data/processed train_val/test parquet + processed_metadata.json"
        ),
    )
    return parser.parse_args()


def run_experiment_pipeline(tune: bool = False, skip_preprocessing: bool = False):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    METADATA_DIR = Path(__file__).resolve().parent / "metadata"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "data" / "models"

    required_processed = [
        PROCESSED_DIR / "train_val_final.parquet",
        PROCESSED_DIR / "test_final.parquet",
        PROCESSED_DIR / "processed_metadata.json",
    ]

    if skip_preprocessing:
        missing = [str(p) for p in required_processed if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "--skip-preprocessing was set but processed inputs are missing: "
                + ", ".join(missing)
            )
        logging.info(
            "Skipping metadata generation and preprocessing. "
            "Using existing files under data/processed/."
        )
    else:
        logging.info(
            "Step 1: Identifying data quality issues and generating cleaning rules..."
        )
        gen = MetadataGenerator()
        gen.run(RAW_DATA_DIR, METADATA_DIR)

        metadata_files = sorted(METADATA_DIR.glob("metadata_*.json"))
        if not metadata_files:
            raise FileNotFoundError(
                "Metadata generation failed to produce a JSON file."
            )
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
        tune=tune,
    )

    trainer.train(MODELS_DIR)

    # Can load a pre-trained model for evaluation instead of training from scratch:
    # model_path = MODELS_DIR / "catboost_amd_depth8_lr0.01.cbm"
    # trainer.model = CatBoostClassifier().load_model(model_path)

    trainer.evaluate()

    logging.info(
        "Model training and evaluation complete. "
        "All outputs saved to data/processed and data/models."
    )


if __name__ == "__main__":
    args = parse_args()
    run_experiment_pipeline(
        tune=args.tune,
        skip_preprocessing=args.skip_preprocessing,
    )
