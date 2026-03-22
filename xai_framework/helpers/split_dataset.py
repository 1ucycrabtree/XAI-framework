"""
split_dataset.py

Splits a cleaned parquet dataset into train, val, and test sets using
a time-ordered split (no shuffling) — consistent with notebook preprocessing.

Usage:
    python split_dataset.py --input <path_to_parquet> [--output-dir <dir>]

Defaults:
    --input      code/data/processed/cleaned/dataset_2026-01-26_20:40:38.parquet
    --output-dir code/data/processed/splits/
    --train      0.8
    --val        0.9  (i.e. 10% val, 10% test)
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

TARGET_LABEL = "isFraud"
EXCLUDE_COLS = ["isFraud", "TransactionDT", "TransactionID"]

DEFAULT_INPUT = "code/data/processed/cleaned/dataset_2026-01-26_20:40:38.parquet"
DEFAULT_OUTPUT_DIR = "code/data/processed/splits"
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test.")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to the cleaned parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write split parquet files.",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help="Proportion of data for training (default: 0.8).",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help="Cumulative proportion for train+val boundary (default: 0.9).",
    )
    return parser.parse_args()


def split_dataset(
    df: pd.DataFrame,
    train_split: float,
    val_split: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split — no shuffling.
    Mirrors the notebook's sequential split logic.
    """
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * val_split)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def log_split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    total = len(train) + len(val) + len(test)
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        fraud_rate = split[TARGET_LABEL].mean() * 100
        logging.info(
            f"{name:>5}: {len(split):>7,} rows "
            f"({len(split) / total * 100:.1f}%) | "
            f"fraud rate: {fraud_rate:.3f}%"
        )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a .parquet file, got: {input_path.suffix}")
    if not (0 < args.train < args.val < 1.0):
        raise ValueError(
            f"Split proportions must satisfy 0 < train ({args.train}) "
            f"< val ({args.val}) < 1.0"
        )

    logging.info(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    logging.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    if TARGET_LABEL not in df.columns:
        raise ValueError(
            f"Target label '{TARGET_LABEL}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    train, val, test = split_dataset(df, args.train, args.val)

    logging.info("Split summary:")
    log_split_summary(train, val, test)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    splits = {"train": train, "val": val, "test": test}
    for name, split in splits.items():
        out_path = output_dir / f"{name}_dataset_{timestamp}.parquet"
        split.to_parquet(out_path, index=True)
        logging.info(f"Saved {name} split to {out_path}")


if __name__ == "__main__":
    main()
