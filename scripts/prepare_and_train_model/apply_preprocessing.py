import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessor:
    def __init__(self, metadata_path: str):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        all_dropped = self.metadata.get("all_dropped_cols")
        if all_dropped is None:
            all_dropped = self.metadata.get("sparse_cols", []) + self.metadata.get(
                "redundant_cols", []
            )

        self.raw_time_col = "TransactionDT"
        self.cols_to_drop = sorted(set(all_dropped + [self.raw_time_col]))
        self.id_cols = ["TransactionID"]
        self.target = "isFraud"

    def process(self, raw_path: Path, output_path: Path) -> None:
        """Apply preprocessing rules to the raw dataset and save the processed version."""
        logging.info(f"Loading and merging datasets from {raw_path}...")
        trans = pd.read_csv(raw_path / "train_transaction.csv")
        ident = pd.read_csv(raw_path / "train_identity.csv")
        df = pd.merge(
            trans, ident, on="TransactionID", how="left", validate="one_to_one"
        )
        del trans, ident

        logging.info("Applying temporal split (80% training)...")
        df = df.sort_values("TransactionDT")
        df = self._add_stable_time_features(df)
        n = len(df)
        train_idx, val_idx = int(n * 0.8), int(n * 0.9)

        splits = {
            "train": df.iloc[:train_idx].copy(),
            "val": df.iloc[train_idx:val_idx].copy(),
            "test": df.iloc[val_idx:].copy(),
        }

        # Identify categorical columns
        train_df_raw = splits["train"]
        self.categorical_cols = [
            c
            for c in train_df_raw.columns
            if not pd.api.types.is_numeric_dtype(train_df_raw[c])
            and c not in self.id_cols + [self.target] + self.cols_to_drop
        ]

        processed_splits = {}
        for name, split_df in splits.items():
            logging.info(f"Processing {name} split with {len(split_df)} rows...")
            split_df = self._apply_rules(split_df)
            split_df = self._impute_categorical(split_df)

            # Make sure the target column is of type int for consistency
            if "isFraud" in split_df.columns:
                split_df["isFraud"] = split_df["isFraud"].astype(int)

            processed_splits[name] = split_df

        train_df = processed_splits["train"]
        counts = train_df[self.target].value_counts()

        final_features = [
            c for c in train_df.columns if c not in self.id_cols + [self.target]
        ]
        categorical_cols = [
            c for c in final_features if not pd.api.types.is_numeric_dtype(train_df[c])
        ]
        numerical_cols = [c for c in final_features if c not in categorical_cols]

        self.metadata.update(
            {
                "target": self.target,
                "id_cols": self.id_cols,
                "engineered_time_features": [
                    "time_hour_of_day",
                    "time_day_of_week",
                    "time_is_weekend",
                    "time_hour_sin",
                    "time_hour_cos",
                    "time_dow_sin",
                    "time_dow_cos",
                ],
                "final_features": final_features,
                "categorical_cols": categorical_cols,
                "numerical_cols": numerical_cols,
                "scale_pos_weight": float(counts[0] / counts[1])
                if counts[1] > 0
                else 1.0,
                "feature_count": len(final_features),
                "train_row_count": len(train_df),
                "val_row_count": len(processed_splits["val"]),
                "test_row_count": len(processed_splits["test"]),
            }
        )

        train_val_df = pd.concat(
            [processed_splits["train"], processed_splits["val"]], axis=0
        )
        self._save_outputs(train_val_df, processed_splits["test"], output_path)
        logging.info("Data preprocessing complete.")

    def _save_outputs(
        self, train_val_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)

        train_val_df.to_parquet(output_path / "train_val_final.parquet", index=False)
        test_df.to_parquet(output_path / "test_final.parquet", index=False)
        logging.info("Saved train_val_final.parquet and test_final.parquet")

        meta_file = output_path / "processed_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(self.metadata, f, indent=4)
        logging.info(f"Saved metadata to {meta_file}")

        split_file = output_path / "feature_splits.json"
        feature_split_summary = {
            "categorical_cols": self.metadata["categorical_cols"],
            "numerical_cols": self.metadata["numerical_cols"],
            "counts": {
                "categorical": len(self.metadata["categorical_cols"]),
                "numerical": len(self.metadata["numerical_cols"]),
                "total_final_features": self.metadata["feature_count"],
            },
        }
        with open(split_file, "w") as f:
            json.dump(feature_split_summary, f, indent=4)
        logging.info(f"Saved feature split summary to {split_file}")

    def _apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the identified preprocessing rules to the given DataFrame."""
        logging.info(f"Dropping {len(self.cols_to_drop)} columns...")
        return df.drop(columns=self.cols_to_drop, errors="ignore")

    def _add_stable_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build stable, coarse-grained time features from TransactionDT and keep raw timestamp out of training."""
        if self.raw_time_col not in df.columns:
            return df

        seconds = pd.to_numeric(df[self.raw_time_col], errors="coerce").fillna(0)
        hour_index = (seconds // 3600).astype("int64")
        day_index = (seconds // 86400).astype("int64")
        day_of_week = (day_index % 7).astype("int64")
        hour_of_day = (hour_index % 24).astype("int64")

        # Build engineered columns in one block and concatenate once to avoid
        # DataFrame fragmentation from repeated insert operations.
        engineered = pd.DataFrame(
            {
                "time_hour_of_day": hour_of_day,
                "time_day_of_week": day_of_week,
                "time_is_weekend": day_of_week.isin([5, 6]).astype("int64"),
                # Cyclical encodings preserve circular structure (e.g. 23h adjacent to 0h).
                "time_hour_sin": np.sin((2 * np.pi * hour_of_day) / 24.0),
                "time_hour_cos": np.cos((2 * np.pi * hour_of_day) / 24.0),
                "time_dow_sin": np.sin((2 * np.pi * day_of_week) / 7.0),
                "time_dow_cos": np.cos((2 * np.pi * day_of_week) / 7.0),
            },
            index=df.index,
        )
        return pd.concat([df, engineered], axis=1)

    def _impute_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in categorical columns with "missing" to avoid type mixing and leverage CatBoost's native encoding of categories."""
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("missing").astype(str)
        return df
