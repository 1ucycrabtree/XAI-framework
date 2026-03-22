import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate CatBoost model on processed datasets"
    )
    parser.add_argument(
        "--train-val-path",
        type=str,
        default=str(REPO_ROOT / "data" / "processed" / "train_val_final.parquet"),
        help="Path to train+validation parquet file",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(REPO_ROOT / "data" / "processed" / "test_final.parquet"),
        help="Path to test parquet file",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(REPO_ROOT / "data" / "processed" / "processed_metadata.json"),
        help="Path to processed metadata JSON",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "models"),
        help="Directory where trained model artifact is saved",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable LR/depth grid search; if omitted, use lr=0.01 and depth=10",
    )
    return parser.parse_args()


class CatBoostTrainer:
    # Addr1 and Addr2 are numeric but represent location categories. Treat them as categorical to preserve their semantic meaning and leverage CatBoost's handling of categorical features.  # noqa: E501
    FORCED_CATEGORICAL_COLS = {"addr1", "addr2"}

    def __init__(
        self,
        train_val_path: Path,
        test_path: Path,
        metadata_path: Path,
        tune: bool = False,
    ):
        self._load_files(train_val_path, test_path, metadata_path)
        self.output_dir = REPO_ROOT / "data" / "models" / "evaluation_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tune = tune

        self.target = "isFraud"
        self.id_cols = self.metadata.get("id_cols", ["TransactionID"])
        self.drop_cols = self.id_cols + [self.target]

        # Define the hyperparameter tuning grid
        # (If don't want to tune, just set 1 value per hyperparameter)
        self.tuning_grid = {
            "learning_rate": [0.01, 0.03],
            "depth": [6, 8, 10],
        }
        self.selected_hyperparameters: dict[str, float | int | str] = {}
        self.tuning_results: list[dict] = []

        self._prepare_datasets()

    def _train_single_model(
        self, learning_rate: float = 0.01, depth: int = 10
    ) -> tuple[CatBoostClassifier, dict]:
        logging.info(
            "Training single CatBoost model with fixed params: "
            "learning_rate=%s, depth=%s",
            learning_rate,
            depth,
        )
        model = self._build_model(learning_rate=learning_rate, depth=depth)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=(self.X_val, self.y_val),
            cat_features=self.cat_feature_indices,
            early_stopping_rounds=100,
            plot=False,
        )

        val_probs = model.predict_proba(self.X_val)[:, 1]
        best_threshold, best_f2, _, _ = self._best_f2_threshold(self.y_val, val_probs)
        result = {
            "learning_rate": learning_rate,
            "depth": depth,
            "best_iteration": int(model.get_best_iteration()),
            "validation_AUROC": float(roc_auc_score(self.y_val, val_probs)),
            "validation_AUPRC": float(average_precision_score(self.y_val, val_probs)),
            "validation_best_F2": float(best_f2),
            "validation_best_threshold": float(best_threshold),
        }

        self.tuning_results = [result]
        self.selected_hyperparameters = {
            "learning_rate": learning_rate,
            "depth": depth,
            "early_stopping_rounds": 100,
            "auto_class_weights": "SqrtBalanced",
            "selection_metric": "Fixed hyperparameters (no tuning)",
        }
        return model, result

    def _build_model(self, learning_rate: float, depth: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=20000,  # super high limit so early stopping can work
            learning_rate=learning_rate,
            depth=depth,
            loss_function="Logloss",
            eval_metric="AUC",
            custom_metric=["F1", "Precision", "Recall", "PRAUC"],
            random_seed=42,
            task_type="CPU",
            subsample=0.66,
            thread_count=-1,
            bootstrap_type="Bernoulli",
            verbose=50,
            use_best_model=True,
            auto_class_weights="SqrtBalanced",
        )

    def _best_f2_threshold(
        self, y_true: pd.Series, y_scores: np.ndarray
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        if len(thresholds) == 0:
            return 0.5, 0.0, thresholds, np.array([])

        f2_scores = (5 * precisions[:-1] * recalls[:-1]) / (
            (4 * precisions[:-1]) + recalls[:-1] + 1e-12
        )
        best_idx = int(np.argmax(f2_scores))
        return (
            float(thresholds[best_idx]),
            float(f2_scores[best_idx]),
            thresholds,
            f2_scores,
        )

    def _run_hyperparameter_tuning(self) -> tuple[CatBoostClassifier, dict]:
        best_model = None
        best_result = None

        logging.info(
            "Starting validation-based tuning over learning rate and tree depth"
        )
        for lr in self.tuning_grid["learning_rate"]:
            for depth in self.tuning_grid["depth"]:
                logging.info("Tuning trial with learning_rate=%s, depth=%s", lr, depth)
                model = self._build_model(learning_rate=lr, depth=depth)
                model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=(self.X_val, self.y_val),
                    cat_features=self.cat_feature_indices,
                    early_stopping_rounds=100,
                    plot=False,
                )

                val_probs = model.predict_proba(self.X_val)[:, 1]
                best_threshold, best_f2, _, _ = self._best_f2_threshold(
                    self.y_val, val_probs
                )

                result = {
                    "learning_rate": lr,
                    "depth": depth,
                    "best_iteration": int(model.get_best_iteration()),
                    "validation_AUROC": float(roc_auc_score(self.y_val, val_probs)),
                    "validation_AUPRC": float(
                        average_precision_score(self.y_val, val_probs)
                    ),
                    "validation_best_F2": float(best_f2),
                    "validation_best_threshold": float(best_threshold),
                }
                self.tuning_results.append(result)

                if best_result is None:
                    best_model = model
                    best_result = result
                else:
                    is_better = (
                        result["validation_best_F2"] > best_result["validation_best_F2"]
                    ) or (
                        np.isclose(
                            result["validation_best_F2"],
                            best_result["validation_best_F2"],
                        )
                        and result["validation_AUPRC"] > best_result["validation_AUPRC"]
                    )
                    if is_better:
                        best_model = model
                        best_result = result

        if best_model is None or best_result is None:
            raise RuntimeError("Hyperparameter tuning produced no valid model.")

        self.selected_hyperparameters = {
            "learning_rate": best_result["learning_rate"],
            "depth": best_result["depth"],
            "early_stopping_rounds": 100,
            "auto_class_weights": "SqrtBalanced",
            "selection_metric": "Validation F2 (threshold optimized on validation)",
        }

        logging.info(
            "Selected model: lr=%s, depth=%s, val_F2=%.4f, val_AUPRC=%.4f",
            best_result["learning_rate"],
            best_result["depth"],
            best_result["validation_best_F2"],
            best_result["validation_AUPRC"],
        )
        return best_model, best_result

    def train(self, output_path: Path) -> None:
        if self.tune:
            logging.info("Initialising and tuning CatBoostClassifier...")
            model, best_result = self._run_hyperparameter_tuning()
        else:
            logging.info(
                "Initialising CatBoostClassifier without hyperparameter tuning..."
            )
            model, best_result = self._train_single_model(learning_rate=0.01, depth=10)

        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / (
            f"catboost_fraud_model_lr{best_result['learning_rate']}"
            f"_depth{best_result['depth']}.cbm"
        )

        logging.info(
            "Saving model to %s (best iteration: %s)",
            model_path,
            model.get_best_iteration(),
        )
        model.save_model(str(model_path))

        self.model = model
        self.model_path = str(model_path)

    def evaluate(self) -> None:
        logging.info("Starting model evaluation...")
        val_probs = self.model.predict_proba(self.X_val)[:, 1]
        test_probs = self.model.predict_proba(self.X_test)[:, 1]

        best_threshold, best_val_f2, threshold_grid, val_f2_grid = (
            self._best_f2_threshold(self.y_val, val_probs)
        )

        logging.info("Optimal validation F2 threshold: %.4f", best_threshold)

        val_preds = (val_probs >= best_threshold).astype(int)
        test_preds = (test_probs >= best_threshold).astype(int)

        val_metrics = {
            "AUROC": float(roc_auc_score(self.y_val, val_probs)),
            "AUPRC": float(average_precision_score(self.y_val, val_probs)),
            "Decision Threshold": float(best_threshold),
            "F2 Score": float(best_val_f2),
            "Confusion Matrix": confusion_matrix(self.y_val, val_preds).tolist(),
        }

        test_metrics = {
            "AUROC": float(roc_auc_score(self.y_test, test_probs)),
            "AUPRC": float(average_precision_score(self.y_test, test_probs)),
            "Decision Threshold": float(best_threshold),
            "F2 Score": float(fbeta_score(self.y_test, test_preds, beta=2)),
            "Confusion Matrix": confusion_matrix(self.y_test, test_preds).tolist(),
        }

        logging.info(
            "Validation Evaluation Metrics: %s", json.dumps(val_metrics, indent=2)
        )
        logging.info("Test Evaluation Metrics: %s", json.dumps(test_metrics, indent=2))

        importance = self.model.get_feature_importance()
        feat_importance = pd.DataFrame(
            {"feature": self.X_train.columns, "importance": importance}
        ).sort_values(by="importance", ascending=False)

        logging.info("Top 10 Feature Importances:")
        for _, row in feat_importance.head(10).iterrows():
            logging.info("%s: %.4f", row["feature"], row["importance"])

        self._save_evaluation_outputs(
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feat_importance=feat_importance,
        )

        self._plot_diagnostics(
            val_probs=val_probs,
            test_probs=test_probs,
            threshold_grid=threshold_grid,
            val_f2_grid=val_f2_grid,
            best_threshold=best_threshold,
        )

        self._investigate_feature_importance(feat_importance)

    def _load_files(
        self, train_val_path: Path, test_path: Path, metadata_path: Path
    ) -> None:
        if not train_val_path.exists():
            raise FileNotFoundError(
                f"Training/validation file not found: {train_val_path}"
            )
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        logging.info("Loading training/validation and test datasets...")
        self.train_val_df = pd.read_parquet(train_val_path)
        self.test_df = pd.read_parquet(test_path)
        self.metadata = json.loads(metadata_path.read_text())
        logging.info("Loaded files.")

    def _memory_optimisation(self) -> None:
        for cat_feat in self._resolved_categorical_cols():
            if cat_feat in self.train_val_df.columns:
                self.train_val_df[cat_feat] = self.train_val_df[cat_feat].astype(str)
            if cat_feat in self.test_df.columns:
                self.test_df[cat_feat] = self.test_df[cat_feat].astype(str)

        numeric_candidates = self.metadata.get("numerical_cols")
        if numeric_candidates is None:
            numeric_candidates = [
                c
                for c in self.train_val_df.columns
                if c
                not in self.metadata["categorical_cols"] + self.id_cols + [self.target]
            ]

        for col in numeric_candidates:
            if col in self.train_val_df.columns:
                self.train_val_df[col] = pd.to_numeric(
                    self.train_val_df[col], downcast="float"
                )
            if col in self.test_df.columns:
                self.test_df[col] = pd.to_numeric(self.test_df[col], downcast="float")

    def _resolved_categorical_cols(self) -> list[str]:
        metadata_cats = set(self.metadata.get("categorical_cols", []))
        available_forced = {
            c
            for c in self.FORCED_CATEGORICAL_COLS
            if c in self.train_val_df.columns or c in self.test_df.columns
        }
        return sorted(metadata_cats.union(available_forced))

    def _prepare_datasets(self) -> None:
        self._memory_optimisation()

        expected_train = self.metadata["train_row_count"]
        expected_val = self.metadata["val_row_count"]

        split_idx = int(len(self.train_val_df) * 8 / 9)  # 80% train, 10% val
        train_df = self.train_val_df.iloc[:split_idx]
        val_df = self.train_val_df.iloc[split_idx:]

        if expected_train and expected_val:
            assert len(train_df) == expected_train, (
                f"Expected {expected_train} rows in training set, got {len(train_df)}"
            )
            assert len(val_df) == expected_val, (
                f"Expected {expected_val} rows in validation set, got {len(val_df)}"
            )
            logging.info("Split recreation verified successfully.")

        self.X_train = train_df.drop(
            columns=[col for col in self.drop_cols if col in train_df.columns]
        )
        self.y_train = train_df[self.target]

        self.X_val = val_df.drop(
            columns=[col for col in self.drop_cols if col in val_df.columns]
        )
        self.y_val = val_df[self.target]

        self.X_test = self.test_df.drop(
            columns=[col for col in self.drop_cols if col in self.test_df.columns]
        )
        self.y_test = self.test_df[self.target]

        resolved_cats = self._resolved_categorical_cols()
        self.cat_feature_names = [c for c in resolved_cats if c in self.X_train.columns]
        self.cat_feature_indices = [
            self.X_train.columns.get_loc(c) for c in self.cat_feature_names
        ]

        missing_cat = set(resolved_cats) - set(self.cat_feature_names)
        if missing_cat:
            logging.warning(
                "Some metadata categorical columns are absent after "
                "preprocessing and were skipped: %s",
                sorted(missing_cat),
            )

        logging.info(
            "Final Sets - Train: %s | Val: %s | Test: %s",
            self.X_train.shape[0],
            self.X_val.shape[0],
            self.X_test.shape[0],
        )
        logging.info(
            "CatBoost cat_features parsed: %s columns", len(self.cat_feature_indices)
        )

    def _save_evaluation_outputs(
        self, val_metrics: dict, test_metrics: dict, feat_importance: pd.DataFrame
    ) -> None:
        time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        metrics_path = self.output_dir / f"evaluation_metrics_{time}.json"
        importance_path = self.output_dir / f"feature_importance_{time}.csv"
        hyperparams_path = self.output_dir / f"model_hyperparameters_{time}.json"

        with open(metrics_path, "w") as f:
            json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=4)
        logging.info("Saved evaluation metrics to %s", metrics_path)

        feat_importance.to_csv(importance_path, index=False)
        logging.info("Saved feature importance to %s", importance_path)

        hyperparam_payload = {
            "selected_hyperparameters": self.selected_hyperparameters,
            "selected_model_all_params": self.model.get_all_params(),
            "hyperparameter_tuning_results": self.tuning_results,
            "catboost_cat_feature_names": self.cat_feature_names,
            "catboost_cat_feature_indices": self.cat_feature_indices,
            "model_path": getattr(self, "model_path", None),
        }
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparam_payload, f, indent=4)
        logging.info("Saved hyperparameter report to %s", hyperparams_path)

    def _plot_diagnostics(
        self,
        val_probs: np.ndarray,
        test_probs: np.ndarray,
        threshold_grid: np.ndarray,
        val_f2_grid: np.ndarray,
        best_threshold: float,
    ) -> None:
        val_prec, val_rec, _ = precision_recall_curve(self.y_val, val_probs)
        test_prec, test_rec, _ = precision_recall_curve(self.y_test, test_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(val_rec, val_prec, label="Validation")
        plt.plot(test_rec, test_prec, label="Test")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        pr_path = self.output_dir / "precision_recall_curve.png"
        plt.savefig(pr_path)
        plt.close()
        logging.info("Saved PR curve to %s", pr_path)

        fpr_val, tpr_val, _ = roc_curve(self.y_val, val_probs)
        fpr_test, tpr_test, _ = roc_curve(self.y_test, test_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_val, tpr_val, label="Validation")
        plt.plot(fpr_test, tpr_test, label="Test")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        roc_path = self.output_dir / "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()
        logging.info("Saved ROC curve to %s", roc_path)

        if len(threshold_grid) > 0 and len(val_f2_grid) == len(threshold_grid):
            plt.figure(figsize=(8, 6))
            plt.plot(threshold_grid, val_f2_grid, color="darkgreen", linewidth=2)
            plt.axvline(
                x=best_threshold,
                color="red",
                linestyle="--",
                label=f"Best threshold={best_threshold:.4f}",
            )
            plt.xlabel("Decision Threshold")
            plt.ylabel("Validation F2")
            plt.title("Validation F2 vs Decision Threshold")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            f2_path = self.output_dir / "f2_vs_threshold.png"
            plt.savefig(f2_path)
            plt.close()
            logging.info("Saved F2-threshold curve to %s", f2_path)

    def _investigate_feature_importance(self, feat_importance: pd.DataFrame) -> None:
        df = feat_importance.sort_values("importance", ascending=False).reset_index(
            drop=True
        )
        df["cumulative_importance"] = df["importance"].cumsum()
        total_sum = df["importance"].sum()
        df["cumulative_percentage"] = (df["cumulative_importance"] / total_sum) * 100

        thresholds = [50, 75, 90, 95, 99]
        logging.info("Feature importance thresholds:")
        for t in thresholds:
            num_features = df[df["cumulative_percentage"] >= t].index[0] + 1
            logging.info(
                "Features required for %s%% of model signal: %s", t, num_features
            )

        self._visualise_importance(df)
        self._log_features_to_drop(df, threshold=90)

    def _visualise_importance(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df.head(20),
            x="importance",
            y="feature",
            palette="viridis",
        )
        plt.title("Top 20 Feature Importances (Relative %)")
        plt.xlabel("Importance Score (%)")
        plt.ylabel("Feature")
        plt.tight_layout()

        bar_save_path = self.output_dir / "feature_importance_top20.png"
        plt.savefig(bar_save_path)
        plt.close()
        logging.info("Saved Top 20 bar chart to: %s", bar_save_path)

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(df) + 1),
            df["cumulative_percentage"],
            linewidth=2,
            color="navy",
        )

        plt.axhline(y=99, color="purple", linestyle="--", alpha=0.6, label="99% Signal")
        plt.axhline(y=95, color="red", linestyle="--", alpha=0.6, label="95% Signal")
        plt.axhline(y=90, color="orange", linestyle="--", alpha=0.6, label="90% Signal")

        plt.title("Cumulative Feature Importance Curve")
        plt.xlabel("Number of Features")
        plt.ylabel("Cumulative Importance (%)")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        curve_save_path = self.output_dir / "feature_importance_cumulative.png"
        plt.savefig(curve_save_path)
        plt.close()
        logging.info("Saved cumulative curve to: %s", curve_save_path)

    def _log_features_to_drop(self, df: pd.DataFrame, threshold: int) -> None:
        features_to_drop = df[df["cumulative_percentage"] > threshold][
            "feature"
        ].tolist()
        drop_path = self.output_dir / "features_to_drop.txt"

        with open(drop_path, "w") as f:
            for feat in features_to_drop:
                f.write(f"- {feat}\n")
        logging.info("Saved low-importance features list to %s", drop_path)


def main() -> None:
    args = parse_args()
    trainer = CatBoostTrainer(
        train_val_path=Path(args.train_val_path),
        test_path=Path(args.test_path),
        metadata_path=Path(args.metadata_path),
        tune=args.tune,
    )
    trainer.train(Path(args.models_dir))
    trainer.evaluate()


if __name__ == "__main__":
    main()
