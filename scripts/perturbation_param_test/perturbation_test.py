#!/usr/bin/env python3
"""Parameter calibration runner for perturbation locality/OOD checks.

Design goals:
- Sample fixed counts from TP/FP groups.
- Generate perturbations only (no SHAP/LIME explainers are run).
- Sweep parameter grids for lambda, drift ranges, and K.
- Emit value-only diagnostics to inspect whether perturbations are too local
  or too far/out-of-distribution.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import dataset  # noqa: F401
import model  # noqa: F401
import perturbation  # noqa: F401
from dataset.registry import get_dataset
from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult
from load_config import PerturbationConfig, load_config
from model.registry import get_model
from perturbation.registry import get_perturbation

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate perturbation params (lambda, drift range, K) on values only"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="perturbation_calibration_ieee_cis.yaml",
        help="YAML under config/ describing calibration sweep",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def load_calibration_yaml(config_name: str) -> dict[str, Any]:
    path = REPO_ROOT / "config" / config_name
    if not path.exists():
        raise FileNotFoundError(f"Calibration config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Calibration YAML must be a mapping at top level.")

    required = [
        "base_xai_config",
        "output_dir",
        "random_seed",
        "sample_groups",
        "samples_per_group",
        "n_perturbations_per_sample",
        "perturbations",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Calibration YAML missing required fields: {missing}")

    return cfg


def get_group_mask(y: np.ndarray, preds: np.ndarray, group: str) -> np.ndarray:
    group_upper = str(group).upper()
    mapping = {
        "TP": (1, 1),
        "TN": (0, 0),
        "FP": (1, 0),
        "FN": (0, 1),
    }
    if group_upper not in mapping:
        raise ValueError(f"Unsupported group '{group}'. Use TP/TN/FP/FN.")

    pred_val, actual_val = mapping[group_upper]
    return (preds == pred_val) & (y == actual_val)


def sample_instances(
    X: pd.DataFrame,
    y: pd.Series,
    preds: np.ndarray,
    groups: list[str],
    n_per_group: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y_array = np.asarray(y)
    out: dict[str, pd.DataFrame] = {}

    for group in groups:
        mask = get_group_mask(y_array, preds, group)
        group_df = X.loc[mask]
        if group_df.empty:
            raise ValueError(f"No rows found for sample group '{group}'.")

        n_take = min(n_per_group, len(group_df))
        chosen_idx = rng.choice(group_df.index.to_numpy(), size=n_take, replace=False)
        out[str(group).upper()] = group_df.loc[chosen_idx].copy()

    return out


def normalised_ranges(train_df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    ranges: dict[str, float] = {}
    for col in numeric_cols:
        col_min = pd.to_numeric(train_df[col], errors="coerce").min()
        col_max = pd.to_numeric(train_df[col], errors="coerce").max()
        width = float(col_max - col_min) if pd.notna(col_max) and pd.notna(col_min) else 0.0
        ranges[col] = width if width > 0 else 1.0
    return ranges


def tight_normalised_ranges(
    feature_stats: dict[str, dict[str, Any]], numeric_cols: list[str]
) -> dict[str, float]:
    ranges: dict[str, float] = {}
    for col in numeric_cols:
        low = float(feature_stats[col]["q05"])
        high = float(feature_stats[col]["q95"])
        width = high - low
        ranges[col] = width if width > 0 else 1.0
    return ranges


def values_equal(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def gower_row_distance(
    base_row: pd.Series,
    pert_row: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    ranges: dict[str, float],
) -> float:
    parts: list[float] = []

    for col in numeric_cols:
        a = pd.to_numeric(base_row[col], errors="coerce")
        b = pd.to_numeric(pert_row[col], errors="coerce")
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            parts.append(1.0)
            continue
        parts.append(min(abs(float(b) - float(a)) / ranges[col], 1.0))

    for col in categorical_cols:
        a = base_row[col]
        b = pert_row[col]
        if pd.isna(a) and pd.isna(b):
            continue
        parts.append(0.0 if values_equal(a, b) else 1.0)

    if not parts:
        return 0.0
    return float(np.mean(parts))


def make_constant_importance_explanations(
    sample_df: pd.DataFrame,
    feature_names: list[str],
    feature_importances: np.ndarray,
) -> ExplanationResult:
    instances = [
        Explanation(
            instance_id=idx,
            values=np.asarray(feature_importances, dtype=float),
            base_value=0.0,
            prediction=None,
        )
        for idx in sample_df.index
    ]

    return ExplanationResult(
        explainer_name="CatBoostFeatureImportanceProxy",
        instances=instances,
        base_value=0.0,
        feature_names=feature_names,
        instance_ids=list(sample_df.index),
        metadata={"source": "catboost_feature_importance"},
    )


def build_trials(cal_cfg: dict[str, Any], n_perturbations: int) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    perturb_cfg = cal_cfg["perturbations"]

    noise_cfg = perturb_cfg.get("local_gaussian_noise", {})
    if noise_cfg.get("enabled", False):
        for lam in noise_cfg.get("lambdas", []):
            params = {"lambda": float(lam)}
            trials.append(
                {
                    "name": "LocalGaussianNoise",
                    "params": params,
                    "n_perturbations": n_perturbations,
                }
            )

    drift_cfg = perturb_cfg.get("directional_drift", {})
    if drift_cfg.get("enabled", False):
        drift_factor = float(drift_cfg.get("drift_factor", 0.1))
        target_features = list(drift_cfg.get("target_features", []))
        for drift_range in drift_cfg.get("drift_factor_ranges", []):
            if not isinstance(drift_range, (list, tuple)) or len(drift_range) != 2:
                raise ValueError(
                    "Each directional_drift.drift_factor_ranges entry must be [low, high]."
                )
            params = {
                "drift_factor": drift_factor,
                "drift_factor_range": [float(drift_range[0]), float(drift_range[1])],
                "target_features": target_features,
            }
            trials.append(
                {
                    "name": "DirectionalDrift",
                    "params": params,
                    "n_perturbations": n_perturbations,
                }
            )

    topk_cfg = perturb_cfg.get("top_k_features", {})
    if topk_cfg.get("enabled", False):
        lambdas = topk_cfg.get("lambdas", [0.05])
        ks = topk_cfg.get("ks", [])

        for lam in lambdas:
            for k in ks:
                params = {"lambda": float(lam), "k": int(k)}
                trials.append(
                    {
                        "name": "TopKFeatures",
                        "params": params,
                        "n_perturbations": n_perturbations,
                    }
                )

    if not trials:
        raise ValueError("No enabled perturbation trials found in calibration YAML.")

    return trials


def evaluate_trial(
    trial: dict[str, Any],
    sample_group: str,
    sample_df: pd.DataFrame,
    perturbation_obj,
    numeric_cols: list[str],
    categorical_cols: list[str],
    ranges: dict[str, float],
    tight_ranges: dict[str, float],
    feature_stats: dict[str, dict[str, Any]],
    valid_categories: dict[str, list[Any]],
    thresholds: dict[str, float],
    topk_explanations: ExplanationResult | None = None,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if trial["name"] == "TopKFeatures":
        if topk_explanations is None:
            raise ValueError("TopK trial requires prebuilt explanation_result.")
        kwargs["explanation_result"] = topk_explanations

    perturbed_df = perturbation_obj.perturb(sample_df, **kwargs)
    params_json = json.dumps(trial["params"], sort_keys=True)

    records: list[dict[str, Any]] = []
    for instance_id, pert_rows in perturbed_df.groupby(level=0, sort=False):
        base_row = sample_df.loc[instance_id]
        base_row_df = sample_df.loc[[instance_id]]
        already_ood = perturbation_obj._get_already_ood_features(base_row_df)
        already_ood_numeric = {c for c in numeric_cols if c in already_ood}
        already_ood_categorical = {c for c in categorical_cols if c in already_ood}

        for rep_i, (_, pert_row) in enumerate(pert_rows.iterrows(), start=1):
            changed_count = 0
            for col in sample_df.columns:
                if not values_equal(base_row[col], pert_row[col]):
                    changed_count += 1

            numeric_norm_deltas: list[float] = []
            numeric_tight_norm_deltas: list[float] = []
            ood_numeric_count = 0
            for col in numeric_cols:
                a = pd.to_numeric(base_row[col], errors="coerce")
                b = pd.to_numeric(pert_row[col], errors="coerce")
                if not (pd.isna(a) or pd.isna(b)):
                    numeric_norm_deltas.append(abs(float(b) - float(a)) / ranges[col])
                    numeric_tight_norm_deltas.append(
                        abs(float(b) - float(a)) / tight_ranges[col]
                    )

                if col in already_ood_numeric:
                    continue
                if pd.isna(b):
                    continue
                low = feature_stats[col]["q05"]
                high = feature_stats[col]["q95"]
                if float(b) < float(low) or float(b) > float(high):
                    ood_numeric_count += 1

            ood_categorical_count = 0
            changed_categorical_count = 0
            for col in categorical_cols:
                if not values_equal(base_row[col], pert_row[col]):
                    changed_categorical_count += 1
                if col in already_ood_categorical:
                    continue
                val = pert_row[col]
                if pd.isna(val):
                    continue
                if val not in valid_categories[col]:
                    ood_categorical_count += 1

            gower = gower_row_distance(
                base_row=base_row,
                pert_row=pert_row,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                ranges=ranges,
            )

            numeric_delta_mean = (
                float(np.mean(numeric_norm_deltas)) if numeric_norm_deltas else 0.0
            )
            numeric_delta_max = (
                float(np.max(numeric_norm_deltas)) if numeric_norm_deltas else 0.0
            )
            continuous_tight_distance = (
                float(np.mean(numeric_tight_norm_deltas))
                if numeric_tight_norm_deltas
                else 0.0
            )
            continuous_tight_distance_max = (
                float(np.max(numeric_tight_norm_deltas))
                if numeric_tight_norm_deltas
                else 0.0
            )
            categorical_change_rate = (
                changed_categorical_count / len(categorical_cols)
                if categorical_cols
                else 0.0
            )

            rec = {
                "sample_group": sample_group,
                "perturbation": trial["name"],
                "params": params_json,
                "instance_id": instance_id,
                "replicate": rep_i,
                "gower_distance": gower,
                "numeric_delta_mean": numeric_delta_mean,
                "numeric_delta_max": numeric_delta_max,
                "continuous_tight_distance": continuous_tight_distance,
                "continuous_tight_distance_max": continuous_tight_distance_max,
                "changed_feature_count": changed_count,
                "categorical_change_rate": categorical_change_rate,
                "ood_numeric_count": ood_numeric_count,
                "ood_categorical_count": ood_categorical_count,
                "ood_any": int((ood_numeric_count + ood_categorical_count) > 0),
                "too_local": int(gower <= thresholds["too_local_max_gower"]),
                "too_far": int(gower >= thresholds["too_far_min_gower"]),
            }
            records.append(rec)

    return records


def summarise_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["sample_group", "perturbation", "params"]
    summaries: list[dict[str, Any]] = []

    if pair_df.empty:
        return pd.DataFrame(columns=[
            "sample_group",
            "perturbation",
            "params",
            "n_pairs",
            "n_instances",
            "gower_mean",
            "gower_median",
            "gower_p05",
            "gower_p95",
            "numeric_delta_mean",
            "numeric_delta_max_mean",
            "continuous_tight_distance_mean",
            "continuous_tight_distance_max_mean",
            "changed_feature_count_mean",
            "categorical_change_rate_mean",
            "ood_any_rate",
            "too_local_rate",
            "too_far_rate",
        ])

    for keys, g in pair_df.groupby(group_cols, dropna=False):
        sample_group, perturbation_name, params = keys
        summaries.append(
            {
                "sample_group": sample_group,
                "perturbation": perturbation_name,
                "params": params,
                "n_pairs": int(len(g)),
                "n_instances": int(g["instance_id"].nunique()),
                "gower_mean": float(g["gower_distance"].mean()),
                "gower_median": float(g["gower_distance"].median()),
                "gower_p05": float(g["gower_distance"].quantile(0.05)),
                "gower_p95": float(g["gower_distance"].quantile(0.95)),
                "numeric_delta_mean": float(g["numeric_delta_mean"].mean()),
                "numeric_delta_max_mean": float(g["numeric_delta_max"].mean()),
                "continuous_tight_distance_mean": float(
                    g["continuous_tight_distance"].mean()
                ),
                "continuous_tight_distance_max_mean": float(
                    g["continuous_tight_distance_max"].mean()
                ),
                "changed_feature_count_mean": float(g["changed_feature_count"].mean()),
                "categorical_change_rate_mean": float(g["categorical_change_rate"].mean()),
                "ood_any_rate": float(g["ood_any"].mean()),
                "too_local_rate": float(g["too_local"].mean()),
                "too_far_rate": float(g["too_far"].mean()),
            }
        )

    return pd.DataFrame(summaries).sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s| %(levelname)s| %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting perturbation calibration. config=%s", args.config)
    cal_cfg = load_calibration_yaml(args.config)
    logger.info("Loaded calibration YAML from config/%s", args.config)

    base_xai_cfg_name = str(cal_cfg["base_xai_config"])
    base_cfg = load_config(base_xai_cfg_name)
    logger.info("Loaded base XAI config: config/%s", base_xai_cfg_name)

    datasets = get_dataset(base_cfg.dataset)
    train_data = datasets.train
    test_data = datasets.test

    fraud_model = get_model(base_cfg.model)
    fraud_model.validate_features(test_data.model_feature_names)

    X_train = train_data.X_model
    X_test = test_data.X_model
    y_test = test_data.y
    preds = np.asarray(fraud_model.predict(X_test)).flatten().astype(int)

    random_seed = int(cal_cfg["random_seed"])
    sample_groups = [str(g).upper() for g in list(cal_cfg["sample_groups"])]
    samples_per_group = int(cal_cfg["samples_per_group"])
    n_perturbations = int(cal_cfg["n_perturbations_per_sample"])

    sampled = sample_instances(
        X=X_test,
        y=y_test,
        preds=preds,
        groups=sample_groups,
        n_per_group=samples_per_group,
        seed=random_seed,
    )
    sampled_counts = {group: len(df) for group, df in sampled.items()}
    logger.info("Sampled instances per group: %s", sampled_counts)

    mutable_numeric_cols: list[str] = []
    for col in X_train.columns:
        if col in base_cfg.dataset.immutable_features:
            continue
        if col in base_cfg.dataset.categorical_features:
            continue
        if pd.api.types.is_numeric_dtype(X_train[col]):
            mutable_numeric_cols.append(col)

    mutable_categorical_cols = [
        c
        for c in base_cfg.dataset.categorical_features
        if c in X_train.columns and c not in base_cfg.dataset.immutable_features
    ]

    num_ranges = normalised_ranges(X_train, mutable_numeric_cols)

    analysis_cfg = cal_cfg.get("analysis", {})
    thresholds = analysis_cfg.get("locality_thresholds", {})
    too_local_max_gower = float(thresholds.get("too_local_max_gower", 0.01))
    too_far_min_gower = float(thresholds.get("too_far_min_gower", 0.20))

    trials = build_trials(cal_cfg, n_perturbations=n_perturbations)
    logger.info("Built %d perturbation trials", len(trials))

    feature_names = list(X_test.columns)
    catboost_importances = np.asarray(fraud_model.model.get_feature_importance())
    if len(catboost_importances) != len(feature_names):
        raise ValueError(
            "CatBoost feature importances length does not match model feature names."
        )

    output_dir = REPO_ROOT / str(cal_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []

    for sample_group, sample_df in sampled.items():
        logger.info(
            "Processing sample_group=%s with %d sampled instances",
            sample_group,
            len(sample_df),
        )
        for instance_id in sample_df.index:
            selected_rows.append({"sample_group": sample_group, "instance_id": instance_id})

        topk_explanations = make_constant_importance_explanations(
            sample_df=sample_df,
            feature_names=feature_names,
            feature_importances=catboost_importances,
        )

        for trial_idx, trial in enumerate(trials):
            logger.info(
                "Running trial %d/%d for group=%s: %s params=%s",
                trial_idx + 1,
                len(trials),
                sample_group,
                trial["name"],
                trial["params"],
            )
            perturb_cfg = PerturbationConfig(
                name=trial["name"],
                n_perturbations=int(trial["n_perturbations"]),
                random_seed=random_seed + trial_idx,
                params=trial["params"],
            )
            perturbation_obj = get_perturbation(
                perturb_cfg,
                X_train,
                immutable_features=base_cfg.dataset.immutable_features,
                categorical_features=base_cfg.dataset.categorical_features,
                perturbable_categorical_features=base_cfg.dataset.perturbable_categorical_features,
                perturbable_numerical_features=base_cfg.dataset.perturbable_numerical_features,
                integer_features=base_cfg.dataset.integer_features,
                non_negative_features=base_cfg.dataset.non_negative_features,
                non_negative_prefixes=base_cfg.dataset.non_negative_prefixes,
            )
            tight_ranges = tight_normalised_ranges(
                perturbation_obj.feature_stats, mutable_numeric_cols
            )

            trial_records = evaluate_trial(
                trial=trial,
                sample_group=sample_group,
                sample_df=sample_df,
                perturbation_obj=perturbation_obj,
                numeric_cols=mutable_numeric_cols,
                categorical_cols=mutable_categorical_cols,
                ranges=num_ranges,
                tight_ranges=tight_ranges,
                feature_stats=perturbation_obj.feature_stats,
                valid_categories=perturbation_obj.valid_categories,
                thresholds={
                    "too_local_max_gower": too_local_max_gower,
                    "too_far_min_gower": too_far_min_gower,
                },
                topk_explanations=topk_explanations,
            )
            pair_records.extend(trial_records)
            logger.info(
                "Completed trial %d/%d for group=%s: generated %d records",
                trial_idx + 1,
                len(trials),
                sample_group,
                len(trial_records),
            )

    pair_df = pd.DataFrame(pair_records)
    if "changed_feature_count" not in pair_df.columns:
        raise ValueError("pairwise record generation missing 'changed_feature_count'.")

    pair_df_nonzero = pair_df[pair_df["changed_feature_count"] > 0].copy()
    summary_df = summarise_pairs(pair_df_nonzero)

    pair_csv = output_dir / "pairwise_value_metrics.csv"
    pair_nonzero_csv = output_dir / "pairwise_value_metrics_nonzero.csv"
    summary_csv = output_dir / "calibration_summary.csv"
    selected_csv = output_dir / "selected_instances.csv"
    manifest_json = output_dir / "calibration_manifest.json"

    pair_df.to_csv(pair_csv, index=False)
    pair_df_nonzero.to_csv(pair_nonzero_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pd.DataFrame(selected_rows).to_csv(selected_csv, index=False)

    manifest = {
        "calibration_config": cal_cfg,
        "resolved_base_config": base_xai_cfg_name,
        "dataset": asdict(base_cfg.dataset),
        "model": asdict(base_cfg.model),
        "sample_groups": sample_groups,
        "samples_per_group": samples_per_group,
        "n_perturbations_per_sample": n_perturbations,
        "n_trials": len(trials),
        "output_files": {
            "pairwise_value_metrics": str(pair_csv),
            "pairwise_value_metrics_nonzero": str(pair_nonzero_csv),
            "calibration_summary": str(summary_csv),
            "selected_instances": str(selected_csv),
        },
        "summary_excludes_noop_rows": True,
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote: %s", pair_csv)
    logger.info("Wrote: %s", pair_nonzero_csv)
    logger.info("Wrote: %s", summary_csv)
    logger.info("Wrote: %s", selected_csv)
    logger.info("Wrote: %s", manifest_json)
    logger.info("Perturbation calibration design complete.")


if __name__ == "__main__":
    main()
