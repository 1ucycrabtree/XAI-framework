from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Save plots in headless environments.
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import dataset  # noqa: F401
import model  # noqa: F401
import perturbation  # noqa: F401
from dataset.registry import get_dataset
from load_config import PerturbationConfig, load_config
from model.registry import get_model
from perturbation.registry import get_perturbation

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perturbation locality validation")
    parser.add_argument(
        "--config",
        type=str,
        default="locality_validation_noise.yaml",
        help="YAML under config/ describing locality validation",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def load_yaml(config_name: str) -> dict[str, Any]:
    path = REPO_ROOT / "config" / config_name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping at top level.")
    return data


def resolve_perturbation_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("perturbations")
    if raw is None:
        raise ValueError("Config missing required key: 'perturbations'.")

    # Allow either a single mapping or a list of mappings.
    if isinstance(raw, list):
        if not raw:
            raise ValueError("'perturbations' list is empty.")
        spec = raw[0]
    elif isinstance(raw, dict):
        spec = raw
    else:
        raise ValueError(
            "'perturbations' must be either a mapping or a non-empty list of mappings."
        )

    if not isinstance(spec, dict):
        raise ValueError(
            "Resolved perturbation spec is not a mapping. "
            "Expected keys: 'name' and 'params'."
        )
    if "name" not in spec:
        raise ValueError("Perturbation spec missing required key: 'name'.")
    if "params" not in spec:
        raise ValueError("Perturbation spec missing required key: 'params'.")
    if not isinstance(spec["params"], dict):
        raise ValueError("Perturbation spec 'params' must be a mapping.")

    return {"name": str(spec["name"]), "params": dict(spec["params"])}


def get_group_mask(y: np.ndarray, preds: np.ndarray, group: str) -> np.ndarray:
    group_upper = str(group).upper()
    mapping = {"TP": (1, 1), "TN": (0, 0), "FP": (1, 0), "FN": (0, 1)}
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
    y_arr = np.asarray(y)
    out: dict[str, pd.DataFrame] = {}
    for group in groups:
        mask = get_group_mask(y_arr, preds, group)
        group_df = X.loc[mask]
        if group_df.empty:
            raise ValueError(f"No rows found for sample group '{group}'.")
        n_take = min(n_per_group, len(group_df))
        chosen = rng.choice(group_df.index.to_numpy(), size=n_take, replace=False)
        out[group] = group_df.loc[chosen].copy()
    return out


def values_equal(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def is_noop_perturbation(base_row: pd.Series, pert_row: pd.Series) -> bool:
    for col in base_row.index:
        if not values_equal(base_row[col], pert_row[col]):
            return False
    return True


def normalised_ranges(train_df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    ranges: dict[str, float] = {}
    for col in numeric_cols:
        col_min = pd.to_numeric(train_df[col], errors="coerce").min()
        col_max = pd.to_numeric(train_df[col], errors="coerce").max()
        width = float(col_max - col_min) if pd.notna(col_max) and pd.notna(col_min) else 0.0
        ranges[col] = width if width > 0 else 1.0
    return ranges


def gower_row_distance(
    base_row: pd.Series,
    other_row: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    ranges: dict[str, float],
) -> float:
    parts: list[float] = []
    for col in numeric_cols:
        a = pd.to_numeric(base_row[col], errors="coerce")
        b = pd.to_numeric(other_row[col], errors="coerce")
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            parts.append(1.0)
            continue
        parts.append(min(abs(float(b) - float(a)) / ranges[col], 1.0))
    for col in categorical_cols:
        a = base_row[col]
        b = other_row[col]
        if pd.isna(a) and pd.isna(b):
            continue
        parts.append(0.0 if values_equal(a, b) else 1.0)
    if not parts:
        return 0.0
    return float(np.mean(parts))


def constraint_violation_rate_for_row(
    base_row: pd.Series,
    pert_row: pd.Series,
    perturbation_obj,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> int:
    base_df = pd.DataFrame([base_row], index=[base_row.name])
    already_ood = perturbation_obj._get_already_ood_features(base_df)
    already_ood_num = {c for c in numeric_cols if c in already_ood}
    already_ood_cat = {c for c in categorical_cols if c in already_ood}

    for col in numeric_cols:
        if col in already_ood_num:
            continue
        b = pd.to_numeric(pert_row[col], errors="coerce")
        if pd.isna(b):
            continue
        low = perturbation_obj.feature_stats[col]["q05"]
        high = perturbation_obj.feature_stats[col]["q95"]
        if float(b) < float(low) or float(b) > float(high):
            return 1

    for col in categorical_cols:
        if col in already_ood_cat:
            continue
        val = pert_row[col]
        if pd.isna(val):
            continue
        if val not in perturbation_obj.valid_categories[col]:
            return 1

    return 0


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s| %(levelname)s| %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_yaml(args.config)
    base_cfg = load_config(str(cfg["base_xai_config"]))
    output_dir = REPO_ROOT / str(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = get_dataset(base_cfg.dataset)
    train_data = datasets.train
    test_data = datasets.test
    X_train = train_data.X_model
    X_test = test_data.X_model
    y_test = test_data.y

    fraud_model = get_model(base_cfg.model)
    fraud_model.validate_features(test_data.model_feature_names)
    preds_test = np.asarray(fraud_model.predict(X_test)).flatten().astype(int)

    sample_groups = [str(g).upper() for g in cfg["sample_groups"]]
    n_samples = int(cfg["samples_per_group"])
    n_perturbations = int(cfg["n_perturbations_per_sample"])
    seed = int(cfg["random_seed"])

    sampled = sample_instances(X_test, y_test, preds_test, sample_groups, n_samples, seed)
    logger.info("Sampled instances per group: %s", {k: len(v) for k, v in sampled.items()})

    perturb_spec = resolve_perturbation_spec(cfg)
    logger.info(
        "Using perturbation for locality validation: %s params=%s",
        perturb_spec["name"],
        perturb_spec["params"],
    )
    perturb_cfg = PerturbationConfig(
        name=str(perturb_spec["name"]),
        n_perturbations=n_perturbations,
        random_seed=seed,
        params=dict(perturb_spec["params"]),
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

    mutable_numeric_cols = [
        c
        for c in X_train.columns
        if c not in base_cfg.dataset.immutable_features
        and c not in base_cfg.dataset.categorical_features
        and pd.api.types.is_numeric_dtype(X_train[c])
    ]
    mutable_categorical_cols = [
        c
        for c in base_cfg.dataset.categorical_features
        if c in X_train.columns and c not in base_cfg.dataset.immutable_features
    ]
    ranges = normalised_ranges(X_train, mutable_numeric_cols)

    rng = np.random.default_rng(seed + 999)
    all_test_indices = X_test.index.to_numpy()
    records: list[dict[str, Any]] = []

    for group_name, group_df in sampled.items():
        perturbed = perturbation_obj.perturb(group_df)

        for instance_id, pert_rows in perturbed.groupby(level=0, sort=False):
            base_row = group_df.loc[instance_id] # type: ignore
            base_pred = int(
                np.asarray(fraud_model.predict(group_df.loc[[instance_id]])).flatten()[0]
            )

            valid_random_pool = all_test_indices[all_test_indices != instance_id]
            if len(valid_random_pool) == 0:
                continue

            for rep_i, (_, pert_row) in enumerate(pert_rows.iterrows(), start=1):
                if is_noop_perturbation(base_row, pert_row):
                    continue

                rand_id = rng.choice(valid_random_pool)
                rand_row = X_test.loc[rand_id]

                dist_pert = gower_row_distance(
                    base_row, pert_row, mutable_numeric_cols, mutable_categorical_cols, ranges
                )
                dist_rand = gower_row_distance(
                    base_row, rand_row, mutable_numeric_cols, mutable_categorical_cols, ranges
                )

                pert_pred = int(
                    np.asarray(fraud_model.predict(pd.DataFrame([pert_row]))).flatten()[0]
                )
                pred_flip = int(pert_pred != base_pred)

                violation = constraint_violation_rate_for_row(
                    base_row=base_row,
                    pert_row=pert_row,
                    perturbation_obj=perturbation_obj,
                    numeric_cols=mutable_numeric_cols,
                    categorical_cols=mutable_categorical_cols,
                )

                records.append(
                    {
                        "sample_group": group_name,
                        "instance_id": instance_id,
                        "replicate": rep_i,
                        "distance_perturbed": dist_pert,
                        "distance_random": dist_rand,
                        "locality_win": int(dist_pert < dist_rand),
                        "prediction_flip": pred_flip,
                        "constraint_violation": violation,
                    }
                )

    pairwise_df = pd.DataFrame(records)
    if pairwise_df.empty:
        raise ValueError("No locality-validation records generated.")

    summary_rows: list[dict[str, Any]] = []
    for label, sub in [("ALL", pairwise_df)] + [
        (str(k), v) for k, v in pairwise_df.groupby("sample_group")
    ]:
        summary_rows.append(
            {
                "sample_group": label,
                "n_pairs": int(len(sub)),
                "perturbed_distance_mean": float(sub["distance_perturbed"].mean()),
                "perturbed_distance_median": float(sub["distance_perturbed"].median()),
                "random_distance_mean": float(sub["distance_random"].mean()),
                "random_distance_median": float(sub["distance_random"].median()),
                "locality_success_rate": float(sub["locality_win"].mean()),
                "prediction_flip_rate": float(sub["prediction_flip"].mean()),
                "constraint_violation_rate": float(sub["constraint_violation"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    pairwise_csv = output_dir / "pairwise_locality_validation.csv"
    summary_csv = output_dir / "locality_validation_summary.csv"
    figure_png = output_dir / "perturbation_locality.png"
    meta_json = output_dir / "locality_validation_manifest.json"

    pairwise_df.to_csv(pairwise_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plot_df = pairwise_df.melt(
        id_vars=["sample_group", "instance_id", "replicate"],
        value_vars=["distance_perturbed", "distance_random"],
        var_name="distance_type",
        value_name="distance_value",
    )
    plot_df["distance_type"] = plot_df["distance_type"].map(
        {
            "distance_perturbed": "Perturbed vs Original",
            "distance_random": "Random vs Original",
        }
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=plot_df,
        x="distance_type",
        y="distance_value",
        hue="sample_group",
        showfliers=False,
    )
    ax.set_yscale("log")
    ax.set_title("Perturbation Locality Validation: Perturbed vs Random Distances")
    ax.set_xlabel("")
    ax.set_ylabel("Gower Distance (log scale)")
    plt.tight_layout()
    plt.savefig(figure_png, dpi=200)
    plt.close()

    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg,
                "pairwise_output": str(pairwise_csv),
                "summary_output": str(summary_csv),
                "figure_output": str(figure_png),
            },
            f,
            indent=2,
        )

    logger.info("Wrote: %s", pairwise_csv)
    logger.info("Wrote: %s", summary_csv)
    logger.info("Wrote: %s", figure_png)
    logger.info("Wrote: %s", meta_json)


if __name__ == "__main__":
    main()
