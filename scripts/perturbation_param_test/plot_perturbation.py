from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRWISE_METRICS_PATH = (
    REPO_ROOT
    / "results"
    / "perturbation_calibration_design"
    / "pairwise_value_metrics.csv"
)

PERTURBATION_CODES = {
    "LocalGaussianNoise": "GN",
    "DirectionalDrift": "DD",
    "TopKFeatures": "TK",
}
MAX_VALID_TOPK = 6  # Only 6 features are perturbable


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _format_params(perturbation: str, params_json: str) -> str:
    params = json.loads(params_json)
    if perturbation == "LocalGaussianNoise":
        return f"\u03bb={params.get('lambda')}"
    if perturbation == "DirectionalDrift":
        drift_range = params.get("drift_factor_range")
        if isinstance(drift_range, list) and len(drift_range) == 2:
            return f"\u03b1 = ({drift_range[0]}, {drift_range[1]}]"
        return f"\u03b1={params.get('drift_factor')}"
    if perturbation == "TopKFeatures":
        k = params.get("k")
        lam = params.get("lambda")
        return f"K={k}, \u03bb={lam}"
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _trial_order_by_mean_distance(df: pd.DataFrame, col: str) -> list[str]:
    means = df.groupby(col)["gower_distance"].mean().sort_values()
    return means.index.tolist()


def _trial_order_grouped_by_perturbation(df: pd.DataFrame) -> list[str]:
    # Group trials by perturbation family first (GN -> DD -> TK), then
    # sort settings by mean Gower distance within each family.
    code_order = [
        PERTURBATION_CODES.get("LocalGaussianNoise", "GN"),
        PERTURBATION_CODES.get("DirectionalDrift", "DD"),
        PERTURBATION_CODES.get("TopKFeatures", "TK"),
    ]
    present_codes = list(df["perturbation_code"].dropna().unique())
    ordered_codes = [c for c in code_order if c in present_codes] + [
        c for c in sorted(present_codes) if c not in code_order
    ]

    ordered_trials: list[str] = []
    for code in ordered_codes:
        subset = df[df["perturbation_code"] == code].copy()
        if subset.empty:
            continue
        means = subset.groupby("trial")["gower_distance"].mean().sort_values()
        ordered_trials.extend(means.index.tolist())
    return ordered_trials


def _parse_topk_params(df: pd.DataFrame) -> pd.DataFrame:
    topk = df.copy()
    parsed = topk["params"].apply(json.loads)
    topk["k"] = parsed.apply(lambda p: p.get("k"))
    topk["lambda"] = parsed.apply(lambda p: p.get("lambda"))
    topk["k"] = pd.to_numeric(topk["k"], errors="coerce")
    topk["lambda"] = pd.to_numeric(topk["lambda"], errors="coerce")
    # Stable unique key for x-axis category.
    topk["topk_setting"] = topk.apply(
        lambda r: f"K={int(r['k'])}|L={r['lambda']}",
        axis=1,
    )
    return topk


def _filter_topk_by_k_cap(
    df: pd.DataFrame, k_cap: int = MAX_VALID_TOPK
) -> pd.DataFrame:
    topk_mask = df["perturbation"] == "TopKFeatures"
    if not topk_mask.any():
        return df

    topk = _parse_topk_params(df[topk_mask].copy())
    keep_topk = topk[topk["k"] <= k_cap].copy()
    non_topk = df[~topk_mask].copy()
    if keep_topk.empty:
        return non_topk
    return pd.concat([non_topk, keep_topk], ignore_index=True)


def _plot_topk_grouped(
    subset: pd.DataFrame,
    output_dir: Path,
    perturbation_name: str,
) -> None:
    topk = _parse_topk_params(subset)
    topk = topk.dropna(subset=["k", "lambda"]).copy()
    if topk.empty:
        return

    # Group by K, then lambda.
    order_df = (
        topk[["topk_setting", "k", "lambda"]]
        .drop_duplicates()
        .sort_values(["k", "lambda"])
    )
    order = order_df["topk_setting"].tolist()

    lam_count = int(order_df["lambda"].nunique())
    if lam_count > 1:
        tick_labels = [
            f"K={int(r.k)}\n\u03bb={r['lambda']}" for _, r in order_df.iterrows()
        ]
    else:
        tick_labels = [f"K={int(r.k)}" for _, r in order_df.iterrows()]

    plt.figure(figsize=(14, 6.5))
    ax = sns.boxplot(
        data=topk,
        x="topk_setting",
        y="gower_distance_plot",
        hue="sample_group",
        order=order,
        showfliers=False,
    )
    ax.set_yscale("log")
    ax.set_title(
        f"Parameter Calibration Experiment: {PERTURBATION_CODES.get(perturbation_name, perturbation_name)}"  # noqa: E501
    )
    ax.set_xlabel("Top-K setting")
    ax.set_ylabel("Gower Distance (log scale)")

    # Apply compact K labels.
    ax.set_xticklabels(tick_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(
        output_dir
        / f"gower_distance_{PERTURBATION_CODES.get(perturbation_name, perturbation_name)}.png",  # noqa: E501
        dpi=300,
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot perturbation calibration results and save figures."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PAIRWISE_METRICS_PATH),
        help="Path to pairwise_value_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/perturbation_calibration_design/plots",
        help="Directory where figures will be written",
    )
    parser.add_argument(
        "--include-no-perturbed",
        action="store_true",
        help=(
            "Include rows with changed_feature_count == 0. "
            "By default these rows are excluded."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    input_path = resolve_repo_path(args.input)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("pairwise_value_metrics.csv is empty.")
    if "gower_distance" not in df.columns:
        raise ValueError("Input CSV must include column: gower_distance")

    if "changed_feature_count" in df.columns and not args.include_no_perturbed:
        df = df[df["changed_feature_count"] > 0].copy()
        if df.empty:
            raise ValueError(
                "No rows left after excluding non-perturbed rows "
                "(changed_feature_count == 0)."
            )

    # Log scales cannot represent zero
    df["gower_is_zero"] = (
        pd.to_numeric(df["gower_distance"], errors="coerce") <= 0
    ).astype(int)
    df["gower_distance_plot"] = pd.to_numeric(df["gower_distance"], errors="coerce")

    # Explicit experiment labels
    df["perturbation_code"] = (
        df["perturbation"].map(PERTURBATION_CODES).fillna(df["perturbation"])
    )
    df["param_setting"] = df.apply(
        lambda row: _format_params(row["perturbation"], row["params"]), axis=1
    )
    df["trial"] = df["perturbation_code"] + " | " + df["param_setting"]

    # Exclude impossible Top-K settings for this setup (only 6 perturbable features).
    df = _filter_topk_by_k_cap(df, k_cap=MAX_VALID_TOPK)

    zero_summary = (
        df.groupby(["perturbation", "sample_group", "param_setting"], dropna=False)
        .agg(
            n_pairs=("gower_is_zero", "size"),
            n_zero_gower=("gower_is_zero", "sum"),
            zero_gower_rate=("gower_is_zero", "mean"),
        )
        .reset_index()
        .sort_values(["perturbation", "sample_group", "param_setting"])
    )
    zero_summary.to_csv(output_dir / "gower_zero_summary.csv", index=False)

    # Every trial in one plot
    df_overview = df.copy()
    overview_order = _trial_order_grouped_by_perturbation(df_overview)
    plt.figure(figsize=(16, 7))
    ax = sns.boxplot(
        data=df_overview,
        x="trial",
        y="gower_distance_plot",
        hue="sample_group",
        order=overview_order,
        showfliers=False,
    )
    ax.set_yscale("log")
    ax.set_title("Parameter Calibration Experiment: Gower Distance by Setting")
    ax.set_xlabel("Perturbation Code | Parameter Setting")
    ax.set_ylabel("Gower Distance (log scale)")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(output_dir / "gower_distance_overview.png", dpi=300)
    plt.close()

    # Figures for each perturbation family separately (GN, DD, TK).
    for perturbation_name in sorted(df["perturbation"].unique()):
        subset = df[df["perturbation"] == perturbation_name].copy()
        if subset.empty:
            continue

        if perturbation_name == "TopKFeatures":
            _plot_topk_grouped(subset, output_dir, perturbation_name)
            continue

        setting_order = _trial_order_by_mean_distance(subset, "param_setting")

        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(
            data=subset,
            x="param_setting",
            y="gower_distance_plot",
            hue="sample_group",
            order=setting_order,
            showfliers=False,
        )
        ax.set_yscale("log")
        ax.set_title(
            f"Parameter Calibration Experiment: {PERTURBATION_CODES.get(perturbation_name, perturbation_name)}"  # noqa: E501
        )
        ax.set_xlabel("Parameter Setting")
        ax.set_ylabel("Gower Distance (log scale s)")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        perturb_code = PERTURBATION_CODES.get(perturbation_name, perturbation_name)
        plt.savefig(output_dir / f"gower_distance_{perturb_code}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
