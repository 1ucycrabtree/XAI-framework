from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]

LOCAL_METRICS = {
    "RIS": "RelativeInputStability",
    "RBO": "RankBiasedOverlap",
    "SCR": "SignConsistencyRate",
}
GLOBAL_METRICS = {
    "GC": "GlobalConsistencyMetric",
    "GS": "GlobalSufficiencyMetric",
}


PERTURBATION_NAME_MAP = {
    "localgaussiannoise": "Noise",
    "directionaldrift": "Drift",
    "topkfeatures": "K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot experiment result JSON files (RIS/RBO/SCR/GC/GS) from an experiment "
            "results directory."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/DissertationExperiment_TabularLIME",
        help="Directory containing one run subdirectory per experiment execution.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots/CSVs (default: <results-dir>/plots).",
    )
    parser.add_argument(
        "--prediction-flips-dir",
        type=str,
        default=None,
        help=(
            "Directory containing prediction flip CSVs from analyse_prediction_flips.py "
            "(default: <results-dir>/prediction_flip_analysis)."
        ),
    )
    parser.add_argument(
        "--topk-analysis-dir",
        type=str,
        default=None,
        help=(
            "Directory containing Top-K analysis CSVs from "
            "analyse_topk_perturbation_logs.py (default: <results-dir>/topk_analysis)."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def normalise_perturbation(raw: str) -> str:
    token = raw.strip().lower().replace("_", "")
    return PERTURBATION_NAME_MAP.get(token, raw)


def parse_run_folder_name(run_dir_name: str) -> dict[str, str | None]:
    # <experiment>_<method>_<sample>_<perturbation>_run_<timestamp>
    parts = run_dir_name.split("_")
    method = parts[1] if len(parts) > 1 else None
    sample_group = parts[2] if len(parts) > 2 else None
    perturbation = parts[3] if len(parts) > 3 else None
    return {
        "method": method,
        "sample_group": sample_group,
        "perturbation": normalise_perturbation(perturbation) if perturbation else None,
    }


def parse_result_file_name(result_file_name: str) -> dict[str, str | None]:
    # <method>_<perturbation>_result.json
    stem = Path(result_file_name).stem
    parts = stem.split("_")
    method = parts[0] if len(parts) > 0 else None
    perturbation = parts[1] if len(parts) > 1 else None
    return {
        "file_method": method,
        "file_perturbation": normalise_perturbation(perturbation)
        if perturbation
        else None,
    }


def list_result_jsons(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*_result.json"))


def _first_list_value(payload: dict, key: str) -> float | None:
    values = payload.get(key, [])
    if not isinstance(values, list) or not values:
        return None
    value = values[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_with_ids_rows(
    payload: dict,
    with_ids_key: str,
    value_key: str,
) -> list[dict]:
    rows: list[dict] = []
    items = payload.get(with_ids_key, [])
    if not isinstance(items, list):
        return rows
    for pair_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        val = item.get(value_key)
        inst = item.get("instance_id")
        if val is None:
            continue
        try:
            rows.append(
                {
                    "instance_id": inst,
                    "pair_index": pair_index,
                    "value": float(val),
                }
            )
        except (TypeError, ValueError):
            continue
    return rows


def build_dataframes(
    result_files: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local_dist_rows: list[dict] = []
    local_summary_rows: list[dict] = []
    global_rows: list[dict] = []

    for file_path in result_files:
        run_meta = parse_run_folder_name(file_path.parent.name)
        file_meta = parse_result_file_name(file_path.name)

        with file_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        metrics = obj.get("metrics", {})

        context = {
            "result_file": str(file_path),
            "run_dir": file_path.parent.name,
            "experiment_name": obj.get("experiment_name"),
            "method": run_meta.get("method") or file_meta.get("file_method"),
            "sample_group": run_meta.get("sample_group"),
            "perturbation": run_meta.get("perturbation")
            or file_meta.get("file_perturbation"),
        }

        for short_name, long_name in LOCAL_METRICS.items():
            with_ids_key = f"{long_name}_with_ids"
            mean_key = f"{long_name}_mean"
            std_key = f"{long_name}_std"
            min_key = f"{long_name}_min"
            max_key = f"{long_name}_max"
            n_key = f"{long_name}_n_instances"

            dist_rows = _extract_with_ids_rows(metrics, with_ids_key, long_name)
            for row in dist_rows:
                local_dist_rows.append(
                    {
                        **context,
                        "metric": short_name,
                        "instance_id": row["instance_id"],
                        "pair_index": row["pair_index"],
                        "value": row["value"],
                    }
                )

            local_summary_rows.append(
                {
                    **context,
                    "metric": short_name,
                    "mean": _first_list_value(metrics, mean_key),
                    "std": _first_list_value(metrics, std_key),
                    "min": _first_list_value(metrics, min_key),
                    "max": _first_list_value(metrics, max_key),
                    "n_instances": _first_list_value(metrics, n_key),
                }
            )

        for short_name, long_name in GLOBAL_METRICS.items():
            baseline = _first_list_value(metrics, f"{long_name}_baseline")
            perturbed = _first_list_value(metrics, f"{long_name}_perturbed")
            if baseline is not None:
                global_rows.append(
                    {
                        **context,
                        "metric": short_name,
                        "phase": "baseline",
                        "value": baseline,
                    }
                )
            if perturbed is not None:
                global_rows.append(
                    {
                        **context,
                        "metric": short_name,
                        "phase": "perturbed",
                        "value": perturbed,
                    }
                )

    local_dist_df = pd.DataFrame(local_dist_rows)
    local_summary_df = pd.DataFrame(local_summary_rows)
    global_df = pd.DataFrame(global_rows)
    return local_dist_df, local_summary_df, global_df


def load_retention_from_prediction_flips(prediction_flips_dir: Path) -> pd.DataFrame:
    required_cols = {"sample_group", "perturbation", "instance_id", "changed_rate"}

    def _postprocess(_df: pd.DataFrame) -> pd.DataFrame:
        out = _df.copy()
        out["perturbation"] = out["perturbation"].apply(
            lambda x: normalise_perturbation(str(x)) if pd.notna(x) else x
        )
        out["changed_rate"] = pd.to_numeric(out["changed_rate"], errors="coerce")
        out["retention_rate"] = 1.0 - out["changed_rate"]
        return out

    csv_path = prediction_flips_dir / "changed_instances_all.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if required_cols.issubset(set(df.columns)):
            df = _postprocess(df)
            n_pert = df["perturbation"].dropna().nunique()
            if n_pert >= 2:
                return df

    parts: list[pd.DataFrame] = []
    for p in sorted(prediction_flips_dir.glob("changed_instances_*.csv")):
        name = p.stem
        if name in {
            "changed_instances_all",
            "changed_instances_TP",
            "changed_instances_FP",
        }:
            continue
        suffix = name.replace("changed_instances_", "")
        if suffix.endswith("_TP") or suffix.endswith("_FP"):
            continue
        df_part = pd.read_csv(p)
        if required_cols.issubset(set(df_part.columns)):
            parts.append(df_part)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    return _postprocess(df)


def load_topk_feature_frequency(topk_analysis_dir: Path) -> pd.DataFrame:
    csv_path = topk_analysis_dir / "topk_feature_frequency.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    required = {"sample_group", "feature", "instance_frequency"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    return df


def load_topk_unique_instances(topk_analysis_dir: Path) -> pd.DataFrame:
    csv_path = topk_analysis_dir / "topk_unique_instances_by_sample_group.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    required = {"sample_group", "instance_id", "selected_features"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    return df


def _cat_order_if_present(
    df: pd.DataFrame, col: str, desired: list[str]
) -> list[str] | None:
    if col not in df.columns:
        return None
    present_values = [v for v in df[col].dropna().unique().tolist() if v is not None]
    present_values = [str(v).capitalize() for v in present_values]
    if not present_values:
        return None
    ordered_desired = [v for v in desired if v in set(present_values)]
    extras = [v for v in present_values if v not in set(ordered_desired)]
    return ordered_desired + extras


def plot_local_distributions(
    method: str, local_dist_df: pd.DataFrame, out_dir: Path
) -> None:
    if local_dist_df.empty:
        return

    pert_order = _cat_order_if_present(
        local_dist_df, "perturbation", list(PERTURBATION_NAME_MAP.values())
    )

    g = sns.catplot(
        data=local_dist_df,
        kind="box",
        x="perturbation",
        y="value",
        hue="sample_group",
        col="metric",
        col_order=[
            m for m in ["RIS", "RBO", "SCR"] if m in set(local_dist_df["metric"])
        ],
        order=pert_order,
        showfliers=False,
        height=4,
        aspect=1.1,
        sharey=False,
    )
    g.set_axis_labels("Perturbation", "Metric value")
    g.figure.subplots_adjust(top=0.84)
    g.figure.suptitle(f"{method} Stability Metrics by Perturbation and Sample Group")
    g.savefig(out_dir / f"{method}_stability_metrics_by_metric_boxplots.png", dpi=300)
    plt.close(g.figure)


def plot_ris_vs_rbo_scatter(
    method: str, local_dist_df: pd.DataFrame, out_dir: Path
) -> None:
    if local_dist_df.empty:
        return

    df = local_dist_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    piv = (
        df[df["metric"].isin(["RIS", "RBO"])]
        .copy()
        .pivot_table(
            index=[
                "run_dir",
                "result_file",
                "method",
                "sample_group",
                "perturbation",
                "instance_id",
                "pair_index",
            ],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    if piv.empty or "RIS" not in piv.columns or "RBO" not in piv.columns:
        return

    piv = piv.dropna(subset=["RIS", "RBO"])
    if piv.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    panel_groups = ["TP", "FP"]
    n_methods = piv["method"].nunique(dropna=True)

    for ax, group in zip(axes, panel_groups):
        sub = piv[piv["sample_group"] == group].copy()
        if sub.empty:
            ax.set_title(group)
            ax.set_xlabel("RIS")
            ax.set_ylabel("RBO")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        scatter_kwargs = {
            "data": sub,
            "x": "RIS",
            "y": "RBO",
            "hue": "perturbation",
            "s": 18,
            "alpha": 0.5,
            "ax": ax,
        }
        if n_methods > 1:
            scatter_kwargs["style"] = "method"

        sns.scatterplot(**scatter_kwargs)
        ax.set_title(group)
        ax.set_xlabel("RIS")
        ax.set_ylabel("RBO")
        legend = ax.get_legend()
        if legend is not None:
            legend.set_loc("lower right")

    fig.suptitle(f"{method}: RIS vs RBO (per instance)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{method}_ris_vs_rbo_scatter_per_instance.png", dpi=300)
    plt.close()


def plot_rbo_conditioned_on_ris(
    method: str, local_dist_df: pd.DataFrame, out_dir: Path
) -> None:
    if local_dist_df.empty:
        return

    df = local_dist_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    piv = (
        df[df["metric"].isin(["RIS", "RBO"])]
        .copy()
        .pivot_table(
            index=[
                "run_dir",
                "result_file",
                "method",
                "sample_group",
                "perturbation",
                "instance_id",
                "pair_index",
            ],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    if piv.empty or "RIS" not in piv.columns or "RBO" not in piv.columns:
        return

    piv = piv.dropna(subset=["RIS", "RBO"]).copy()
    if piv.empty:
        return

    ris_bins = [0.0, 0.02, 0.05, 0.10, 0.20, 1.0]
    ris_labels = ["0-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", "0.20+"]
    piv["ris_bin"] = pd.cut(
        piv["RIS"],
        bins=ris_bins,
        labels=ris_labels,
        include_lowest=True,
        right=False,
    )
    piv = piv.dropna(subset=["ris_bin"])
    if piv.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    panel_groups = ["TP", "FP"]
    for ax, group in zip(axes, panel_groups):
        sub = piv[piv["sample_group"] == group].copy()
        if sub.empty:
            ax.set_title(group)
            ax.set_xlabel("RIS bin")
            ax.set_ylabel("RBO")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        sns.boxplot(
            data=sub,
            x="ris_bin",
            y="RBO",
            hue="perturbation",
            order=ris_labels,
            showfliers=False,
            ax=ax,
        )
        ax.set_title(group)
        ax.set_xlabel("RIS bin")
        ax.set_ylabel("RBO")
        legend = ax.get_legend()
        if legend is not None:
            legend.set_loc("lower right")

    fig.suptitle(f"{method}: Distribution of RBO conditioned on RIS bins")
    plt.tight_layout()
    plt.savefig(out_dir / f"{method}_rbo_conditioned_on_ris_bins_boxplot.png", dpi=300)
    plt.close()


def plot_metric_correlation_heatmap(
    method: str,
    local_dist_df: pd.DataFrame,
    global_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if local_dist_df.empty or global_df.empty:
        return

    local_piv = (
        local_dist_df[local_dist_df["metric"].isin(["RIS", "RBO", "SCR"])]
        .copy()
        .pivot_table(
            index=[
                "run_dir",
                "result_file",
                "method",
                "sample_group",
                "perturbation",
                "instance_id",
                "pair_index",
            ],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    if local_piv.empty:
        return

    global_piv = (
        global_df[global_df["phase"] == "perturbed"]
        .copy()
        .pivot_table(
            index=["run_dir", "result_file", "method", "sample_group", "perturbation"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    if global_piv.empty:
        return

    merged = local_piv.merge(
        global_piv,
        on=["run_dir", "result_file", "method", "sample_group", "perturbation"],
        how="left",
    )

    corr_cols = [c for c in ["RIS", "RBO", "SCR", "GC", "GS"] if c in merged.columns]
    if len(corr_cols) < 2:
        return

    panel_groups = ["TP", "FP"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), sharex=False, sharey=False)

    for ax, group in zip(axes, panel_groups):
        sub = merged[merged["sample_group"] == group].copy()
        if sub.empty:
            ax.set_title(group)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")
            continue

        corr = (
            sub[corr_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        )

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-1,
            vmax=1,
            square=True,
            cbar=(group == "FP"),
            cbar_kws={"label": "Spearman correlation", "shrink": 0.85},
            ax=ax,
        )
        ax.set_title(group)

    fig.suptitle(f"{method}: Metric Correlation Heatmap by Sample Group")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{method}_metric_correlation_heatmap_by_sample_group.png", dpi=300
    )
    plt.close()


def plot_faithfulness_summary(
    method: str,
    global_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if global_df.empty:
        return

    faith_df = global_df[
        (global_df["phase"] == "perturbed") & (global_df["metric"].isin(["GC", "GS"]))
    ].copy()
    if faith_df.empty:
        return

    pert_order = _cat_order_if_present(
        faith_df, "perturbation", list(PERTURBATION_NAME_MAP.values())
    )
    g = sns.catplot(
        data=faith_df,
        kind="bar",
        x="perturbation",
        y="value",
        hue="sample_group",
        col="metric",
        col_order=[m for m in ["GC", "GS"] if m in set(faith_df["metric"])],
        order=pert_order,
        height=4,
        aspect=1.1,
        sharey=False,
        errorbar=None,
    )
    g.set_axis_labels("Perturbation", "Faithfulness score")
    for ax in g.axes.flatten():
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=6)
    g.figure.subplots_adjust(top=0.84)
    g.figure.suptitle(f"{method}: Faithfulness Metrics by Perturbation (TP vs FP)")
    g.savefig(out_dir / f"{method}_faithfulness_metrics_barplots.png", dpi=300)
    plt.close(g.figure)


def plot_retention_rate_summary(
    method: str,
    retention_instance_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if retention_instance_df.empty:
        return

    df = retention_instance_df.copy()
    df["retention_rate"] = pd.to_numeric(df["retention_rate"], errors="coerce")
    df = df.dropna(subset=["retention_rate"])
    if df.empty:
        return

    pert_order = _cat_order_if_present(
        df, "perturbation", list(PERTURBATION_NAME_MAP.values())
    )
    if not pert_order:
        return

    fig, axes = plt.subplots(
        1, len(pert_order), figsize=(4.2 * len(pert_order), 5.0), sharey=False
    )
    if len(pert_order) == 1:
        axes = [axes]
    group_order = ["TP", "FP"]
    group_palette = {"TP": "#f28e2b", "FP": "#4e79a7"}

    for i, pert in enumerate(pert_order):
        ax = axes[i]
        sub = df[df["perturbation"] == pert].copy()
        if sub.empty:
            ax.set_title(pert)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_xlabel("sample_group")
            if i == 0:
                ax.set_ylabel("Retention rate (per instance)")
            continue

        present_groups = [
            g
            for g in group_order
            if g in set(sub["sample_group"].dropna().unique().tolist())
        ]
        sns.boxplot(
            data=sub,
            x="sample_group",
            y="retention_rate",
            order=present_groups,
            hue="sample_group",
            showfliers=True,
            width=0.55,
            linewidth=1.0,
            palette=group_palette,
            legend=False,
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "#333333",
                "markeredgecolor": "#333333",
                "alpha": 0.55,
            },
            ax=ax,
        )
        ax.set_title(pert)
        ax.set_xlabel("sample_group")
        if i == 0:
            ax.set_ylabel("Retention rate (per instance)")
        else:
            ax.set_ylabel("")

    legend_handles = [
        Patch(facecolor=group_palette["TP"], edgecolor="black", label="TP"),
        Patch(facecolor=group_palette["FP"], edgecolor="black", label="FP"),
    ]
    fig.legend(handles=legend_handles, title="sample_group", loc="upper right")

    fig.suptitle(f"{method}: Per-Instance Retention Rate by Perturbation")
    plt.tight_layout()
    plt.savefig(out_dir / f"{method}_retention_rate_boxplot_minipanels.png", dpi=300)
    plt.close()


def plot_topk_feature_frequency(
    method: str,
    topk_feature_freq_df: pd.DataFrame,
    topk_unique_instances_df: pd.DataFrame,
    prediction_flips_dir: Path,
    out_dir: Path,
    top_n: int = 6,
) -> None:
    if topk_feature_freq_df.empty:
        return

    df = topk_feature_freq_df.copy()
    df = df[df["sample_group"].isin(["TP", "FP"])].copy()
    if df.empty:
        return

    df["unique_instances_selected"] = pd.to_numeric(
        df.get("unique_instances_selected"),  # type: ignore
        errors="coerce",
    )
    if (
        "unique_instances_selected" in df.columns
        and not df["unique_instances_selected"].isna().all()
    ):
        df["pct_of_500"] = (df["unique_instances_selected"] / 500.0) * 100.0
    else:
        df["instance_frequency"] = pd.to_numeric(
            df["instance_frequency"], errors="coerce"
        )
        df["pct_of_500"] = df["instance_frequency"] * 100.0
    df = df.dropna(subset=["feature", "pct_of_500"])
    if df.empty:
        return

    all_rows = topk_feature_freq_df[
        topk_feature_freq_df["sample_group"] == "ALL"
    ].copy()
    if not all_rows.empty:
        all_rows["unique_instances_selected"] = pd.to_numeric(
            all_rows.get("unique_instances_selected"),  # type: ignore
            errors="coerce",
        )
        if (
            "unique_instances_selected" in all_rows.columns
            and not all_rows["unique_instances_selected"].isna().all()
        ):
            all_rows["pct_of_500"] = (
                all_rows["unique_instances_selected"] / 500.0
            ) * 100.0
        else:
            all_rows["instance_frequency"] = pd.to_numeric(
                all_rows["instance_frequency"], errors="coerce"
            )
            all_rows["pct_of_500"] = all_rows["instance_frequency"] * 100.0
        all_rows = all_rows.dropna(subset=["feature", "pct_of_500"])
        top_features = (
            all_rows.sort_values("pct_of_500", ascending=False)
            .head(top_n)["feature"]
            .tolist()
        )
    else:
        top_features = (
            df.groupby("feature", as_index=False)["pct_of_500"]
            .mean()
            .sort_values("pct_of_500", ascending=False)  # type: ignore
            .head(top_n)["feature"]
            .tolist()
        )
    if not top_features:
        return

    plot_df = df[df["feature"].isin(top_features)].copy()
    plot_df["feature"] = pd.Categorical(
        plot_df["feature"], categories=top_features, ordered=True
    )
    group_palette = {"TP": "#f28e2b", "FP": "#4e79a7"}

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    sns.barplot(
        data=plot_df,
        x="feature",
        y="pct_of_500",
        hue="sample_group",
        hue_order=["TP", "FP"],
        palette=group_palette,
        ax=ax,
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("% appearance out of all instances")
    ax.set_ylim(0.0, min(100.0, max(5.0, plot_df["pct_of_500"].max() * 1.15)))
    ax.set_title(f"{method}: Top-K Feature Frequency by Sample Group (% of instances)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="sample_group", loc="upper right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=2, fontsize=6)  # pyright: ignore[reportArgumentType]
    plt.tight_layout()
    plt.savefig(out_dir / f"{method}_topk_feature_frequency_grouped_bar.png", dpi=300)
    plt.close()

    if topk_unique_instances_df.empty:
        return

    flips_k_path = prediction_flips_dir / "changed_instances_K.csv"
    if not flips_k_path.exists():
        return
    flips_df = pd.read_csv(flips_k_path)
    needed = {
        "sample_group",
        "instance_id",
        "changed_rows",
        "total_perturbation_rows",
    }
    if not needed.issubset(set(flips_df.columns)):
        return

    sel = topk_unique_instances_df.copy()
    sel = sel[sel["sample_group"].isin(["TP", "FP"])].copy()
    if sel.empty:
        return
    sel["instance_id"] = sel["instance_id"].astype(str)
    sel["selected_features"] = sel["selected_features"].fillna("").astype(str)
    sel["feature"] = sel["selected_features"].str.split(",")
    sel = sel.explode("feature")
    sel["feature"] = sel["feature"].astype(str).str.strip()
    sel = sel[(sel["feature"] != "") & (sel["feature"].isin(top_features))].copy()
    if sel.empty:
        return

    flips_df = flips_df.copy()
    flips_df = flips_df[flips_df["sample_group"].isin(["TP", "FP"])].copy()
    flips_df["instance_id"] = flips_df["instance_id"].astype(str)
    flips_df["changed_rows"] = pd.to_numeric(flips_df["changed_rows"], errors="coerce")
    flips_df["total_perturbation_rows"] = pd.to_numeric(
        flips_df["total_perturbation_rows"], errors="coerce"
    )
    flips_df = flips_df.dropna(subset=["changed_rows", "total_perturbation_rows"])
    if flips_df.empty:
        return

    merged = sel.merge(
        flips_df[
            ["sample_group", "instance_id", "changed_rows", "total_perturbation_rows"]
        ],
        on=["sample_group", "instance_id"],
        how="left",
    )
    merged["changed_rows"] = merged["changed_rows"].fillna(0.0)
    merged["total_perturbation_rows"] = merged["total_perturbation_rows"].fillna(0.0)

    agg = (
        merged.groupby(["sample_group", "feature"], as_index=False)[
            ["changed_rows", "total_perturbation_rows"]
        ]
        .sum()
        .copy()
    )
    agg["flip_conditioned_frequency"] = agg.apply(
        lambda r: (
            float(r["changed_rows"]) / float(r["total_perturbation_rows"])
            if float(r["total_perturbation_rows"]) > 0
            else 0.0
        ),
        axis=1,
    )
    agg["flip_conditioned_pct"] = agg["flip_conditioned_frequency"] * 100.0
    agg = agg[agg["feature"].isin(top_features)].copy()
    if agg.empty:
        return
    agg["feature"] = pd.Categorical(
        agg["feature"], categories=top_features, ordered=True
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    sns.barplot(
        data=agg,
        x="feature",
        y="flip_conditioned_pct",
        hue="sample_group",
        hue_order=["TP", "FP"],
        palette=group_palette,
        ax=ax,
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("% of flip-causing perturbations")
    ax.set_ylim(
        0.0,
        min(100.0, max(5.0, agg["flip_conditioned_pct"].max() * 1.15)),
    )
    ax.set_title(f"{method}: Top-K Flip-Conditioned Feature Frequency")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="sample_group", loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=2, fontsize=6)  # pyright: ignore[reportArgumentType]
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{method}_topk_feature_flip_conditioned_frequency_grouped_bar.png",
        dpi=300,
    )
    plt.close()


def main() -> None:
    args = parse_args()
    sns.set_style("whitegrid")

    results_dir = resolve_repo_path(args.results_dir)
    if args.output_dir is None:
        out_dir = results_dir / "plots"
    else:
        out_dir = resolve_repo_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_flips_dir = (
        resolve_repo_path(args.prediction_flips_dir)
        if args.prediction_flips_dir
        else results_dir / "prediction_flip_analysis"
    )
    topk_analysis_dir = (
        resolve_repo_path(args.topk_analysis_dir)
        if args.topk_analysis_dir
        else results_dir / "topk_analysis"
    )

    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    result_files = list_result_jsons(results_dir)
    if not result_files:
        raise ValueError(f"No *_result.json files found under: {results_dir}")

    local_dist_df, local_summary_df, global_df = build_dataframes(result_files)
    retention_instance_df = load_retention_from_prediction_flips(prediction_flips_dir)
    topk_feature_freq_df = load_topk_feature_frequency(topk_analysis_dir)
    topk_unique_instances_df = load_topk_unique_instances(topk_analysis_dir)
    method = (
        local_dist_df["method"].iloc[0] if not local_dist_df.empty else "UnknownMethod"
    )
    plot_local_distributions(method, local_dist_df, out_dir)
    plot_ris_vs_rbo_scatter(method, local_dist_df, out_dir)
    plot_rbo_conditioned_on_ris(method, local_dist_df, out_dir)
    plot_retention_rate_summary(method, retention_instance_df, out_dir)
    plot_faithfulness_summary(method, global_df, out_dir)
    plot_metric_correlation_heatmap(method, local_dist_df, global_df, out_dir)
    plot_topk_feature_frequency(
        method,
        topk_feature_freq_df,
        topk_unique_instances_df,
        prediction_flips_dir,
        out_dir,
    )

    print(f"Processed {len(result_files)} result files.")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
