from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def _cat_order_if_present(
    df: pd.DataFrame, col: str, desired: list[str]
) -> list[str] | None:
    if col not in df.columns:
        return None
    present_values = [v for v in df[col].dropna().unique().tolist() if v is not None]
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


def main() -> None:
    args = parse_args()
    sns.set_style("whitegrid")

    results_dir = resolve_repo_path(args.results_dir)
    if args.output_dir is None:
        out_dir = results_dir / "plots"
    else:
        out_dir = resolve_repo_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    result_files = list_result_jsons(results_dir)
    if not result_files:
        raise ValueError(f"No *_result.json files found under: {results_dir}")

    local_dist_df, local_summary_df, global_df = build_dataframes(result_files)
    method = (
        local_dist_df["method"].iloc[0] if not local_dist_df.empty else "UnknownMethod"
    )
    plot_local_distributions(method, local_dist_df, out_dir)
    plot_ris_vs_rbo_scatter(method, local_dist_df, out_dir)
    plot_rbo_conditioned_on_ris(method, local_dist_df, out_dir)
    plot_faithfulness_summary(method, global_df, out_dir)
    plot_metric_correlation_heatmap(method, local_dist_df, global_df, out_dir)

    print(f"Processed {len(result_files)} result files.")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
