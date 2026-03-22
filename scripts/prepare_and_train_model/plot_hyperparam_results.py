#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model hyperparameter tuning outputs"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to model_hyperparameters_*.json or a CSV containing trial rows "
            "(defaults to most recent JSON)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plots/CSVs (defaults next to input file)",
    )
    return parser.parse_args()


def latest_hyperparam_file(base: Path) -> Path:
    files = sorted(
        base.glob("model_hyperparameters_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No model_hyperparameters_*.json found in {base}")
    return files[0]


def build_output_dir(input_file: Path, explicit: str | None) -> Path:
    if explicit:
        out = Path(explicit)
    else:
        out = input_file.parent / f"plots_{input_file.stem}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_trials(path: Path) -> tuple[dict, pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        trials = pd.read_csv(path)
        if trials.empty:
            raise ValueError("Input CSV has no rows.")
        required = {
            "learning_rate",
            "depth",
            "best_iteration",
            "validation_AUPRC",
            "validation_best_F2",
        }
        missing = [c for c in required if c not in trials.columns]
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {', '.join(sorted(missing))}"
            )
        return {}, trials

    obj = json.loads(path.read_text())
    trials = pd.DataFrame(obj.get("hyperparameter_tuning_results", []))
    if trials.empty:
        raise ValueError("No hyperparameter_tuning_results found in JSON.")
    return obj, trials


def plot_tradeoff(trials: pd.DataFrame, out_file: Path, selected: dict) -> None:
    plt.figure(figsize=(8.5, 6))
    lr_values = sorted(trials["learning_rate"].unique())
    marker_map = {}
    if len(lr_values) > 0:
        marker_map[lr_values[0]] = "o"
    if len(lr_values) > 1:
        marker_map[lr_values[1]] = "D"
    for lr in lr_values[2:]:
        marker_map[lr] = "o"

    sns.scatterplot(
        data=trials,
        x="validation_AUPRC",
        y="validation_best_F2",
        hue="depth",
        style="learning_rate",
        markers=marker_map,
        s=150,
        palette="deep",
        alpha=0.9,
    )

    for _, r in trials.iterrows():
        plt.annotate(
            f"{int(r['best_iteration'])}",
            xy=(r["validation_AUPRC"], r["validation_best_F2"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

    plt.title("Validation Trade-off: AUPRC vs Best F2")
    plt.xlabel("Validation AUPRC")
    plt.ylabel("Validation Best F2")
    plt.tight_layout()
    plt.savefig(out_file, dpi=250)
    plt.close()

def plot_best_iteration_vs_f2(
    trials: pd.DataFrame, out_file: Path, selected: dict
) -> None:
    plt.figure(figsize=(8.5, 5.8))
    sns.scatterplot(
        data=trials,
        x="best_iteration",
        y="validation_best_F2",
        hue="depth",
        marker="D",
        s=130,
        palette="deep",
    )

    for _, r in trials.iterrows():
        plt.annotate(
            f"lr={r['learning_rate']}, d={int(r['depth'])}",
            xy=(r["best_iteration"], r["validation_best_F2"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

    plt.title("Best Iteration vs Validation Best F2")
    plt.xlabel("Best Iteration")
    plt.ylabel("Validation Best F2")
    plt.tight_layout()
    plt.savefig(out_file, dpi=250)
    plt.close()


def plot_marginal_gain_if_available(trials: pd.DataFrame, out_file: Path) -> bool:
    rows = []
    for _, r in trials.iterrows():
        segments = r.get("validation_auc_marginal_gain_per_1000", [])
        if isinstance(segments, list):
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                rows.append(
                    {
                        "learning_rate": r.get("learning_rate"),
                        "depth": r.get("depth"),
                        "to_iteration": seg.get("to_iteration"),
                        "delta_auc_per_1000_trees": seg.get("delta_auc_per_1000_trees"),
                    }
                )

    if not rows:
        return False

    mg = pd.DataFrame(rows).dropna()
    if mg.empty:
        return False

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=mg.sort_values("to_iteration"),
        x="to_iteration",
        y="delta_auc_per_1000_trees",
        hue="depth",
        style="learning_rate",
        markers=True,
        dashes=False,
    )

    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Marginal AUC Gain per 1000 Trees")
    plt.xlabel("iteration")
    plt.ylabel("delta_auc_per_1000_trees")
    plt.tight_layout()
    plt.savefig(out_file, dpi=250)
    plt.close()
    return True


def main() -> None:
    args = parse_args()

    eval_dir = Path("data/models/evaluation_outputs")
    input_path = Path(args.input) if args.input else latest_hyperparam_file(eval_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    obj, trials = load_trials(input_path)
    out_dir = build_output_dir(input_path, args.output_dir)

    trials["learning_rate"] = pd.to_numeric(trials["learning_rate"])
    trials["depth"] = pd.to_numeric(trials["depth"]).astype(int)
    if "best_iteration" in trials.columns:
        trials["best_iteration"] = pd.to_numeric(trials["best_iteration"]).astype(int)

    selected = obj.get("selected_hyperparameters", {}) or {}

    trials_sorted = trials.sort_values(
        ["validation_best_F2", "validation_AUPRC"], ascending=False
    )
    trials_sorted.to_csv(out_dir / "hyperparameter_trials_table.csv", index=False)

    plot_tradeoff(trials, out_dir / "scatter_AUPRC_vs_bestF2.png", selected)

    if "best_iteration" in trials.columns:
        plot_best_iteration_vs_f2(
            trials, out_dir / "scatter_best_iteration_vs_F2.png", selected
        )


if __name__ == "__main__":
    main()
