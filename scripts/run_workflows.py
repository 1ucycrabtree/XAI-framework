from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def _run(cmd: list[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _resolve_config_name(config_name: str) -> str:
    path = CONFIG_DIR / config_name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return config_name


def _train(args: argparse.Namespace) -> None:
    script = REPO_ROOT / "scripts" / "prepare_and_train_model" / "run_pipeline.py"
    cmd = [sys.executable, str(script)]
    if args.tune:
        cmd.append("--tune")
    if args.skip_preprocessing:
        cmd.append("--skip-preprocessing")
    _run(cmd)


def _perturbation(args: argparse.Namespace) -> None:
    cal_script = (
        REPO_ROOT / "scripts" / "perturbation_param_test" / "perturbation_test.py"
    )
    loc_script = (
        REPO_ROOT
        / "scripts"
        / "perturbation_param_test"
        / "perturbation_locality_validation.py"
    )

    cal_cfg = _resolve_config_name(args.calibration_config)
    loc_cfg = _resolve_config_name(args.locality_config)

    _run([sys.executable, str(cal_script), "--config", cal_cfg])
    _run([sys.executable, str(loc_script), "--config", loc_cfg])


def _prune_nested_dirs(dirs: list[Path]) -> list[Path]:
    ordered = sorted({d.resolve() for d in dirs}, key=lambda p: len(p.parts))
    kept: list[Path] = []
    for candidate in ordered:
        if any(candidate.is_relative_to(parent) for parent in kept):
            continue
        kept.append(candidate)
    return kept


def _discover_method_result_dirs(results_root: Path, method_name: str) -> list[Path]:
    candidates: list[Path] = []
    method_token = method_name.lower()
    for d in results_root.rglob("*"):
        if not d.is_dir():
            continue
        if method_token not in d.name.lower():
            continue
        if any(d.rglob("*_result.json")):
            candidates.append(d)
    return _prune_nested_dirs(candidates)


def _plot_results(args: argparse.Namespace) -> None:
    plot_script = REPO_ROOT / "scripts" / "plot_experiment_results.py"
    flip_script = REPO_ROOT / "scripts" / "analyse_prediction_flips.py"
    topk_script = REPO_ROOT / "scripts" / "analyse_topk_perturbation_logs.py"
    results_root = (REPO_ROOT / args.results_root).resolve()
    methods = [m.strip() for m in args.methods if m.strip()]

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    if not plot_script.exists():
        raise FileNotFoundError(f"Plot script not found: {plot_script}")
    if not flip_script.exists():
        raise FileNotFoundError(f"Prediction flip script not found: {flip_script}")
    if not topk_script.exists():
        raise FileNotFoundError(f"Top-K analysis script not found: {topk_script}")

    discovered_any = False
    for method in methods:
        method_dirs = _discover_method_result_dirs(results_root, method)
        if not method_dirs:
            logging.warning("No result directories found for method '%s'.", method)
            continue

        discovered_any = True
        for method_dir in method_dirs:
            flips_out_dir = method_dir / "prediction_flip_analysis"
            topk_out_dir = method_dir / "topk_analysis"
            _run(
                [
                    sys.executable,
                    str(flip_script),
                    "--results-dir",
                    str(method_dir),
                    "--output-dir",
                    str(flips_out_dir),
                ]
            )
            _run(
                [
                    sys.executable,
                    str(topk_script),
                    "--results-dir",
                    str(method_dir),
                    "--output-dir",
                    str(topk_out_dir),
                    "--dedupe",
                ]
            )
            cmd = [
                sys.executable,
                str(plot_script),
                "--results-dir",
                str(method_dir),
                "--prediction-flips-dir",
                str(flips_out_dir),
                "--topk-analysis-dir",
                str(topk_out_dir),
            ]
            _run(cmd)

    if not discovered_any:
        raise ValueError(
            f"No method result directories found under: {results_root}. "
            f"Methods searched: {methods}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XAI project workflows")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser(
        "train",
        help="Run full preprocessing + model training pipeline",
    )
    train_parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable CatBoost grid tuning. If omitted, uses lr=0.01 depth=10.",
    )
    train_parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help=(
            "Skip metadata generation and preprocessing, and train directly "
            "from data/processed files"
        ),
    )
    train_parser.set_defaults(func=_train)

    perturbation_parser = sub.add_parser(
        "perturbation",
        help="Run perturbation calibration and locality validation sequentially",
    )
    perturbation_parser.add_argument(
        "--calibration-config",
        default="perturbation_calibration_ieee_cis.yaml",
        help="Config filename under config/ for calibration run",
    )
    perturbation_parser.add_argument(
        "--locality-config",
        default="perturbation_locality_ieee_cis.yaml",
        help="Config filename under config/ for locality validation run",
    )
    perturbation_parser.set_defaults(func=_perturbation)

    plot_parser = sub.add_parser(
        "plot_results",
        help=(
            "Auto-discover method result folders and generate plots "
            "(e.g. TabularLIME and KernelSHAP)."
        ),
    )
    plot_parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing experiment result folders.",
    )
    plot_parser.add_argument(
        "--methods",
        nargs="+",
        default=["TabularLIME", "KernelSHAP"],
        help="Method names to auto-discover under results root.",
    )
    plot_parser.set_defaults(func=_plot_results)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
