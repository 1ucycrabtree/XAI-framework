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
