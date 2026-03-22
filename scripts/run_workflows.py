from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import uuid
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


def _write_temp_experiment_config(
    base_config_name: str, explainer_method: str, run_suffix: str
) -> str:
    import yaml

    base_path = CONFIG_DIR / base_config_name
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Base config must be a mapping: {base_path}")
    if "explainer" not in cfg or "experiment" not in cfg:
        raise ValueError(
            "Base config missing required top-level keys: "
            f"explainer/experiment ({base_path})"
        )

    cfg["explainer"]["method"] = explainer_method
    experiment_name = str(cfg["experiment"].get("name", "Experiment"))
    cfg["experiment"]["name"] = f"{experiment_name}_{explainer_method}_{run_suffix}"

    tmp_name = f".tmp_{Path(base_config_name).stem}_{explainer_method}_{uuid.uuid4().hex[:8]}.yaml"  # noqa: E501
    tmp_path = CONFIG_DIR / tmp_name
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_name


def _experiments(args: argparse.Namespace) -> None:
    configs_to_run: list[str] = []
    generated_configs: list[str] = []

    if args.configs:
        for cfg_name in args.configs:
            configs_to_run.append(_resolve_config_name(cfg_name))
    else:
        base_cfg = _resolve_config_name(args.base_config)
        for method in args.explainers:
            generated = _write_temp_experiment_config(base_cfg, method, args.run_suffix)
            generated_configs.append(generated)
            configs_to_run.append(generated)

    try:
        for cfg_name in configs_to_run:
            _run([sys.executable, "-m", "xai_framework", "--config", cfg_name])
    finally:
        if not args.keep_temp_configs:
            for cfg_name in generated_configs:
                path = CONFIG_DIR / cfg_name
                if path.exists():
                    path.unlink()


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

    experiments_parser = sub.add_parser(
        "experiments",
        help="Run explanation experiments sequentially",
    )
    experiments_parser.add_argument(
        "--configs",
        nargs="+",
        help=(
            "Explicit config filenames under config/ to run in order. "
            "If set, --base-config/--explainers are ignored."
        ),
    )
    experiments_parser.add_argument(
        "--base-config",
        default="default.yaml",
        help=(
            "Base config filename under config/ used to generate per-explainer configs"
        ),
    )
    experiments_parser.add_argument(
        "--explainers",
        nargs="+",
        default=["KernelSHAP", "TabularLIME"],
        help="Explainer methods for generated sequential runs",
    )
    experiments_parser.add_argument(
        "--run-suffix",
        default="seq",
        help="Suffix appended to generated experiment names",
    )
    experiments_parser.add_argument(
        "--keep-temp-configs",
        action="store_true",
        help="Keep generated temporary config files under config/",
    )
    experiments_parser.set_defaults(func=_experiments)

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
