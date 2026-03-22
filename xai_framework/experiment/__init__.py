import importlib
from pathlib import Path

for experiment_file in Path(__file__).parent.glob("*_experiment.py"):
    if experiment_file.stem != "base_experiment":
        importlib.import_module(f"experiment.{experiment_file.stem}")
