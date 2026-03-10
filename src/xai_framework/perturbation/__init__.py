import importlib
from pathlib import Path

for perturbation_file in Path(__file__).parent.glob("*_perturbation.py"):
    if perturbation_file.stem != "base_perturbation":
        importlib.import_module(f"perturbation.{perturbation_file.stem}")
