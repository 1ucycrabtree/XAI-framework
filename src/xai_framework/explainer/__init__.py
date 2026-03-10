import importlib
from pathlib import Path

for explainer_file in Path(__file__).parent.glob("*_explainer.py"):
    if explainer_file.stem != "base_explainer":
        importlib.import_module(f"explainer.{explainer_file.stem}")
