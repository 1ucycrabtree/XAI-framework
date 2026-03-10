import importlib
from pathlib import Path

for model_file in Path(__file__).parent.glob("*_model.py"):
    if model_file.stem != "base_model":
        importlib.import_module(f"model.{model_file.stem}")
