import importlib
from pathlib import Path

for model_file in Path(__file__).parent.glob("*_model.py"):
    importlib.import_module(f"model.{model_file.stem}")
