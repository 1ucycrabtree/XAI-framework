import importlib
from pathlib import Path

for metric_file in Path(__file__).parent.glob("*_metric.py"):
    if metric_file.stem != "base_metric":
        importlib.import_module(f"metric.{metric_file.stem}")
