import importlib
from pathlib import Path

# Auto-import all *_loader.py modules so decorators register themselves
for loader_file in Path(__file__).parent.glob("*_loader.py"):
    importlib.import_module(f"dataset.{loader_file.stem}")
