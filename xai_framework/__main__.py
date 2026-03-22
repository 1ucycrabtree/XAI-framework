import sys
from pathlib import Path

# Preserve existing intra-package imports that use top-level module names
# (e.g., `from dataset...`) when running with `python -m xai_framework`.
PACKAGE_DIR = str(Path(__file__).resolve().parent)
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from .main import main  # noqa: E402

if __name__ == "__main__":
    main()
