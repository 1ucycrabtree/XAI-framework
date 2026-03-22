import sys
from pathlib import Path

# Ensure `xai_framework/` is on the path for test imports.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "xai_framework"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
