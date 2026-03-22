import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable


def write_json(
    path: Path | str,
    payload: dict[str, Any],
    *,
    indent: int = 2,
    default: Callable[[Any], Any] | None = None,
) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent, default=default)


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    indent: int = 2,
    default: Callable[[Any], Any] | None = None,
) -> None:
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=indent, default=default)
        tmp.flush()
        Path(tmp.name).replace(path)


def read_json(path: Path | str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
