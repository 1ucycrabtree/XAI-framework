import json
import logging
import math
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from load_config import ExperimentConfig


class BaseExperiment(ABC):
    def __init__(self, cfg: ExperimentConfig, name: str):
        self.cfg = cfg
        self.name = name

        self.chunk_size = self.cfg.chunk_size
        # Concrete defaults avoid "possibly unbound attribute" type errors.
        self.run_root: Path = Path(".")
        self.chunks_dir: Path = Path(".")
        self.run_manifest_path: Path = Path(".")
        self.sampled_instances_path: Path = Path(".")
        self.chunk_manifest_path: Path = Path(".")
        self.results_dir: Path = Path(".")
        self._chunking_paths_initialised = False

    def _default_run_id(self) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", self.cfg.name).strip("_")
        if not safe_name:
            safe_name = "experiment"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_run_{timestamp}"

    def initialise_chunking_paths(self, stage_id: str) -> None:
        run_id = self.cfg.run_id or self._default_run_id()
        self.run_root = Path(self.cfg.checkpoint_dir) / run_id / stage_id
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.run_root / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self.run_manifest_path = self.run_root / "run_manifest.json"
        self.sampled_instances_path = self.run_root / "sampled_instances.parquet"
        self.chunk_manifest_path = self.run_root / "chunk_manifest.parquet"
        self.results_dir = Path(self.cfg.results_dir) / run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._chunking_paths_initialised = True

    def _require_chunking_paths(self) -> None:
        if not self._chunking_paths_initialised:
            raise RuntimeError(
                "Chunking paths are not initialised. Call initialise_chunking_paths() first."  # noqa: E501
            )

    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _json_default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj is pd.NA:
            return None
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serialisable"
        )

    def _write_json_atomic(self, path: Path, data: dict[str, Any]) -> None:
        with NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
            json.dump(data, tmp, indent=2, default=self._json_default)
            tmp.flush()
            Path(tmp.name).replace(path)

    def _load_or_create_run_manifest(
        self, random_seed: int, config_hash: str
    ) -> dict[str, Any]:
        self._require_chunking_paths()
        if self.run_manifest_path.exists():
            logging.info(
                "Found existing run manifest at %s. resume=%s",
                self.run_manifest_path,
                self.cfg.resume,
            )
            if not self.cfg.resume:
                raise ValueError(
                    "Checkpoint exists but experiment.resume is false. "
                    "Set experiment.resume=true to continue this run."
                )
            with open(self.run_manifest_path, "r") as f:
                manifest = json.load(f)
            if manifest["seed"] != random_seed:
                raise ValueError(
                    "Resume aborted: checkpoint seed does not match current config."
                )
            if manifest["config_hash"] != config_hash:
                raise ValueError(
                    "Resume aborted: checkpoint config hash does not match current config. "  # noqa: E501
                    f"checkpoint_hash={manifest['config_hash']} current_hash={config_hash}. "  # noqa: E501
                    "If you changed config, use a new run_id or set resume=false."
                )
            logging.info(
                "Resuming run from checkpoint. run_root=%s status=%s",
                manifest.get("run_root"),
                manifest.get("status"),
            )
            return manifest

        manifest = {
            "run_root": str(self.run_root),
            "config_hash": config_hash,
            "seed": random_seed,
            "status": "running",
            "created_at": self._utcnow(),
            "updated_at": self._utcnow(),
        }
        self._write_json_atomic(self.run_manifest_path, manifest)
        logging.info("Created new run manifest at %s", self.run_manifest_path)
        return manifest

    def _save_chunk_manifest(self, chunk_df: pd.DataFrame) -> None:
        self._require_chunking_paths()
        tmp_path = self.chunk_manifest_path.with_suffix(".tmp.parquet")
        chunk_df.to_parquet(tmp_path, index=False)
        tmp_path.replace(self.chunk_manifest_path)

    def _init_or_load_chunk_manifest(self, n_rows: int) -> pd.DataFrame:
        self._require_chunking_paths()
        if self.chunk_manifest_path.exists():
            chunk_df = pd.read_parquet(self.chunk_manifest_path)
            status_counts = chunk_df["status"].value_counts(dropna=False).to_dict()
            logging.info(
                "Loaded chunk manifest from %s with %s chunks. Status counts: %s",
                self.chunk_manifest_path,
                len(chunk_df),
                status_counts,
            )
            return chunk_df

        n_chunks = math.ceil(n_rows / self.chunk_size)
        chunk_ids = np.arange(n_chunks, dtype=int)
        start_rows = chunk_ids * self.chunk_size
        end_rows = np.minimum(start_rows + self.chunk_size, n_rows)

        chunk_df = pd.DataFrame(
            {
                "chunk_id": chunk_ids,
                "start_row": start_rows,
                "end_row": end_rows,
                "status": "pending",
                "attempts": 0,
                "started_at": pd.NA,
                "completed_at": pd.NA,
                "error_message": pd.NA,
            }
        )
        self._save_chunk_manifest(chunk_df)
        logging.info(
            "Created chunk manifest at %s with %s chunks (chunk_size=%s, n_rows=%s).",
            self.chunk_manifest_path,
            n_chunks,
            self.chunk_size,
            n_rows,
        )
        return chunk_df

    @abstractmethod
    def run(self) -> Any:
        pass
