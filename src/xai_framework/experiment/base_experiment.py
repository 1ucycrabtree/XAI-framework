import logging
import math
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_config import ExperimentConfig
from utils.json_utils import read_json, write_json_atomic


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
            manifest = read_json(self.run_manifest_path)
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
        write_json_atomic(
            self.run_manifest_path, manifest, indent=2, default=self._json_default
        )
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

    def _resolve_workers(self, n_pending_chunks: int) -> int:
        workers = max(1, int(self.cfg.max_workers))
        return min(workers, max(1, n_pending_chunks))

    def _stop_requested(self) -> bool:
        return False

    def _request_stop(self, reason: str) -> None:
        logging.warning("Stop requested: %s", reason)

    def _mark_chunk_running(self, chunk_df: pd.DataFrame, row_idx: int) -> None:
        chunk_df.at[row_idx, "status"] = "running"
        attempts_raw = chunk_df.at[row_idx, "attempts"]
        attempts_num = pd.to_numeric(pd.Series([attempts_raw]), errors="coerce").iloc[0]
        current_attempts = 0 if pd.isna(attempts_num) else int(attempts_num)
        chunk_df.at[row_idx, "attempts"] = current_attempts + 1
        chunk_df.at[row_idx, "started_at"] = self._utcnow()
        chunk_df.at[row_idx, "error_message"] = pd.NA
        self._save_chunk_manifest(chunk_df)

    def _mark_chunk_done(self, chunk_df: pd.DataFrame, row_idx: int) -> None:
        chunk_df.at[row_idx, "status"] = "done"
        chunk_df.at[row_idx, "completed_at"] = self._utcnow()
        self._save_chunk_manifest(chunk_df)

    def _mark_chunk_failed(
        self, chunk_df: pd.DataFrame, row_idx: int, err: Exception
    ) -> None:
        chunk_df.at[row_idx, "status"] = "failed"
        chunk_df.at[row_idx, "error_message"] = str(err)
        self._save_chunk_manifest(chunk_df)

    def _process_chunks_sequential(
        self,
        sampled_data: pd.DataFrame,
        chunk_df: pd.DataFrame,
        pending_row_idxs: list[int],
    ) -> pd.DataFrame:
        logging.info(
            "Processing %s pending chunks sequentially.", len(pending_row_idxs)
        )
        for row_idx in pending_row_idxs:
            row = chunk_df.iloc[row_idx]
            chunk_id = int(row["chunk_id"])
            start_row = int(row["start_row"])
            end_row = int(row["end_row"])

            attempts_raw = chunk_df.at[row_idx, "attempts"]
            attempts_num = pd.to_numeric(
                pd.Series([attempts_raw]), errors="coerce"
            ).iloc[0]
            logging.info(
                "Starting chunk %s rows[%s:%s) attempt=%s",
                chunk_id,
                start_row,
                end_row,
                int(attempts_num) + 1,
            )
            self._mark_chunk_running(chunk_df, row_idx)

            try:
                if self._stop_requested():
                    raise InterruptedError("Stop requested before chunk execution.")
                chunk_started = time.perf_counter()
                (
                    _,
                    chunk_data,
                    perturbed_data,
                    baseline_explanations,
                    perturbed_explanations,
                ) = self._compute_chunk(
                    chunk_id=chunk_id,
                    start_row=start_row,
                    end_row=end_row,
                    sampled_data=sampled_data,
                )

                self._write_chunk_output(
                    chunk_id=chunk_id,
                    baseline_data=chunk_data,
                    perturbed_data=perturbed_data,
                    baseline_explanations=baseline_explanations,
                    perturbed_explanations=perturbed_explanations,
                )
                self._mark_chunk_done(chunk_df, row_idx)
                chunk_elapsed = time.perf_counter() - chunk_started
                logging.info(
                    "Completed chunk %s in %.2fs (baseline=%s, perturbed=%s).",
                    chunk_id,
                    chunk_elapsed,
                    len(chunk_data),
                    len(perturbed_data),
                )

            except Exception as e:
                self._mark_chunk_failed(chunk_df, row_idx, e)
                logging.exception("Chunk %s failed: %s", chunk_id, e)
                raise

        return chunk_df

    def _process_chunks_parallel(
        self,
        sampled_data: pd.DataFrame,
        chunk_df: pd.DataFrame,
        pending_row_idxs: list[int],
    ) -> pd.DataFrame:
        workers = self._resolve_workers(len(pending_row_idxs))
        if workers == 1:
            return self._process_chunks_sequential(
                sampled_data, chunk_df, pending_row_idxs
            )

        logging.info(
            "Processing %s pending chunks with %s worker threads.",
            len(pending_row_idxs),
            workers,
        )
        future_to_row_idx: dict[Future[Any], int] = {}
        future_to_started_at: dict[Future[Any], float] = {}

        pending_iter = iter(pending_row_idxs)
        executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="chunk-worker"
        )
        try:
            for _ in range(workers):
                try:
                    row_idx = next(pending_iter)
                except StopIteration:
                    break
                row = chunk_df.iloc[row_idx]
                self._mark_chunk_running(chunk_df, row_idx)
                future = executor.submit(
                    self._compute_chunk,
                    int(row["chunk_id"]),
                    int(row["start_row"]),
                    int(row["end_row"]),
                    sampled_data,
                )
                future_to_row_idx[future] = row_idx
                future_to_started_at[future] = time.perf_counter()

            while future_to_row_idx:
                done, _ = wait(future_to_row_idx.keys(), return_when=FIRST_COMPLETED)

                for finished in done:
                    row_idx = future_to_row_idx.pop(finished)
                    chunk_started = future_to_started_at.pop(
                        finished, time.perf_counter()
                    )
                    row = chunk_df.iloc[row_idx]
                    chunk_id = int(row["chunk_id"])

                    try:
                        (
                            _,
                            chunk_data,
                            perturbed_data,
                            baseline_explanations,
                            perturbed_explanations,
                        ) = finished.result()

                        self._write_chunk_output(
                            chunk_id=chunk_id,
                            baseline_data=chunk_data,
                            perturbed_data=perturbed_data,
                            baseline_explanations=baseline_explanations,
                            perturbed_explanations=perturbed_explanations,
                        )
                        self._mark_chunk_done(chunk_df, row_idx)
                        chunk_elapsed = time.perf_counter() - chunk_started
                        logging.info(
                            "Completed chunk %s in %.2fs (baseline=%s, perturbed=%s).",
                            chunk_id,
                            chunk_elapsed,
                            len(chunk_data),
                            len(perturbed_data),
                        )

                    except Exception as e:
                        self._mark_chunk_failed(chunk_df, row_idx, e)
                        logging.exception("Chunk %s failed: %s", chunk_id, e)
                        for queued in future_to_row_idx:
                            queued.cancel()
                        raise

                    if self._stop_requested():
                        raise InterruptedError(
                            "Stop requested during parallel chunk processing."
                        )

                    try:
                        next_row_idx = next(pending_iter)
                    except StopIteration:
                        continue

                    next_row = chunk_df.iloc[next_row_idx]
                    self._mark_chunk_running(chunk_df, next_row_idx)
                    next_future = executor.submit(
                        self._compute_chunk,
                        int(next_row["chunk_id"]),
                        int(next_row["start_row"]),
                        int(next_row["end_row"]),
                        sampled_data,
                    )
                    future_to_row_idx[next_future] = next_row_idx
                    future_to_started_at[next_future] = time.perf_counter()
        except KeyboardInterrupt:
            self._request_stop("KeyboardInterrupt received in chunk processor.")
            for queued in future_to_row_idx:
                queued.cancel()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return chunk_df

    def _process_chunks(
        self, sampled_data: pd.DataFrame, chunk_df: pd.DataFrame
    ) -> pd.DataFrame:
        pending_row_idxs = [
            row_idx
            for row_idx in range(len(chunk_df))
            if chunk_df.iloc[row_idx]["status"] != "done"
        ]

        if not pending_row_idxs:
            logging.info("No pending chunks found. Skipping chunk processing.")
            return chunk_df

        logging.info(
            "Chunk processing summary before execution: total=%s done=%s pending_or_failed=%s",  # noqa: E501
            len(chunk_df),
            int((chunk_df["status"] == "done").sum()),
            len(pending_row_idxs),
        )
        return self._process_chunks_parallel(sampled_data, chunk_df, pending_row_idxs)

    @abstractmethod
    def _compute_chunk(
        self,
        chunk_id: int,
        start_row: int,
        end_row: int,
        sampled_data: pd.DataFrame,
    ) -> tuple[Any, Any, Any, Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def _write_chunk_output(
        self,
        chunk_id: int,
        baseline_data: pd.DataFrame,
        perturbed_data: pd.DataFrame,
        baseline_explanations: Any,
        perturbed_explanations: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> Any:
        pass
