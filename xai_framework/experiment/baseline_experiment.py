import logging
import threading
from typing import Any, Callable

import numpy as np
import pandas as pd

from experiment.base_experiment import BaseExperiment
from experiment.experiment_result import ExperimentResult
from experiment.registry import EXPERIMENTS
from experiment.sample_group_mixin import SampleGroupMixin
from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult
from metric.base_metric import BaseGlobalMetric, BaseLocalMetric
from utils.json_utils import read_json, write_json_atomic


@EXPERIMENTS.register_module("DissertationExperiment")
class DissertationExperiment(SampleGroupMixin, BaseExperiment):
    def __init__(
        self,
        cfg,
        dataset,
        model,
        explainer_method_name: str,
        perturbation_name: str,
        metrics,
        sample_size,
        random_seed,
        config_hash: str,
        explainer_factory: Callable[[int], Any],
        perturbation_factory: Callable[[int], Any],
    ):
        super().__init__(cfg=cfg, name="DissertationExperiment")
        self.dataset = dataset
        self.model = model
        self.explainer_method_name = explainer_method_name
        self.perturbation_name = perturbation_name
        self.metrics = metrics or []
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.config_hash = config_hash
        self.explainer_factory = explainer_factory
        self.perturbation_factory = perturbation_factory
        self._single_explainer: Any | None = None
        self._single_perturbation: Any | None = None
        self._thread_explainers: dict[int, Any] = {}
        self._thread_explainer_lock = threading.Lock()
        self._stop_event = threading.Event()

        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError(
                f"Experiment random_seed must be a non-negative integer. Got: {self.random_seed!r}."  # noqa: E501
            )

        stage_id = (
            f"{self.explainer_method_name}_{self.perturbation_name}".lower().replace(  # noqa: E501
                " ", "_"
            )
        )
        self.initialise_chunking_paths(stage_id)
        self.result_path = self.results_dir / f"{stage_id}_result.json"

    def _request_stop(self, reason: str) -> None:
        if not self._stop_event.is_set():
            logging.warning("Stop requested: %s", reason)
            self._stop_event.set()

    def _stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _attach_stop_event(self, explainer: Any, perturbation: Any) -> None:
        setattr(explainer, "stop_event", self._stop_event)
        setattr(perturbation, "stop_event", self._stop_event)

    def _build_chunk_components(self, chunk_id: int):
        if self.cfg.max_workers == 1:
            if self._single_explainer is None:
                self._single_explainer = self.explainer_factory(chunk_id)
            if self._single_perturbation is None:
                self._single_perturbation = self.perturbation_factory(chunk_id)
            self._attach_stop_event(self._single_explainer, self._single_perturbation)
            return self._single_explainer, self._single_perturbation

        thread_id = threading.get_ident()
        with self._thread_explainer_lock:
            explainer = self._thread_explainers.get(thread_id)
            if explainer is None:
                explainer = self.explainer_factory(chunk_id)
                self._thread_explainers[thread_id] = explainer
                logging.info(
                    "Initialised worker-local explainer for thread=%s", thread_id
                )

        perturbation = self.perturbation_factory(chunk_id)
        self._attach_stop_event(explainer, perturbation)
        return explainer, perturbation

    def _sample(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.sample(n=min(self.sample_size, len(X)), random_state=self.random_seed)

    def _save_or_load_sample(self, tp_data: pd.DataFrame) -> pd.DataFrame:
        if self.sampled_instances_path.exists():
            sampled_idx = pd.read_parquet(self.sampled_instances_path)["instance_id"]
            return tp_data.loc[sampled_idx.tolist()]

        sampled_data = self._sample(tp_data)
        sampled_df = pd.DataFrame(
            {
                "instance_id": sampled_data.index.to_list(),
                "group": str(self.cfg.sample_group).upper(),
            }
        )
        sampled_df.to_parquet(self.sampled_instances_path, index=False)
        return sampled_data

    def _serialize_explanation(self, explanation: Explanation) -> dict[str, Any]:
        return {
            "instance_id": explanation.instance_id,
            "values": np.ravel(explanation.values).tolist(),
            "base_value": float(explanation.base_value),
            "prediction": (
                None
                if explanation.prediction is None
                else float(np.ravel(explanation.prediction)[0])
            ),
        }

    def _write_chunk_output(
        self,
        chunk_id: int,
        baseline_data: pd.DataFrame,
        perturbed_data: pd.DataFrame,
        baseline_explanations: ExplanationResult,
        perturbed_explanations: ExplanationResult,
    ) -> None:
        if len(perturbed_data) != len(perturbed_explanations.instances):
            raise ValueError(
                f"Chunk {chunk_id}: perturbed_data and perturbed_explanations length mismatch."  # noqa: E501
            )

        baseline_rows = baseline_data.copy()
        baseline_rows["instance_id"] = baseline_rows.index
        baseline_records = baseline_rows.to_dict(orient="records")

        perturbed_records = []
        for row_idx, perturbed_row in enumerate(
            perturbed_data.to_dict(orient="records")
        ):
            perturbed_records.append(
                {
                    "instance_id": perturbed_explanations.instances[
                        row_idx
                    ].instance_id,
                    **perturbed_row,
                }
            )

        payload = {
            "chunk_id": chunk_id,
            "feature_names": perturbed_explanations.feature_names,
            "baseline_rows": baseline_records,
            "perturbed_rows": perturbed_records,
            "perturbation_instance_logs": (perturbed_explanations.metadata or {}).get(
                "perturbation_instance_logs", []
            ),
            "baseline_explanations": [
                self._serialize_explanation(exp)
                for exp in baseline_explanations.instances
            ],
            "perturbed_explanations": [
                self._serialize_explanation(exp)
                for exp in perturbed_explanations.instances
            ],
        }
        write_json_atomic(
            self.chunks_dir / f"chunk_{chunk_id}.json",
            payload,
            indent=2,
            default=self._json_default,
        )

    def _compute_chunk(
        self,
        chunk_id: int,
        start_row: int,
        end_row: int,
        sampled_data: pd.DataFrame,
    ) -> tuple[
        int,
        pd.DataFrame,
        pd.DataFrame,
        ExplanationResult,
        ExplanationResult,
    ]:
        if self._stop_event.is_set():
            raise InterruptedError("Chunk processing interrupted before start.")
        explainer, perturbation_strategy = self._build_chunk_components(chunk_id)
        chunk_data = sampled_data.iloc[start_row:end_row]

        if explainer is None or perturbation_strategy is None:
            raise ValueError(
                f"Failed to build explainer or perturbation for chunk {chunk_id}."
            )

        baseline_explanations = explainer.explain(chunk_data)

        perturbed_data = perturbation_strategy.perturb(
            chunk_data, explanation_result=baseline_explanations
        )
        if list(perturbed_data.columns) != list(chunk_data.columns):
            raise ValueError(
                "Perturbation strategy returned malformed DataFrame for chunk "
                f"{chunk_id}."
            )

        perturbed_explanations = explainer.explain(perturbed_data)
        perturbation_instance_logs = []
        if hasattr(perturbation_strategy, "get_last_instance_logs"):
            perturbation_instance_logs = (
                perturbation_strategy.get_last_instance_logs() or []
            )
        if perturbed_explanations.metadata is None:
            perturbed_explanations.metadata = {}
        perturbed_explanations.metadata["perturbation_instance_logs"] = (
            perturbation_instance_logs
        )
        return (
            chunk_id,
            chunk_data,
            perturbed_data,
            baseline_explanations,
            perturbed_explanations,
        )

    def _load_chunk_output(self, chunk_id: int) -> dict[str, Any]:
        return read_json(self.chunks_dir / f"chunk_{chunk_id}.json")

    def _collect_perturbation_logs(self, chunk_df: pd.DataFrame) -> list[dict]:
        logs: list[dict] = []
        for _, row in chunk_df.sort_values("chunk_id").iterrows():
            payload = self._load_chunk_output(int(row["chunk_id"]))
            logs.extend(payload.get("perturbation_instance_logs", []))
        return logs

    def _rebuild_explanation_results(
        self, chunk_df: pd.DataFrame
    ) -> tuple[ExplanationResult, ExplanationResult, pd.DataFrame, pd.DataFrame]:
        baseline_instances: list[Explanation] = []
        perturbed_instances: list[Explanation] = []
        sampled_rows: list[dict[str, Any]] = []
        perturbed_rows: list[dict[str, Any]] = []

        feature_names: list[str] | None = None
        explainer_name = (
            self._single_explainer.__class__.__name__
            if self._single_explainer is not None
            else self.explainer_method_name
        )

        for _, row in chunk_df.sort_values("chunk_id").iterrows():
            payload = self._load_chunk_output(int(row["chunk_id"]))
            feature_names = payload["feature_names"]

            for baseline_payload in payload["baseline_explanations"]:
                baseline_instances.append(
                    Explanation(
                        instance_id=baseline_payload["instance_id"],
                        values=np.array(baseline_payload["values"], dtype=float),
                        base_value=float(baseline_payload["base_value"]),
                        prediction=baseline_payload["prediction"],
                    )
                )

            sampled_rows.extend(payload["baseline_rows"])

            for pert_row, pert_payload in zip(
                payload["perturbed_rows"],
                payload["perturbed_explanations"],
                strict=True,
            ):
                perturbed_rows.append(pert_row)
                perturbed_instances.append(
                    Explanation(
                        instance_id=pert_payload["instance_id"],
                        values=np.array(pert_payload["values"], dtype=float),
                        base_value=float(pert_payload["base_value"]),
                        prediction=pert_payload["prediction"],
                    )
                )

        if feature_names is None:
            raise ValueError("No chunk outputs found for rebuilding explanations.")

        sampled_df = pd.DataFrame(sampled_rows).set_index("instance_id")
        perturbed_df = pd.DataFrame(perturbed_rows).set_index("instance_id")

        baseline_result = ExplanationResult(
            explainer_name=explainer_name,
            instances=baseline_instances,
            base_value=float(np.mean([exp.base_value for exp in baseline_instances])),
            feature_names=feature_names,
            instance_ids=[exp.instance_id for exp in baseline_instances],
        )
        perturbed_result = ExplanationResult(
            explainer_name=explainer_name,
            instances=perturbed_instances,
            base_value=float(np.mean([exp.base_value for exp in perturbed_instances])),
            feature_names=feature_names,
            instance_ids=[exp.instance_id for exp in perturbed_instances],
        )

        return baseline_result, perturbed_result, sampled_df, perturbed_df

    def _evaluate_local_metrics(
        self,
        result: ExperimentResult,
        sampled_data: pd.DataFrame,
        perturbed_data: pd.DataFrame,
        baseline_explanations: ExplanationResult,
        perturbed_explanations: ExplanationResult,
    ) -> None:
        baseline_by_id = {
            exp.instance_id: exp for exp in baseline_explanations.instances
        }
        if len(baseline_by_id) != len(baseline_explanations.instances):
            raise ValueError(
                "Baseline explanations contain duplicate instance_id values. "
                "Expected one baseline explanation per original instance."
            )

        baseline_input_by_id = {idx: row for idx, row in sampled_data.iterrows()}
        total_perturbed = len(perturbed_explanations.instances)
        prediction_changed = 0
        valid_row_positions: list[int] = []

        for row_pos, perturbed_exp in enumerate(perturbed_explanations.instances):
            instance_id = perturbed_exp.instance_id
            baseline_exp = baseline_by_id.get(instance_id)
            if baseline_exp is None:
                raise ValueError(
                    f"Missing baseline explanation for perturbed instance_id '{instance_id}'."  # noqa: E501
                )
            if instance_id not in baseline_input_by_id:
                raise ValueError(
                    f"Missing baseline input row for perturbed instance_id '{instance_id}'."  # noqa: E501
                )
            if baseline_exp.prediction is None or perturbed_exp.prediction is None:
                raise ValueError(
                    "Missing prediction in explanations. Local stability "
                    "evaluation requires baseline and perturbed predictions."
                )
            if int(np.ravel(baseline_exp.prediction)[0]) != int(
                np.ravel(perturbed_exp.prediction)[0]
            ):
                prediction_changed += 1
                continue
            valid_row_positions.append(row_pos)

        prediction_preserving = len(valid_row_positions)

        for metric in self.metrics:
            if not isinstance(metric, BaseLocalMetric):
                continue

            metric_scores = []
            for row_pos in valid_row_positions:
                perturbed_exp = perturbed_explanations.instances[row_pos]
                instance_id = perturbed_exp.instance_id
                baseline_exp = baseline_by_id.get(instance_id)
                if baseline_exp is None:
                    raise ValueError(
                        f"Missing baseline explanation for perturbed instance_id '{instance_id}'."  # noqa: E501
                    )
                if instance_id not in baseline_input_by_id:
                    raise ValueError(
                        f"Missing baseline input row for perturbed instance_id '{instance_id}'."  # noqa: E501
                    )

                score = metric.evaluate(
                    baseline_explanation=baseline_exp,
                    perturbed_explanation=perturbed_exp,
                    baseline_input=baseline_input_by_id[instance_id],
                    perturbed_input=perturbed_data.iloc[row_pos],
                )

                score = float(score)
                metric_scores.append(score)
                result.add_metric(
                    f"{metric.name}_with_ids",
                    {"instance_id": instance_id, metric.name: score},
                )

            if not metric_scores:
                logging.warning(f"No scores computed for metric '{metric.name}'.")
                continue

            result.add_metric(f"{metric.name}_mean", float(np.nanmean(metric_scores)))
            result.add_metric(f"{metric.name}_std", float(np.nanstd(metric_scores)))
            result.add_metric(f"{metric.name}_min", float(np.nanmin(metric_scores)))
            result.add_metric(f"{metric.name}_max", float(np.nanmax(metric_scores)))
            result.add_metric(f"{metric.name}_n_instances", len(metric_scores))

        if total_perturbed > 0:
            logging.info(
                "Local-metric population filter: kept=%s, dropped_prediction_changed=%s, total=%s",  # noqa: E501
                prediction_preserving,
                prediction_changed,
                total_perturbed,
            )
            result.add_metric(
                "LocalMetricFilter_prediction_preserving_n", prediction_preserving
            )
            result.add_metric(
                "LocalMetricFilter_prediction_changed_n", prediction_changed
            )
            result.add_metric("LocalMetricFilter_total_n", total_perturbed)

    def _evaluate_global_metrics(
        self,
        result: ExperimentResult,
        baseline_explanations: ExplanationResult,
        perturbed_explanations: ExplanationResult,
    ) -> None:
        for metric in self.metrics:
            if not isinstance(metric, BaseGlobalMetric):
                continue

            baseline_score = float(metric.evaluate(baseline_explanations))
            perturbed_score = float(metric.evaluate(perturbed_explanations))

            result.add_metric(f"{metric.name}_baseline", baseline_score)
            result.add_metric(f"{metric.name}_perturbed", perturbed_score)

    def run(self) -> ExperimentResult:
        manifest = self._load_or_create_run_manifest(
            random_seed=self.random_seed, config_hash=self.config_hash
        )
        logging.info("Run directory: %s", self.run_root)
        manifest["status"] = "running"
        manifest["updated_at"] = self._utcnow()
        write_json_atomic(
            self.run_manifest_path, manifest, indent=2, default=self._json_default
        )

        try:
            X = self.dataset.X_model
            y = self.dataset.y
            preds = self.model.predict(X)
            selected_data = self._get_masked_data_by_group(
                X=X, y=y, preds=preds, sample_group=self.cfg.sample_group
            )
            group = str(self.cfg.sample_group).upper()
            if selected_data.empty:
                raise ValueError(f"No samples found for sample_group='{group}'.")
            logging.info(f"Found {len(selected_data)} {group} samples.")
            logging.info(f"Sampling {self.sample_size} for the experiment.")

            sampled_data = self._save_or_load_sample(selected_data)
            chunk_df = self._init_or_load_chunk_manifest(len(sampled_data))

            self._process_chunks(sampled_data, chunk_df)
            chunk_df = pd.read_parquet(self.chunk_manifest_path)
            failed_chunks = chunk_df[chunk_df["status"] != "done"]
            if not failed_chunks.empty:
                manifest["status"] = "failed"
                manifest["updated_at"] = self._utcnow()
                write_json_atomic(
                    self.run_manifest_path,
                    manifest,
                    indent=2,
                    default=self._json_default,
                )
                raise RuntimeError(
                    "Experiment did not complete: some chunks are not in 'done' state."
                )

            (
                baseline_explanations,
                perturbed_explanations,
                sampled_data_rebuilt,
                perturbed_data_rebuilt,
            ) = self._rebuild_explanation_results(chunk_df)

            result = ExperimentResult(experiment_name=self.name)
            logging.info("Computing configured local metrics.")
            self._evaluate_local_metrics(
                result=result,
                sampled_data=sampled_data_rebuilt,
                perturbed_data=perturbed_data_rebuilt,
                baseline_explanations=baseline_explanations,
                perturbed_explanations=perturbed_explanations,
            )

            logging.info("Computing configured global metrics.")
            self._evaluate_global_metrics(
                result=result,
                baseline_explanations=baseline_explanations,
                perturbed_explanations=perturbed_explanations,
            )

            perturbation_logs = self._collect_perturbation_logs(chunk_df)
            for log_row in perturbation_logs:
                result.add_metric("PerturbationLog_with_ids", log_row)
            if perturbation_logs:
                realised = [
                    int(row.get("realised_k", 0))
                    for row in perturbation_logs
                    if "realised_k" in row
                ]
                eligible = [
                    int(row.get("eligible_perturbable", 0))
                    for row in perturbation_logs
                    if "eligible_perturbable" in row
                ]
                if realised:
                    result.add_metric(
                        "PerturbationLog_realised_k_mean",
                        float(np.mean(realised)),
                    )
                if eligible:
                    result.add_metric(
                        "PerturbationLog_eligible_perturbable_mean",
                        float(np.mean(eligible)),
                    )

            result.save(str(self.result_path))
            manifest["status"] = "completed"
            manifest["updated_at"] = self._utcnow()
            manifest["result_path"] = str(self.result_path)
            write_json_atomic(
                self.run_manifest_path, manifest, indent=2, default=self._json_default
            )
            return result
        except (KeyboardInterrupt, InterruptedError):
            self._request_stop("Run interrupted by user.")
            if self.chunk_manifest_path.exists():
                chunk_df = pd.read_parquet(self.chunk_manifest_path)
                running_mask = chunk_df["status"] == "running"
                if running_mask.any():
                    chunk_df.loc[running_mask, "status"] = "pending"
                    chunk_df.loc[running_mask, "error_message"] = "interrupted"
                    self._save_chunk_manifest(chunk_df)
            manifest["status"] = "interrupted"
            manifest["updated_at"] = self._utcnow()
            write_json_atomic(
                self.run_manifest_path, manifest, indent=2, default=self._json_default
            )
            logging.warning("Run interrupted and checkpoint state saved.")
            raise KeyboardInterrupt
