# XAI Robustness Framework

Framework for robustness evaluation of tabular explainers under controlled perturbations.

## Quickstart

```bash
docker-compose up -d --build
docker-compose exec xai-robustness bash
conda activate xai_env
python src/main.py --config default.yaml
```

## Project Structure

```
xai-robustness/
├── config/
│   └── default.yaml                    # Default experiment variables
│   └── KernelSHAP_*_*.yaml             # KernelSHAP configs by group/perturbation
│   └── TabularLIME_*_*.yaml            # TabularLIME configs by group/perturbation
├── data/                     # Mount your dataset and model here
├── results/                  # Outputs and checkpoints written here
├── src/
│   ├── dataset/              # Data loading (extensible)
│   ├── experiment/           # Experiment (extensible)
│   ├── explainer/            # Explainer wrappers (extensible)
│   ├── metrics/              # Metrics wrappers (extensible)
│   ├── model/                # Model wrappers (extensible)
│   ├── perturbation/         # Perturbation wrappers (extensible)
│   ├── load_config.py        # Maps yaml to objects
│   └── main.py
├── Dockerfile
├── docker-compose.yml
└── environment.yml
```

---

## How It Works

`src/main.py` orchestrates one run:

1. Load config (`config/*.yaml`).
2. Load dataset/model/explainer.
3. Build metrics from registry.
4. For each perturbation strategy, build and run an experiment via the experiment registry.

The default experiment is `BaselineExperiment`, selected by:

```yaml
experiment:
  name: BaselineExperiment
```

## Registry Pattern

The project uses decorator-based registries:

- Models: `src/model/registry.py`
- Explainers: `src/explainer/registry.py`
- Perturbations: `src/perturbation/registry.py`
- Metrics: `src/metric/registry.py`
- Experiments: `src/experiment/registry.py`

To add a new component to any of the above, create a class and register it:

```python
# Example for new experiment
from experiment.registry import EXPERIMENTS

@EXPERIMENTS.register_module("MyExperiment")
class MyExperiment(...):
    ...
```

Then set `experiment.name: MyExperiment` and associated params in config.

--

### Below is a list of already implemented components:
#### Dataset Formats

| Format  | `file_path` extension |
| ------- | --------------------- |
| Parquet | `.parquet`            |
| CSV     | `.csv`                |

The correct loader is selected automatically from the file extension.

#### Model Architectures

| Architecture | `architecture` value |
| ------------ | -------------------- |
| CatBoost     | `CatBoost`           |

#### Explanation Methods

| Explainer    | `method` value |
| ------------ | -------------- |
| Kernel SHAP  | `KernelSHAP`   |
| Tabular LIME | `TabularLIME`  |
| Tree SHAP    | `TreeSHAP`     |

#### Metrics

| Metric                         | `name` value             |
| ------------------------------ | ------------------------ |
| Relative Input Stability (RIS) | `RelativeInputStability` |
| Rank Biased Overlap (RBO)      | `RankBiasedOverlap`      |
| Sign Consistency Rate (SCR)    | `SignConsistencyRate`    |
| Global Consistency             | `GlobalConsistency`      |
| Global Sufficiency             | `GlobalSufficiency`      |

#### Perturbations

| Perturbation Type    | `name` value         |
| -------------------- | -------------------- |
| Local Gaussian Noise | `LocalGaussianNoise` |
| Directional Drift    | `DirectionalDrift`   |
| Top K Features       | `TopKFeatures`       |

#### Experiments

| Experiment   | `name` value         |
| -------------------- | -------------------- |
| Baseline Experiment | `BaselineExperiment` |

---

## Deterministic + Resumable Runs

Experiment seed is required:

```yaml
experiment:
  random_seed: 42
```

Resume/checkpoint behaviour:

- Sampled IDs are frozen in `sampled_instances.parquet`.
- Chunk states are tracked in `chunk_manifest.parquet`.
- Run metadata is tracked in `run_manifest.json`.
- Completed chunks are skipped on resume.
- On interruption (`Ctrl+C`), in-flight `running` chunks are reset to `pending`, run status is set to `interrupted`, and resume can continue from chunk boundaries.

Parallel chunk execution:

- Set `experiment.max_workers` to process multiple chunks concurrently in one run.
- `max_workers: 1` keeps the original single-worker behaviour.
- For heavy explainers, start with small values and scale based on RAM/CPU headroom.

Sampling group selection:

- Set `experiment.sample_group` to choose which confusion-matrix group is sampled: `TP`, `TN`, `FP`, or `FN`.

Important:

- Resume only works for the same `run_id`.
- If you want restart-safe resume across process reruns, set a fixed `experiment.run_id`.
- If `run_id` is omitted, the default format is `<experiment_name>_run_<YYYYMMDD_HHMMSS>` in UTC.

## Output Layout

For a run id like `BaselineExperiment_run_20260309_165129`:

- Checkpoints:
  - `results/checkpoints/<run_id>/<explainer>_<perturbation>/run_manifest.json`
  - `results/checkpoints/<run_id>/<explainer>_<perturbation>/sampled_instances.parquet`
  - `results/checkpoints/<run_id>/<explainer>_<perturbation>/chunk_manifest.parquet`
  - `results/checkpoints/<run_id>/<explainer>_<perturbation>/chunks/chunk_<id>.json`
- Final metrics:
  - `results/<run_id>/<explainer>_<perturbation>_result.json`

All perturbations from one launch share the same `<run_id>`.

## Metric Result Keys

Local metrics produce:

- `<MetricName>_with_ids`
- `<MetricName>_mean`
- `<MetricName>_std`
- `<MetricName>_min`
- `<MetricName>_max`
- `<MetricName>_n_instances`

Global metrics produce:

- `GlobalConsistencyMetric_baseline`
- `GlobalConsistencyMetric_perturbed`
- `GlobalSufficiencyMetric_baseline`
- `GlobalSufficiencyMetric_perturbed`

## Minimal Config Example

```yaml
dataset:
  train_file_path: "data/processed/splits/train_dataset.parquet"
  test_file_path: "data/processed/splits/test_dataset.parquet"
  target_label: isFraud

model:
  architecture: CatBoost
  file_path: "data/models/catboost_model.cbm"

explainer:
  method: KernelSHAP
  params:
    random_seed: 42

perturbations:
  - name: TopKFeatures
    n_perturbations: 10
    params:
      k: 5
      lambda: 0.3

metrics:
  - name: RelativeInputStability
  - name: RankBiasedOverlap
  - name: SignConsistencyRate
  - name: GlobalConsistencyMetric
  - name: GlobalSufficiencyMetric

experiment:
  name: BaselineExperiment
  sample_size: 1000
  sample_group: TP
  random_seed: 42
  chunk_size: 50
  max_workers: 4
  resume: true
  checkpoint_dir: "results/checkpoints"
  results_dir: "results"
  # run_id: "run_my_reproducible_trial"
```

## Run

```bash
python src/main.py --config TabularLIME_TP_Noise.yaml
```

## License

Apache License 2.0. See `LICENSE`.
