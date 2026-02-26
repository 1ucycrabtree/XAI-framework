#  XAI Robustness Framework

A model-agnostic framework for evaluating the robustness and explainability of machine learning models via perturbation experiments.

---

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
│   └── default.yaml          # Experiment variables (model, explanation method, perturbation rules)
├── data/                    # Mount your dataset and model here
├── results/                 # Outputs written here
├── src/
│   ├── dataset/             # Data loading (extensible)
│   ├── model/               # Model wrappers (extensible)
│   ├── explainer/               # Explainer wrappers (extensible)
│   ├── metrics/               # Metrics wrappers (extensible)
│   ├── config_loader.py          # Maps yaml to objects
│   └── main.py
├── Dockerfile
├── docker-compose.yml
└── environment.yml
```

---

## Configuration

Modify or create new `config/default.yaml`. This defines the experiment input types and perturbation rules. 

```yaml
dataset:
  file_path: "data/test.parquet"  # path to your test dataset (must match a registered file type (see below))
  target_label: my_target      # column name of the target/label
  drop_columns:                # columns to exclude before passing to model
    - id_column
    - datetime_column
  metadata:
    name: "My Dataset"

model:
  architecture: CatBoost       # must match a registered model (see below)
  file_path: "data/models/my_model.cbm"
  metadata:
    name: "My Model"

experiment:
  n_perturbations: 10
  perturbation_magnitude: 0.1

kernel_shap:
  background_samples: 100
  random_seed: 42
```


### Supported Dataset Formats

| Format  | `file_path` extension |
|---------|-----------------------|
| Parquet | `.parquet`            |
| CSV     | `.csv`                |

The correct loader is selected automatically from the file extension.

### Supported Model Architectures

| Architecture | `architecture` value |
|--------------|----------------------|
| CatBoost     | `CatBoost`           |

---

## Extending the Framework

The framework uses a **decorator-based registry pattern** (inspired by [MMDetection](https://mmdetection.com)) so that new formats /architectures / metrics / perturbation types  are automatically discovered — no changes to core framework code required.

### Adding a New Dataset Format

1. Create `src/dataset/myformat_loader.py`
2. Implement `_load_file()` and register with `@register_loader`:

```python
from pathlib import Path
import pandas as pd
from dataset.data_loader import DataLoader, register_loader

@register_loader(".myformat")
class MyFormatLoader(DataLoader):
    SUPPORTED_SUFFIX = ".myformat"

    def _load_file(self, path: Path) -> pd.DataFrame:
        return my_library.read(path)
```

3. That's it. The framework auto-imports all `*_loader.py` files in `src/dataset/` at startup.
4. Update your `config.yaml` to point at a file with your new extension.

---

## Docker Volumes

The following directories are bind-mounted into the container — place your files here on the host:

| Host path   | Container path | Purpose              |
|-------------|----------------|----------------------|
| `./data/`   | `/app/data`    | Dataset and models   |
| `./config/` | `/app/config`  | Config YAML          |
| `./results/`| `/app/results` | Experiment outputs   |

---

## Environment
Dependencies are managed via `environment.yml` and installed into a conda env called `xai_env`.

To rebuild after changing `environment.yml`:
```bash
docker-compose --profile cpu down
docker rmi xai-robustness:cpu
docker-compose --profile cpu up -d --build