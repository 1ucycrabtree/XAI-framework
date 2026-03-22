import numpy as np
import pandas as pd
import pytest

from load_config import PerturbationConfig
from perturbation.direction_drift_perturbation import DirectionalDriftPerturbation


def _make_training_data():
    x = np.arange(0, 101, dtype=float)
    return pd.DataFrame(
        {
            "x": x,
            "y": x * 2,
            "int_f": x,
            "const": 1.0,
            "cat": ["A" if i % 2 == 0 else "B" for i in range(len(x))],
            "imm": 7.0,
        }
    )


def test_directional_drift_moves_only_target_features():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="DirectionalDrift",
        n_perturbations=1,
        random_seed=123,
        params={"drift_factor": 1.0, "target_features": ["x"]},
    )
    perturb = DirectionalDriftPerturbation(cfg, training_data=train)

    X = pd.DataFrame({"x": [80.0, 20.0], "y": [5.0, 6.0]})
    perturbed = perturb._perturb_instance(X)

    assert perturbed["y"].tolist() == [5.0, 6.0]
    assert perturbed["x"].tolist() == [5.0, 95.0]


def test_directional_drift_invalid_params():
    train = _make_training_data()
    with pytest.raises(ValueError):
        DirectionalDriftPerturbation(
            PerturbationConfig(
                name="DirectionalDrift",
                n_perturbations=1,
                random_seed=123,
                params={"drift_factor": 0.0, "target_features": ["x"]},
            ),
            training_data=train,
        )
    with pytest.raises(ValueError):
        DirectionalDriftPerturbation(
            PerturbationConfig(
                name="DirectionalDrift",
                n_perturbations=1,
                random_seed=123,
                params={"drift_factor": 0.5, "target_features": ["missing"]},
            ),
            training_data=train,
        )
