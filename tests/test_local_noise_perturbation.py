import numpy as np
import pandas as pd

from load_config import PerturbationConfig
from perturbation.local_noise_perturbation import LocalGaussianNoisePerturbation


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


def test_local_noise_respects_immutable_and_constant_features():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="LocalGaussianNoise",
        n_perturbations=1,
        random_seed=123,
        params={"lambda": 0.5},
    )
    perturb = LocalGaussianNoisePerturbation(
        cfg,
        training_data=train,
        immutable_features=["imm"],
        integer_features=["int_f"],
        non_negative_features=["int_f"],
    )

    X = train.iloc[[10]].copy()
    perturbed = perturb._perturb_instance(X)

    assert perturbed["imm"].iloc[0] == X["imm"].iloc[0]
    assert perturbed["const"].iloc[0] == X["const"].iloc[0]
