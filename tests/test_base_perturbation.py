import numpy as np
import pandas as pd
import pytest

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


def test_ood_detection_and_clipping():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="LocalGaussianNoise",
        n_perturbations=1,
        random_seed=123,
        params={"lambda": 0.0},
    )
    perturb = LocalGaussianNoisePerturbation(cfg, training_data=train)

    row_high = pd.DataFrame(
        {"x": [200.0], "y": [0.0], "int_f": [0.0], "const": [1.0], "cat": ["A"], "imm": [7.0]}
    )
    assert not perturb._is_in_distribution(row_high)
    clipped = perturb._clip_to_distribution(row_high)
    assert clipped["x"].iloc[0] == pytest.approx(95.0, rel=1e-6)

    row_low = pd.DataFrame(
        {"x": [-50.0], "y": [0.0], "int_f": [0.0], "const": [1.0], "cat": ["A"], "imm": [7.0]}
    )
    clipped_low = perturb._clip_to_distribution(row_low)
    assert clipped_low["x"].iloc[0] == pytest.approx(5.0, rel=1e-6)


def test_integer_rounding_and_non_negative_clamp_helpers():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="LocalGaussianNoise",
        n_perturbations=1,
        random_seed=123,
        params={"lambda": 0.0},
    )
    perturb = LocalGaussianNoisePerturbation(
        cfg,
        training_data=train,
        integer_features=["int_f"],
        non_negative_features=["int_f"],
    )

    df = pd.DataFrame({"int_f": [1.7, -1.2]})
    rounded = perturb._round_integer_features_for(df)
    assert rounded["int_f"].tolist() == [2.0, -1.0]

    clamped = perturb._enforce_non_negative_for(df)
    assert clamped["int_f"].tolist() == [1.7, 0.0]


def test_non_negative_prefix_clamp():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="LocalGaussianNoise",
        n_perturbations=1,
        random_seed=123,
        params={"lambda": 0.0},
    )
    perturb = LocalGaussianNoisePerturbation(
        cfg,
        training_data=train,
        non_negative_prefixes=["V", "C_"],
    )

    df = pd.DataFrame({"V1": [-1.0], "V_2": [-2.0], "C_1": [-3.0], "CA": [-4.0]})
    clamped = perturb._enforce_non_negative_for(df)
    assert clamped["V1"].iloc[0] == 0.0
    assert clamped["V_2"].iloc[0] == 0.0
    assert clamped["C_1"].iloc[0] == 0.0
    assert clamped["CA"].iloc[0] == -4.0
