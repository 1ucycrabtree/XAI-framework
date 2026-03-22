import numpy as np
import pandas as pd
import pytest
from explainer.explanation import Explanation
from explainer.explanation_result import ExplanationResult
from load_config import PerturbationConfig
from perturbation.top_k_perturbation import TopKPerturbation


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


def test_top_k_perturbation_only_changes_top_feature():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="TopKFeatures",
        n_perturbations=1,
        random_seed=123,
        params={"k": 1, "lambda": 1.0},
    )
    perturb = TopKPerturbation(cfg, training_data=train)

    X = pd.DataFrame({"x": [10.0], "y": [20.0], "int_f": [30.0]})
    exp = Explanation(
        instance_id=0, values=np.array([5.0, 1.0, 0.5]), base_value=0.0, prediction=1
    )
    expl_result = ExplanationResult(
        explainer_name="unit",
        instances=[exp],
        base_value=0.0,
        feature_names=["x", "y", "int_f"],
        instance_ids=[0],
    )

    mad_x = perturb.feature_stats["x"]["mad"]
    expected_noise = np.random.default_rng(123).normal(scale=1.0 * mad_x)

    perturbed = perturb._perturb_instance(X, explanation_result=expl_result)

    assert perturbed["x"].iloc[0] == pytest.approx(10.0 + expected_noise, rel=1e-6)
    assert perturbed["y"].iloc[0] == 20.0
    assert perturbed["int_f"].iloc[0] == 30.0


def test_top_k_uses_eligible_perturbable_features():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="TopKFeatures",
        n_perturbations=1,
        random_seed=123,
        params={"k": 5, "lambda": 1.0},
    )
    perturb = TopKPerturbation(
        cfg,
        training_data=train,
        immutable_features=["x"],
        perturbable_numerical_features=["y"],
        perturbable_categorical_features=["cat"],
        categorical_features=["cat"],
    )

    X = pd.DataFrame({"x": [10.0], "y": [20.0], "cat": ["A"], "imm": [7.0]})
    exp = Explanation(
        instance_id=0,
        values=np.array([10.0, 9.0, 8.0, 7.0]),
        base_value=0.0,
        prediction=1,
    )
    expl_result = ExplanationResult(
        explainer_name="unit",
        instances=[exp],
        base_value=0.0,
        feature_names=["x", "imm", "y", "cat"],
        instance_ids=[0],
    )

    perturbed = perturb._perturb_instance(X, explanation_result=expl_result)
    logs = perturb.get_last_instance_logs()

    # x is immutable and imm is non-perturbable, so only y/cat are eligible.
    assert logs[-1]["eligible_perturbable"] == 2
    assert logs[-1]["realised_k"] == 2
    assert logs[-1]["perturbed_features_count"] >= 1
    assert set(logs[-1]["selected_features"]) == {"y", "cat"}
    assert perturbed["x"].iloc[0] == X["x"].iloc[0]


def test_top_k_records_no_eligible_features_without_perturbing():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="TopKFeatures",
        n_perturbations=1,
        random_seed=123,
        params={"k": 5, "lambda": 1.0},
    )
    perturb = TopKPerturbation(
        cfg,
        training_data=train,
        perturbable_numerical_features=["y"],  # x is intentionally excluded
    )

    X = pd.DataFrame({"x": [10.0], "cat": ["A"]})
    exp = Explanation(
        instance_id=0, values=np.array([5.0, 1.0]), base_value=0.0, prediction=1
    )
    expl_result = ExplanationResult(
        explainer_name="unit",
        instances=[exp],
        base_value=0.0,
        feature_names=["x", "cat"],
        instance_ids=[0],
    )

    perturbed = perturb._perturb_instance(X, explanation_result=expl_result)
    logs = perturb.get_last_instance_logs()

    assert perturbed.equals(X)
    assert logs[-1]["eligible_perturbable"] == 0
    assert logs[-1]["realised_k"] == 0
    assert logs[-1]["perturbed_features_count"] == 0
    assert logs[-1]["skipped_no_eligible_features"] is True


def test_top_k_requires_explanation_result():
    train = _make_training_data()
    cfg = PerturbationConfig(
        name="TopKFeatures",
        n_perturbations=1,
        random_seed=123,
        params={"k": 1, "lambda": 1.0},
    )
    perturb = TopKPerturbation(cfg, training_data=train)
    X = pd.DataFrame({"x": [10.0]})

    with pytest.raises(ValueError):
        perturb._perturb_instance(X)
