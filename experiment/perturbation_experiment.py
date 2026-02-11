from experiment.experiment import BaseExperiment
from experiment.experiment_result import ExperimentResult
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import logging


class PerturbationExperiment(BaseExperiment):
    def __init__(
        self,
        dataset,
        model,
        explainer,
        perturbation_strategy,
        sample_size=100,
        random_seed=42,
    ):
        super().__init__(name="Basic Perturbation Experiment")
        self.dataset = dataset
        self.model = model
        self.explainer = explainer
        self.perturbation_strategy = perturbation_strategy
        self.sample_size = sample_size
        self.random_seed = random_seed

    def _get_masked_data(self, predicted_val, actual_val) -> pd.DataFrame:
        X = self.dataset.X
        y = self.dataset.y

        preds = self.model.predict(X)
        preds = np.array(preds).flatten()

        mask = (preds == predicted_val) & (y == actual_val)
        return X.loc[mask]

    def _sample(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.sample(n=min(self.sample_size, len(X)), random_state=self.random_seed)

    def _compute_rank_corrleation(self, original_explanation, perturbed_explanation):
        correlations = []
        for orig, pert in zip(
            original_explanation.instances, perturbed_explanation.instances
        ):
            rho, _ = spearmanr(orig.values, pert.values)
            correlations.append(rho)

        return correlations

    def run(self) -> ExperimentResult:

        result = ExperimentResult(experiment_name=self.name)

        tp_data = self._get_masked_data(predicted_val=1, actual_val=1)

        if tp_data.empty:
            raise ValueError("No true positive samples found in the dataset.")
        logging.info(f"Found {len(tp_data)} true positive samples. Sampling {self.sample_size} for the experiment.")

        sampled_data = self._sample(tp_data)

        logging.info("Generating baseline explanations for the sampled data.")
        baseline_explanations = self.explainer.explain(sampled_data)
        
        logging.info("Applying perturbation strategy to the sampled data.")
        perturbed_data = self.perturbation_strategy.perturb(sampled_data)

        logging.info("Generating explanations for the perturbed data.")
        perturbed_explanations = self.explainer.explain(perturbed_data)

        logging.info("Computing rank correlations between baseline and perturbed explanations.")
        rank_correlations = self._compute_rank_corrleation(
            baseline_explanations, perturbed_explanations
        )

        logging.info("Rank correlations computed. Summarising results.")

        
        for idx, corr in enumerate(rank_correlations):
            instance_id = baseline_explanations.instance_ids[idx]
            result.add_metric("rank_correlation_with_ids", {"instance_id": instance_id, "rank_correlation": float(corr)})
        result.add_metric("mean_rank_correlation", float(np.nanmean(rank_correlations)))
        result.add_metric("std_rank_correlation", float(np.nanstd(rank_correlations)))
        result.add_metric("min_rank_correlation", float(np.nanmin(rank_correlations)))
        result.add_metric("max_rank_correlation", float(np.nanmax(rank_correlations)))
        result.add_metric("n_instances", len(rank_correlations))

        return result
