import numpy as np
from scipy.stats import spearmanr

def compute_spearman_rank_correlation(
    original_values: np.ndarray, perturbed_values: np.ndarray
) -> float:
    corr, _ = spearmanr(original_values, perturbed_values)
    return corr