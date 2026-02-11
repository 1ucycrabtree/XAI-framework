import numpy as np
import pandas as pd
from perturbation.perturbation import BasePerturbation


class GaussianNoisePerturbation(BasePerturbation):
    def __init__(self, noise_std: float = 1.0):
        self.noise_std = noise_std

    def perturb(self, X: pd.DataFrame) -> pd.DataFrame:
        X_perturbed = X.copy()
        
        numeric_cols = X_perturbed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in the dataset to perturb.")

        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=X_perturbed[numeric_cols].shape)
        X_perturbed[numeric_cols] = X_perturbed[numeric_cols] + noise

        return X_perturbed
