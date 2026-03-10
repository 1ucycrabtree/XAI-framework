from typing import Any

import numpy as np
import pandas as pd


class SampleGroupMixin:
    def _get_masked_data_by_group(
        self,
        X: pd.DataFrame,
        y: Any,
        preds: Any,
        sample_group: str,
    ) -> pd.DataFrame:
        group = str(sample_group).upper()
        mapping = {
            "TP": (1, 1),
            "TN": (0, 0),
            "FP": (1, 0),
            "FN": (0, 1),
        }
        if group not in mapping:
            raise ValueError(
                f"Unsupported sample_group '{sample_group}'. Use TP/TN/FP/FN."
            )
        predicted_val, actual_val = mapping[group]

        preds_array = np.array(preds).flatten()
        y_array = np.asarray(y)

        mask = (preds_array == predicted_val) & (y_array == actual_val)
        return X.loc[mask]
