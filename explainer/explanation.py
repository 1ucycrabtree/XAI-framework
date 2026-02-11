from dataclasses import dataclass
import numpy as np


@dataclass
class Explanation:
    instance_id: int | str
    values: np.ndarray
    base_value: float
    prediction: float | None = None
