import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExperimentResult:
    experiment_name: str
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def summary(self) -> None:
        logging.info(f"Printing summary for experiment: {self.experiment_name}")
        print(f"Experiment: {self.experiment_name}")
        for metric, values in self.metrics.items():
            if len(values) > 5:
                print(f"{metric}: {values[:5]}... (total {len(values)} values)")
            else:
                print(f"{metric}: {values}")

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(
                {"experiment_name": self.experiment_name, "metrics": self.metrics},
                f,
                indent=4,
            )

    def add_metric(self, metric_name: str, value: Any) -> None:
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
