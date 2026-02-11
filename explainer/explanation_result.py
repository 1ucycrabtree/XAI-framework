from dataclasses import dataclass
from typing import List, Optional
from explainer.explanation import Explanation

@dataclass
class ExplanationResult:
    explainer_name: str
    instances: List[Explanation]
    base_value: float
    feature_names: List[str]
    instance_ids: List[int | str]
    metadata: Optional[dict] = None

    
    