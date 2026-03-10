from typing import Optional

from dataset.dataset import Dataset
from explainer.base_explainer import BaseExplainer
from load_config import ExplainerConfig
from model.base_model import BaseModel
from utils.registry import Registry

EXPLAINERS = Registry("explainer")


def get_explainer(
    cfg: ExplainerConfig,
    model_wrapper: BaseModel,
    train_dataset: Optional[Dataset] = None,
) -> BaseExplainer:
    explainer_cls = EXPLAINERS.get(cfg.method)
    return explainer_cls(cfg, model_wrapper, train_dataset)
