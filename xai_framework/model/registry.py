from load_config import ModelConfig
from model.base_model import BaseModel
from utils.registry import Registry

MODELS = Registry("model")


def get_model(cfg: ModelConfig) -> BaseModel:
    model_cls = MODELS.get(cfg.architecture)
    return model_cls(cfg)
