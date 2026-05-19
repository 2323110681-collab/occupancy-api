from typing import Dict
from .predictor import Predictor
import config


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Predictor] = {}
        self._load_models()

    def _load_models(self):
        for name, path in config.MODEL_PATHS.items():
            try:
                self._models[name] = Predictor(path)
            except Exception as e:
                print(f"Error loading model {name}: {e}")

    def get(self, name: str) -> Predictor:
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        return self._models[name]

    def get_all(self) -> Dict[str, Predictor]:
        return self._models

    def list_models(self) -> list:
        return list(self._models.keys())


registry = ModelRegistry()