from typing import Dict, Type, Any
from .base import BaseModel


class ModelRegistry:
    """Registry for ML models"""

    def __init__(self):
        """Initialize empty registry"""
        self._models: Dict[str, Type[BaseModel]] = {}

    def register(self, name: str, model_class: Type[BaseModel]):
        """
        Register a model class.

        Args:
            name: Unique name for the model
            model_class: Model class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")
        self._models[name] = model_class

    def create(self, name: str, config: Dict[str, Any] = None) -> BaseModel:
        """
        Create model instance by name.

        Args:
            name: Name of registered model
            config: Configuration to pass to model constructor

        Returns:
            Model instance

        Raises:
            ValueError: If name is not registered
        """
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        return self._models[name](config)

    def list_models(self) -> list:
        """
        List all registered model names.

        Returns:
            List of registered names
        """
        return list(self._models.keys())


registry = ModelRegistry()

from .xgboost_model import XGBoostModel
from .random_forest_model import RandomForestModel

registry.register('xgboost', XGBoostModel)
registry.register('random_forest', RandomForestModel)
