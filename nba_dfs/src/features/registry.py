from typing import Dict, Type, Any
from .base import FeatureTransformer


class FeatureRegistry:
    """Registry for feature transformers"""

    def __init__(self):
        """Initialize empty registry"""
        self._transformers: Dict[str, Type[FeatureTransformer]] = {}

    def register(self, name: str, transformer_class: Type[FeatureTransformer]):
        """
        Register a transformer class.

        Args:
            name: Unique name for the transformer
            transformer_class: Transformer class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in self._transformers:
            raise ValueError(f"Transformer '{name}' is already registered")
        self._transformers[name] = transformer_class

    def create(self, name: str, **kwargs) -> FeatureTransformer:
        """
        Create transformer instance by name.

        Args:
            name: Name of registered transformer
            **kwargs: Arguments to pass to transformer constructor

        Returns:
            Transformer instance

        Raises:
            ValueError: If name is not registered
        """
        if name not in self._transformers:
            raise ValueError(f"Unknown transformer: {name}")
        return self._transformers[name](**kwargs)

    def list_transformers(self) -> list:
        """
        List all registered transformer names.

        Returns:
            List of registered names
        """
        return list(self._transformers.keys())


registry = FeatureRegistry()
