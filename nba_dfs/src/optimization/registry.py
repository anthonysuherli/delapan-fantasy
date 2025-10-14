from typing import Dict, Type, Any, List
from .base import BaseOptimizer, BaseConstraint


class OptimizerRegistry:
    """Registry for lineup optimizers"""

    def __init__(self):
        """Initialize empty registry"""
        self._optimizers: Dict[str, Type[BaseOptimizer]] = {}

    def register(self, name: str, optimizer_class: Type[BaseOptimizer]):
        """
        Register an optimizer class.

        Args:
            name: Unique name for the optimizer
            optimizer_class: Optimizer class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in self._optimizers:
            raise ValueError(f"Optimizer '{name}' is already registered")
        self._optimizers[name] = optimizer_class

    def create(
        self,
        name: str,
        constraints: List[BaseConstraint],
        **kwargs
    ) -> BaseOptimizer:
        """
        Create optimizer instance by name.

        Args:
            name: Name of registered optimizer
            constraints: List of constraints to enforce
            **kwargs: Additional arguments to pass to optimizer constructor

        Returns:
            Optimizer instance

        Raises:
            ValueError: If name is not registered
        """
        if name not in self._optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
        return self._optimizers[name](constraints, **kwargs)

    def list_optimizers(self) -> list:
        """
        List all registered optimizer names.

        Returns:
            List of registered names
        """
        return list(self._optimizers.keys())


registry = OptimizerRegistry()

from .optimizers.linear_program import LinearProgramOptimizer

registry.register('linear_program', LinearProgramOptimizer)
