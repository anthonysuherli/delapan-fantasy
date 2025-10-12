from typing import Dict, Type
from .base import BaseMetric


class MetricRegistry:
    """Registry for evaluation metrics"""

    def __init__(self):
        """Initialize empty registry"""
        self._metrics: Dict[str, Type[BaseMetric]] = {}

    def register(self, name: str, metric_class: Type[BaseMetric]):
        """
        Register a metric class.

        Args:
            name: Unique name for the metric
            metric_class: Metric class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in self._metrics:
            raise ValueError(f"Metric '{name}' is already registered")
        self._metrics[name] = metric_class

    def create(self, name: str, **kwargs) -> BaseMetric:
        """
        Create metric instance by name.

        Args:
            name: Name of registered metric
            **kwargs: Arguments to pass to metric constructor

        Returns:
            Metric instance

        Raises:
            ValueError: If name is not registered
        """
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return self._metrics[name](**kwargs)

    def create_suite(self, metric_names: list) -> list:
        """
        Create multiple metrics at once.

        Args:
            metric_names: List of metric names to create

        Returns:
            List of metric instances
        """
        return [self.create(name) for name in metric_names]

    def list_metrics(self) -> list:
        """
        List all registered metric names.

        Returns:
            List of registered names
        """
        return list(self._metrics.keys())


registry = MetricRegistry()

from .accuracy import MAPEMetric, RMSEMetric, MAEMetric, CorrelationMetric

try: 
    registry.register('mape', MAPEMetric)
except: 
    pass
try: 
    registry.register('rmse', RMSEMetric)
except: 
    pass
try: 
    registry.register('mae', MAEMetric)
except: 
    pass
try: 
    registry.register('correlation', CorrelationMetric)
except: 
    pass
