from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any


class BaseModel(ABC):
    """Abstract base for all ML models"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self._is_trained = False

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_inputs: bool = False,
        input_save_path: str = None
    ) -> 'BaseModel':
        """
        Train model on data.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Target variable with shape (n_samples,)
            save_inputs: Whether to save training inputs to disk
            input_save_path: Path to save training inputs

        Returns:
            Self for method chaining

        Raises:
            ValueError: If X and y have mismatched lengths
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix with shape (n_samples, n_features)

        Returns:
            Predictions with shape (n_samples,)

        Raises:
            ValueError: If model has not been trained
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serialize model to disk.

        Args:
            path: File path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> 'BaseModel':
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            Self for method chaining
        """
        pass

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained"""
        return self._is_trained
