from abc import ABC, abstractmethod
import pandas as pd


class FeatureTransformer(ABC):
    """Abstract base for all feature transformers"""

    def __init__(self, name: str):
        """
        Initialize feature transformer.

        Args:
            name: Unique name for this transformer
        """
        self.name = name
        self._fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'FeatureTransformer':
        """
        Fit transformer on training data.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to add features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with added features

        Raises:
            ValueError: If transformer has not been fitted
        """
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            data: Data to fit and transform

        Returns:
            DataFrame with added features
        """
        return self.fit(data).transform(data)

    @property
    def is_fitted(self) -> bool:
        """Check if transformer has been fitted"""
        return self._fitted
