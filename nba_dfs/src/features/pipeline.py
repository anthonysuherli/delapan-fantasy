from typing import List
import pandas as pd
from tqdm import tqdm
from .base import FeatureTransformer


class FeaturePipeline:
    """Compose multiple feature transformers"""

    def __init__(self, transformers: List[FeatureTransformer] = None):
        """
        Initialize feature pipeline.

        Args:
            transformers: Optional list of transformers to start with
        """
        self.transformers = transformers or []

    def add(self, transformer: FeatureTransformer) -> 'FeaturePipeline':
        """
        Add transformer to pipeline.

        Args:
            transformer: Transformer to add

        Returns:
            Self for method chaining
        """
        self.transformers.append(transformer)
        return self

    def fit(self, data: pd.DataFrame) -> 'FeaturePipeline':
        """
        Fit all transformers on training data.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        for transformer in tqdm(self.transformers, desc="Fitting transformers", leave=False):
            transformer.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data through all transformers sequentially.

        Args:
            data: Data to transform

        Returns:
            DataFrame with all features added

        Raises:
            ValueError: If any transformer has not been fitted
        """
        result = data.copy()
        for transformer in tqdm(self.transformers, desc="Applying transformers", leave=False):
            if not transformer.is_fitted:
                raise ValueError(f"Transformer '{transformer.name}' has not been fitted")
            result = transformer.transform(result)
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all transformers and transform data in one step.

        Args:
            data: Data to fit and transform

        Returns:
            DataFrame with all features added
        """
        return self.fit(data).transform(data)
