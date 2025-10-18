from typing import List, Dict, Any, Optional
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
        self.context = {}

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

    def set_context(self, context: Dict[str, Any]) -> 'FeaturePipeline':
        """
        Set context data for transformers that require additional inputs.

        Args:
            context: Dictionary of context data (e.g., {'injuries': DataFrame})

        Returns:
            Self for method chaining
        """
        self.context = context
        return self

    def fit(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> 'FeaturePipeline':
        """
        Fit all transformers on training data.

        Args:
            data: Training data
            context: Optional context data for transformers

        Returns:
            Self for method chaining
        """
        if context:
            self.context = context

        for transformer in tqdm(self.transformers, desc="Fitting transformers", leave=False):
            transformer.fit(data)
        return self

    def transform(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Transform data through all transformers sequentially.

        Args:
            data: Data to transform
            context: Optional context data for transformers

        Returns:
            DataFrame with all features added

        Raises:
            ValueError: If any transformer has not been fitted
        """
        if context:
            self.context = context

        result = data.copy()
        for transformer in tqdm(self.transformers, desc="Applying transformers", leave=False):
            if not transformer.is_fitted:
                raise ValueError(f"Transformer '{transformer.name}' has not been fitted")

            if transformer.name == 'injury':
                injuries = self.context.get('injuries', pd.DataFrame())
                result = transformer.transform(result, injuries=injuries)
            else:
                result = transformer.transform(result)
        return result

    def fit_transform(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fit all transformers and transform data in one step.

        Args:
            data: Data to fit and transform
            context: Optional context data for transformers

        Returns:
            DataFrame with all features added
        """
        return self.fit(data, context).transform(data, context)
