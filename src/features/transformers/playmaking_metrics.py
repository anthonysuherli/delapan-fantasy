import pandas as pd
import numpy as np
from ..base import FeatureTransformer


class PlaymakingMetricsTransformer(FeatureTransformer):
    """Calculate playmaking and ball security metrics"""

    def __init__(self):
        """Initialize playmaking metrics transformer."""
        super().__init__('playmaking_metrics')

    def fit(self, data: pd.DataFrame) -> 'PlaymakingMetricsTransformer':
        """
        Fit transformer on training data.

        Args:
            data: Training data with player game logs

        Returns:
            Self for method chaining
        """
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to add playmaking metrics.

        Calculates:
        - AST_TO_ratio: ast / TOV (assists per turnover)

        Args:
            data: Data to transform

        Returns:
            DataFrame with playmaking features added

        Raises:
            ValueError: If transformer has not been fitted
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before transform")

        result = data.copy()

        # Assist to Turnover Ratio
        # Higher is better - shows ball security and playmaking
        if all(col in result.columns for col in ['ast', 'TOV']):
            result['AST_TO_ratio'] = np.where(
                result['TOV'] > 0,
                result['ast'] / result['TOV'],
                result['ast']  # If no turnovers, use assist count
            )

        return result
