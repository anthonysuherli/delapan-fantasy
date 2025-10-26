import pandas as pd
import numpy as np
from ..base import FeatureTransformer


class ImpactMetricsTransformer(FeatureTransformer):
    """Calculate advanced impact metrics that condense box score performance"""

    def __init__(self):
        """Initialize impact metrics transformer."""
        super().__init__('impact_metrics')

    def fit(self, data: pd.DataFrame) -> 'ImpactMetricsTransformer':
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
        Transform data to add impact metrics.

        Calculates:
        - GameScore: Hollinger's Game Score metric that condenses box score into single number
          Formula: pts + (0.4 × fgm) − (0.7 × fga) − (0.4 × (fta − ftm)) +
                   (0.7 × OffReb) + (0.3 × DefReb) + stl + (0.7 × ast) +
                   (0.7 × blk) − (0.4 × PF) − TOV

        Args:
            data: Data to transform

        Returns:
            DataFrame with impact features added

        Raises:
            ValueError: If transformer has not been fitted
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before transform")

        result = data.copy()

        # Game Score
        # Hollinger's metric that estimates overall player performance
        required_cols = ['pts', 'fgm', 'fga', 'fta', 'ftm', 'OffReb', 'DefReb',
                        'stl', 'ast', 'blk', 'PF', 'TOV']

        if all(col in result.columns for col in required_cols):
            result['GameScore'] = (
                result['pts'] +
                (0.4 * result['fgm']) -
                (0.7 * result['fga']) -
                (0.4 * (result['fta'] - result['ftm'])) +
                (0.7 * result['OffReb']) +
                (0.3 * result['DefReb']) +
                result['stl'] +
                (0.7 * result['ast']) +
                (0.7 * result['blk']) -
                (0.4 * result['PF']) -
                result['TOV']
            )

        return result
