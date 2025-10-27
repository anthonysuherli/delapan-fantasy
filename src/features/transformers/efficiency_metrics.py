import pandas as pd
import numpy as np
from ..base import FeatureTransformer


class EfficiencyMetricsTransformer(FeatureTransformer):
    """Calculate advanced shooting efficiency metrics"""

    def __init__(self):
        """Initialize efficiency metrics transformer."""
        super().__init__('efficiency_metrics')

    def fit(self, data: pd.DataFrame) -> 'EfficiencyMetricsTransformer':
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
        Transform data to add efficiency metrics.

        Calculates:
        - eFG% (Effective Field Goal Percentage): (fgm + 0.5 × tptfgm) / fga
        - TS% (True Shooting Percentage): pts / (2 × (fga + 0.44 × fta))
        - FTR (Free Throw Rate): ftm / fga
        - TotalReb: OffReb + DefReb (if different from reb)

        Args:
            data: Data to transform

        Returns:
            DataFrame with efficiency features added

        Raises:
            ValueError: If transformer has not been fitted
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before transform")

        result = data.copy()

        # Effective Field Goal Percentage
        # Accounts for 3-pointers being worth more
        if all(col in result.columns for col in ['fgm', 'tptfgm', 'fga']):
            result['eFG_pct'] = np.where(
                result['fga'] > 0,
                (result['fgm'] + 0.5 * result['tptfgm']) / result['fga'],
                0
            )

        # True Shooting Percentage
        # Accounts for free throws and 3-pointers
        if all(col in result.columns for col in ['pts', 'fga', 'fta']):
            result['TS_pct'] = np.where(
                (result['fga'] + 0.44 * result['fta']) > 0,
                result['pts'] / (2 * (result['fga'] + 0.44 * result['fta'])),
                0
            )

        # Free Throw Rate
        # How often player gets to the line
        if all(col in result.columns for col in ['ftm', 'fga']):
            result['FTR'] = np.where(
                result['fga'] > 0,
                result['ftm'] / result['fga'],
                0
            )

        # Total Rebounds (if separated)
        if all(col in result.columns for col in ['OffReb', 'DefReb']):
            result['TotalReb'] = result['OffReb'] + result['DefReb']

        return result
