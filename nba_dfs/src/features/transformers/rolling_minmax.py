import pandas as pd
import numpy as np
from typing import List
from ..base import FeatureTransformer


class RollingMinMaxTransformer(FeatureTransformer):
    """Calculate rolling min and max statistics"""

    def __init__(
        self,
        windows: List[int] = [3, 5, 10],
        stats: List[str] = ['pts', 'reb', 'ast', 'stl', 'blk', 'mins']
    ):
        """
        Initialize rolling min/max transformer.

        Args:
            windows: List of window sizes for rolling calculations
            stats: List of statistics to calculate rolling min/max for
        """
        super().__init__('rolling_minmax')
        self.windows = windows
        self.stats = stats

    def fit(self, data: pd.DataFrame) -> 'RollingMinMaxTransformer':
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
        Transform data to add rolling min and max features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with rolling min/max features added

        Raises:
            ValueError: If transformer has not been fitted
        """
        if not self._fitted:
            raise ValueError(f"Transformer '{self.name}' has not been fitted")

        df = data.copy()

        if 'gameDate' not in df.columns or 'playerID' not in df.columns:
            raise ValueError("Data must contain 'gameDate' and 'playerID' columns")

        df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['playerID', 'gameDate'])

        for stat in self.stats:
            if stat not in df.columns:
                continue

            df[stat] = pd.to_numeric(df[stat], errors='coerce')

            for window in self.windows:
                min_col = f'{stat}_min{window}'
                max_col = f'{stat}_max{window}'

                df[min_col] = df.groupby('playerID')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                df[max_col] = df.groupby('playerID')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )

        return df
