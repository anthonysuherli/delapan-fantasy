import pandas as pd
import numpy as np
from typing import List
from ..base import FeatureTransformer


class EWMATransformer(FeatureTransformer):
    """Calculate exponentially weighted moving average features"""

    def __init__(
        self,
        span: int = 3,
        stats: List[str] = ['pts', 'reb', 'ast', 'stl', 'blk', 'mins']
    ):
        """
        Initialize EWMA transformer.

        Args:
            span: Span for exponential weighting (higher = more smoothing)
            stats: List of statistics to calculate EWMA for
        """
        super().__init__('ewma')
        self.span = span
        self.stats = stats

    def fit(self, data: pd.DataFrame) -> 'EWMATransformer':
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
        Transform data to add EWMA features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with EWMA features added

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

        new_columns = {}

        for stat in self.stats:
            if stat not in df.columns:
                continue

            df[stat] = pd.to_numeric(df[stat], errors='coerce')

            col_name = f'ewma_{stat}_{self.span}'
            new_columns[col_name] = df.groupby('playerID')[stat].transform(
                lambda x: x.shift(1).ewm(span=self.span, adjust=False).mean()
            )

        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

        return df
