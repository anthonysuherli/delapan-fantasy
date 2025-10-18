import pandas as pd
from ..base import FeatureTransformer


class TargetTransformer(FeatureTransformer):
    """Create target variable for prediction"""

    def __init__(self, target_col: str = 'fpts', shift_periods: int = -1):
        """
        Initialize target transformer.

        Args:
            target_col: Column to use as target (default: 'fpts')
            shift_periods: Number of periods to shift (default: -1 for next game)
        """
        super().__init__('target')
        self.target_col = target_col
        self.shift_periods = shift_periods

    def fit(self, data: pd.DataFrame) -> 'TargetTransformer':
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
        Transform data to add target variable.

        Args:
            data: Data to transform

        Returns:
            DataFrame with target column added

        Raises:
            ValueError: If transformer has not been fitted or target_col not found
        """
        if not self._fitted:
            raise ValueError(f"Transformer '{self.name}' has not been fitted")

        df = data.copy()

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")

        if 'playerID' in df.columns:
            df['target'] = df.groupby('playerID')[self.target_col].shift(
                self.shift_periods
            )
        else:
            df['target'] = df[self.target_col].shift(self.shift_periods)

        return df
