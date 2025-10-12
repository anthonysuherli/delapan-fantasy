import pandas as pd
import numpy as np
from typing import List
from ..base import FeatureTransformer


class RollingStatsTransformer(FeatureTransformer):
    """Calculate rolling average and standard deviation statistics"""

    def __init__(
        self,
        windows: List[int] = [3, 5, 10],
        stats: List[str] = ['pts', 'reb', 'ast', 'stl', 'blk', 'mins'],
        include_std: bool = True
    ):
        """
        Initialize rolling stats transformer.

        Args:
            windows: List of window sizes for rolling calculations
            stats: List of statistics to calculate rolling features for
            include_std: Whether to calculate standard deviation in addition to mean
        """
        super().__init__('rolling_stats')
        self.windows = windows
        self.stats = stats
        self.include_std = include_std

    def fit(self, data: pd.DataFrame) -> 'RollingStatsTransformer':
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
        Transform data to add rolling average and std dev features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with rolling features added

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

            for window in self.windows:
                ma_col = f'{stat}_ma{window}'
                new_columns[ma_col] = df.groupby('playerID')[stat].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )

                if self.include_std:
                    std_col = f'{stat}_std{window}'
                    new_columns[std_col] = df.groupby('playerID')[stat].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                    )

        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

        return df

    def _calculate_features_from_prior_games(
        self,
        prior_games: pd.DataFrame,
        player_id: str
    ) -> dict:
        """
        Calculate rolling features from prior games for a single player.

        Args:
            prior_games: DataFrame of games before current date
            player_id: Player ID

        Returns:
            Dictionary of features
        """
        features = {'playerID': player_id}

        for stat in self.stats:
            if stat not in prior_games.columns:
                continue

            values = pd.to_numeric(prior_games[stat], errors='coerce').dropna()

            for window in self.windows:
                col_name = f'{stat}_rolling_{window}'
                if len(values) >= window:
                    features[col_name] = values.tail(window).mean()
                else:
                    features[col_name] = values.mean() if len(values) > 0 else 0

        return features
