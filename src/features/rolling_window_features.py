import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class RollingWindowFeatureCalculator:

    def __init__(self, window_sizes: List[int] = [3, 5, 10]):
        self.window_sizes = window_sizes
        self.stat_columns = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV']

    def calculate_features(
        self,
        player_name: str,
        game_date: str,
        historical_data: pd.DataFrame
    ) -> Optional[Dict[str, float]]:

        player_history = historical_data[
            (historical_data['longName'] == player_name) &
            (historical_data['gameDate'] < game_date)
        ].sort_values('gameDate')

        if player_history.empty or len(player_history) < min(self.window_sizes):
            return None

        features = {}

        for col in self.stat_columns:
            if col not in player_history.columns:
                continue

            values = pd.to_numeric(player_history[col], errors='coerce').dropna()

            if len(values) == 0:
                continue

            for window in self.window_sizes:
                if len(values) >= window:
                    window_values = values.tail(window)
                    features[f'{col}_avg_{window}'] = window_values.mean()
                    features[f'{col}_std_{window}'] = window_values.std()
                    features[f'{col}_min_{window}'] = window_values.min()
                    features[f'{col}_max_{window}'] = window_values.max()
                else:
                    features[f'{col}_avg_{window}'] = values.mean()
                    features[f'{col}_std_{window}'] = values.std()
                    features[f'{col}_min_{window}'] = values.min()
                    features[f'{col}_max_{window}'] = values.max()

        return features
