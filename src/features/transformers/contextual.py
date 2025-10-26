"""
Contextual features transformer.

Adds game context features that affect player performance:
- Home/away indicator
- Rest days since last game
- Back-to-back game indicator
"""

import pandas as pd
import numpy as np
from src.features.base import FeatureTransformer


class ContextualFeaturesTransformer(FeatureTransformer):
    """
    Add contextual features related to game circumstances.

    Features:
    - home_game: 1 if playing at home, 0 if away
    - rest_days: Days since last game (capped at 7)
    - back_to_back: 1 if playing on consecutive days, 0 otherwise
    - days_off_3plus: 1 if 3+ days rest, 0 otherwise
    """

    def __init__(self, max_rest_days: int = 7):
        """
        Initialize contextual features transformer.

        Parameters
        ----------
        max_rest_days : int
            Cap rest days at this value (default 7)
        """
        super().__init__()
        self.max_rest_days = max_rest_days

    def fit(self, data: pd.DataFrame) -> 'ContextualFeaturesTransformer':
        """
        Fit transformer (no-op for contextual features).

        Parameters
        ----------
        data : pd.DataFrame
            Training data

        Returns
        -------
        self
        """
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add contextual features to data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with columns: playerID, gameDate, team, teamAbv, gameID

        Returns
        -------
        pd.DataFrame
            Data with added contextual features
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        result = data.copy()

        # 1. Home/Away indicator
        result['home_game'] = self._calculate_home_away(result)

        # 2. Rest days since last game
        result['rest_days'] = self._calculate_rest_days(result)

        # 3. Back-to-back indicator
        result['back_to_back'] = (result['rest_days'] == 1).astype(int)

        # 4. Extended rest indicator (3+ days)
        result['days_off_3plus'] = (result['rest_days'] >= 3).astype(int)

        return result

    def _calculate_home_away(self, data: pd.DataFrame) -> pd.Series:
        """
        Determine if player is at home or away.

        Uses gameID format: awayTeam@homeTeam_YYYYMMDD
        """
        home_game = pd.Series(0, index=data.index, dtype=int)

        # Check if gameID column exists
        if 'gameID' not in data.columns:
            return home_game

        for idx, row in data.iterrows():
            game_id = row.get('gameID', '')
            team_abv = row.get('teamAbv', row.get('team', ''))

            if not game_id or not team_abv:
                continue

            # Parse gameID: awayTeam@homeTeam_YYYYMMDD
            if '@' in game_id:
                parts = game_id.split('@')
                if len(parts) == 2:
                    away_team = parts[0]
                    home_team = parts[1].split('_')[0] if '_' in parts[1] else parts[1]

                    # Check if player's team is home team
                    if team_abv.upper() == home_team.upper():
                        home_game.loc[idx] = 1

        return home_game

    def _calculate_rest_days(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate days since last game for each player.

        Returns
        -------
        pd.Series
            Rest days, capped at max_rest_days
        """
        rest_days = pd.Series(np.nan, index=data.index, dtype=float)

        # Ensure gameDate is datetime
        if 'gameDate' not in data.columns:
            return rest_days.fillna(self.max_rest_days)

        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['gameDate']):
            df['gameDate'] = pd.to_datetime(df['gameDate'], errors='coerce')

        # Sort by player and date
        df = df.sort_values(['playerID', 'gameDate'])

        # Calculate days since previous game for each player
        df['prev_game_date'] = df.groupby('playerID')['gameDate'].shift(1)
        df['rest_days_calc'] = (df['gameDate'] - df['prev_game_date']).dt.days

        # For first game of each player, use max rest days
        df['rest_days_calc'] = df['rest_days_calc'].fillna(self.max_rest_days)

        # Cap at max_rest_days
        df['rest_days_calc'] = df['rest_days_calc'].clip(upper=self.max_rest_days)

        # Map back to original index
        rest_days = df.set_index(data.index)['rest_days_calc']

        return rest_days.astype(int)
