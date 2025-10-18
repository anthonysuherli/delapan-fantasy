import pandas as pd
import numpy as np
from typing import Optional
from ..base import FeatureTransformer


class InjuryTransformer(FeatureTransformer):
    """Merge injury data and create injury-related features"""

    def __init__(self):
        """Initialize injury transformer."""
        super().__init__('injury')

    def fit(self, data: pd.DataFrame) -> 'InjuryTransformer':
        """
        Fit transformer on training data.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        self._fitted = True
        return self

    def transform(
        self,
        data: pd.DataFrame,
        injuries: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Transform data by merging injury information.

        Args:
            data: Data to transform (must contain playerID column)
            injuries: Injury DataFrame with columns:
                      designation, injDate, injReturnDate, playerID, description

        Returns:
            DataFrame with injury features added

        Raises:
            ValueError: If transformer has not been fitted or required columns missing
        """
        if not self._fitted:
            raise ValueError(f"Transformer '{self.name}' has not been fitted")

        if 'playerID' not in data.columns:
            raise ValueError("Data must contain 'playerID' column")

        df = data.copy()

        if injuries is None or injuries.empty:
            df['injury_status'] = 'Healthy'
            df['injury_designation'] = None
            df['injury_description'] = None
            df['is_injured'] = 0
            df['is_out'] = 0
            df['is_questionable'] = 0
            df['is_doubtful'] = 0
            df['is_day_to_day'] = 0
            return df

        injuries_processed = injuries.copy()

        if 'playerID' not in injuries_processed.columns:
            raise ValueError("Injuries DataFrame must contain 'playerID' column")

        injuries_processed = injuries_processed[
            ['playerID', 'designation', 'injDate', 'injReturnDate', 'description']
        ].copy()

        injuries_processed.columns = [
            'playerID',
            'injury_designation',
            'injury_date',
            'injury_return_date',
            'injury_description'
        ]

        df = df.merge(
            injuries_processed,
            on='playerID',
            how='left'
        )
        
        print(f"df: {df.columns}")
        df['injury_status'] = df['injury_designation'].fillna('Healthy')

        df['is_injured'] = (df['injury_designation'].notna()).astype(int)

        df['is_out'] = (
            df['injury_designation'].str.upper().str.contains('OUT', na=False)
        ).astype(int)

        df['is_questionable'] = (
            df['injury_designation'].str.upper().str.contains('QUESTIONABLE', na=False)
        ).astype(int)

        df['is_doubtful'] = (
            df['injury_designation'].str.upper().str.contains('DOUBTFUL', na=False)
        ).astype(int)

        df['is_day_to_day'] = (
            df['injury_designation'].str.upper().str.contains('DAY-TO-DAY', na=False)
        ).astype(int)

        return df

    def merge_injuries(
        self,
        player_data: pd.DataFrame,
        injuries: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convenience method to merge injury data directly.

        Args:
            player_data: Player data with playerID column
            injuries: Injury data

        Returns:
            Merged DataFrame with injury features
        """
        if not self._fitted:
            self.fit(player_data)

        return self.transform(player_data, injuries)
