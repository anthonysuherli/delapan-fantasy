import pandas as pd
from typing import List, Optional
from .base import PlayerFilter


class InjuryFilter(PlayerFilter):
    """Filter players based on injury status"""

    def __init__(
        self,
        exclude_out: bool = True,
        exclude_doubtful: bool = False,
        exclude_questionable: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize injury filter.

        Args:
            exclude_out: Exclude players ruled out (default: True)
            exclude_doubtful: Exclude doubtful players (default: False)
            exclude_questionable: Exclude questionable players (default: False)
            name: Optional filter name

        Examples:
            InjuryFilter(exclude_out=True)  # Exclude only OUT players
            InjuryFilter(exclude_out=True, exclude_doubtful=True)  # Exclude OUT and DOUBTFUL
        """
        if name is None:
            excluded = []
            if exclude_out:
                excluded.append('out')
            if exclude_doubtful:
                excluded.append('doubtful')
            if exclude_questionable:
                excluded.append('questionable')
            name = f"injury_exclude_{'_'.join(excluded) if excluded else 'none'}"

        super().__init__(name)

        self.exclude_out = exclude_out
        self.exclude_doubtful = exclude_doubtful
        self.exclude_questionable = exclude_questionable

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply injury filter to player data.

        Args:
            data: Player data to filter (must have injury columns)

        Returns:
            Filtered DataFrame

        Raises:
            ValueError: If required injury columns not found
        """
        if data.empty:
            return data

        required_cols = []
        if self.exclude_out:
            required_cols.append('is_out')
        if self.exclude_doubtful:
            required_cols.append('is_doubtful')
        if self.exclude_questionable:
            required_cols.append('is_questionable')

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Required injury columns not found: {missing_cols}. "
                f"Ensure InjuryTransformer has been applied to data."
            )

        mask = pd.Series([True] * len(data), index=data.index)

        if self.exclude_out and 'is_out' in data.columns:
            mask &= (data['is_out'] == 0)

        if self.exclude_doubtful and 'is_doubtful' in data.columns:
            mask &= (data['is_doubtful'] == 0)

        if self.exclude_questionable and 'is_questionable' in data.columns:
            mask &= (data['is_questionable'] == 0)

        filtered_data = data[mask].copy()

        return filtered_data

    def __repr__(self) -> str:
        return (
            f"InjuryFilter(exclude_out={self.exclude_out}, "
            f"exclude_doubtful={self.exclude_doubtful}, "
            f"exclude_questionable={self.exclude_questionable})"
        )
