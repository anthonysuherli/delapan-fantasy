import pandas as pd
from typing import List, Optional
from .base import PlayerFilter


class CompositeFilter(PlayerFilter):
    """Combine multiple filters with AND or OR logic"""

    def __init__(
        self,
        filters: List[PlayerFilter],
        logic: str = 'and',
        name: Optional[str] = None
    ):
        """
        Initialize composite filter.

        Args:
            filters: List of filters to combine
            logic: Combination logic ('and' or 'or')
            name: Optional filter name

        Examples:
            CompositeFilter([
                ColumnFilter('salary', '>', 5000),
                InjuryFilter(exclude_out=True)
            ], logic='and')
        """
        if name is None:
            name = f"composite_{logic}_{'_'.join([f.name for f in filters])}"

        super().__init__(name)

        if logic not in ['and', 'or']:
            raise ValueError("Logic must be 'and' or 'or'")

        self.filters = filters
        self.logic = logic

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply composite filter to player data.

        Args:
            data: Player data to filter

        Returns:
            Filtered DataFrame
        """
        if data.empty:
            return data

        if not self.filters:
            return data

        if self.logic == 'and':
            result = data.copy()
            for filter_obj in self.filters:
                result = filter_obj.apply(result)
            return result
        else:
            masks = []
            for filter_obj in self.filters:
                try:
                    filtered = filter_obj.apply(data)
                    mask = data.index.isin(filtered.index)
                    masks.append(mask)
                except Exception:
                    continue

            if not masks:
                return pd.DataFrame(columns=data.columns)

            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask |= mask

            return data[combined_mask].copy()

    def __repr__(self) -> str:
        filter_names = [f.name for f in self.filters]
        return f"CompositeFilter(filters={filter_names}, logic='{self.logic}')"
