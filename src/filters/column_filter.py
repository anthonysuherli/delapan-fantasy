import pandas as pd
from typing import Any, Callable, Optional
from .base import PlayerFilter


class ColumnFilter(PlayerFilter):
    """Filter players based on column value conditions"""

    def __init__(
        self,
        column: str,
        operator: str,
        value: Any,
        name: Optional[str] = None
    ):
        """
        Initialize column-based filter.

        Args:
            column: Column name to filter on
            operator: Comparison operator ('>', '>=', '<', '<=', '==', '!=', 'in', 'not_in')
            value: Value to compare against
            name: Optional filter name (auto-generated if not provided)

        Examples:
            ColumnFilter('salary', '>', 5000)
            ColumnFilter('pos', 'in', ['PG', 'SG'])
            ColumnFilter('team', '==', 'LAL')
        """
        if name is None:
            name = f"{column}_{operator}_{value}"
        super().__init__(name)

        self.column = column
        self.operator = operator
        self.value = value

        self._operator_map = {
            '>': lambda df, col, val: df[col] > val,
            '>=': lambda df, col, val: df[col] >= val,
            '<': lambda df, col, val: df[col] < val,
            '<=': lambda df, col, val: df[col] <= val,
            '==': lambda df, col, val: df[col] == val,
            '!=': lambda df, col, val: df[col] != val,
            'in': lambda df, col, val: df[col].isin(val),
            'not_in': lambda df, col, val: ~df[col].isin(val)
        }

        if operator not in self._operator_map:
            raise ValueError(
                f"Invalid operator '{operator}'. "
                f"Must be one of: {list(self._operator_map.keys())}"
            )

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column filter to player data.

        Args:
            data: Player data to filter

        Returns:
            Filtered DataFrame

        Raises:
            ValueError: If column not found in data
        """
        if data.empty:
            return data

        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data")

        # Convert column to numeric if filtering with numeric value and numeric operator
        df = data.copy()
        if self.operator in ['>', '>=', '<', '<='] and isinstance(self.value, (int, float)):
            df[self.column] = pd.to_numeric(df[self.column], errors='coerce')

        mask = self._operator_map[self.operator](df, self.column, self.value)
        filtered_data = df[mask].copy()

        return filtered_data

    def __repr__(self) -> str:
        return f"ColumnFilter(column='{self.column}', operator='{self.operator}', value={self.value})"
