from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class PlayerFilter(ABC):
    """Abstract base for player filtering"""

    def __init__(self, name: str):
        """
        Initialize player filter.

        Args:
            name: Unique name for this filter
        """
        self.name = name

    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter to player data.

        Args:
            data: Player data to filter

        Returns:
            Filtered DataFrame

        Raises:
            ValueError: If filter cannot be applied
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
