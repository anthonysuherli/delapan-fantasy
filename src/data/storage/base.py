from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseStorage(ABC):
    """Abstract storage interface for DFS data"""

    @abstractmethod
    def save(self, data: Any, data_type: str, identifier: str, **kwargs) -> None:
        """
        Save data to storage.

        Args:
            data: Data to save
            data_type: Type of data (e.g., 'betting_odds', 'dfs_salaries')
            identifier: Unique identifier for the data (e.g., date, game_id)
            **kwargs: Additional storage-specific parameters
        """
        pass

    @abstractmethod
    def load(self, data_type: str, filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Load data from storage with optional filters.

        Args:
            data_type: Type of data to load
            filters: Optional filter criteria (e.g., {'start_date': '20240101'})

        Returns:
            DataFrame containing loaded data
        """
        pass

    @abstractmethod
    def exists(self, data_type: str, identifier: str) -> bool:
        """
        Check if data exists in storage.

        Args:
            data_type: Type of data
            identifier: Unique identifier for the data

        Returns:
            True if data exists, False otherwise
        """
        pass
