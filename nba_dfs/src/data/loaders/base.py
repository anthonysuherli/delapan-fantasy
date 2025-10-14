from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd


class DataLoader(ABC):
    """Abstract data loader interface"""

    @abstractmethod
    def load_slate_data(
        self,
        date: str,
        data_types: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all data for a specific slate date.

        Args:
            date: Date in YYYYMMDD format
            data_types: Optional list of data types to load. If None, load all.

        Returns:
            Dictionary mapping data type to DataFrame
        """
        pass

    @abstractmethod
    def load_historical_data(
        self,
        start_date: str,
        end_date: str,
        data_types: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data across a date range.

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            data_types: Optional list of data types to load

        Returns:
            Dictionary mapping data type to concatenated DataFrame
        """
        pass
