from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from .base import BaseStorage
from ..storage_utils import get_partitioned_path, get_all_files_in_date_range


class ParquetStorage(BaseStorage):
    """Parquet-based storage wrapping existing utilities"""

    def __init__(self, base_dir: str = 'data/inputs'):
        """
        Initialize Parquet storage.

        Args:
            base_dir: Base directory for all data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        data: Any,
        data_type: str,
        identifier: str,
        **kwargs
    ) -> None:
        """
        Save data to Parquet file.

        Args:
            data: Data to save (dict or DataFrame)
            data_type: Type of data (e.g., 'betting_odds', 'dfs_salaries')
            identifier: Unique identifier (e.g., date string)
            **kwargs: Additional parameters
        """
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        path = get_partitioned_path(
            str(self.base_dir / data_type),
            identifier
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def load(
        self,
        data_type: str,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load data from Parquet files with optional filters.

        Args:
            data_type: Type of data to load
            filters: Optional filters (e.g., {'start_date': '20240101', 'end_date': '20240331'})

        Returns:
            DataFrame containing loaded data
        """
        start_date = filters.get('start_date') if filters else None
        end_date = filters.get('end_date') if filters else None

        files = get_all_files_in_date_range(
            str(self.base_dir / data_type),
            start_date=start_date,
            end_date=end_date
        )

        if not files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def exists(self, data_type: str, identifier: str) -> bool:
        """
        Check if data exists.

        Args:
            data_type: Type of data
            identifier: Unique identifier (e.g., date string)

        Returns:
            True if data exists
        """
        path = get_partitioned_path(
            str(self.base_dir / data_type),
            identifier
        )
        return path.exists()

    def save_betting_odds(self, data: Dict, date: str) -> None:
        """Save betting odds for a specific date"""
        self.save(data, 'betting_odds', date)

    def save_dfs_salaries(self, data: Dict, date: str) -> None:
        """Save DFS salaries for a specific date"""
        self.save(data, 'dfs_salaries', date)

    def save_projections(self, data: Dict, date: str) -> None:
        """Save projections for a specific date"""
        self.save(data, 'projections', date)

    def save_schedule(self, data: Dict, date: str) -> None:
        """Save schedule for a specific date"""
        self.save(data, 'schedule', date)

    def save_injuries(self, data: Dict, date: str) -> None:
        """Save injuries for a specific date"""
        self.save(data, 'injuries', date)

    def save_teams(self, data: Dict) -> None:
        """Save teams metadata"""
        df = pd.DataFrame(data) if isinstance(data, dict) else data
        path = self.base_dir / 'teams' / 'teams.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def save_box_score(self, data: Dict, game_id: str) -> None:
        """Save box score for a specific game"""
        self.save(data, 'box_scores', game_id)

    def save_depth_charts(self, data: Dict, date: str) -> None:
        """Save depth charts for a specific date"""
        self.save(data, 'depth_charts', date)

    def load_data(
        self,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data with optional date range filtering.

        Args:
            data_type: Type of data to load
            start_date: Optional start date in YYYYMMDD format
            end_date: Optional end date in YYYYMMDD format

        Returns:
            DataFrame containing loaded data
        """
        filters = {}
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date

        return self.load(data_type, filters if filters else None)
