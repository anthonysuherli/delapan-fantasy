import pandas as pd
import duckdb
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm
from .base import DataLoader

logger = logging.getLogger(__name__)


class HistoricalDataLoader(DataLoader):
    """Load historical DFS data using DuckDB from parquet files"""

    def __init__(self, data_dir: str = 'data'):
        """
        Initialize historical data loader with DuckDB.

        Args:
            data_dir: Base directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect(':memory:')

    @staticmethod
    def get_season_start_date(target_date: str) -> str:
        """
        Get the start date of the NBA season for a given date.

        NBA seasons run from October to June. Season year is based on the
        starting year (e.g., 2024-25 season starts Oct 2024).

        Args:
            target_date: Date in YYYYMMDD format

        Returns:
            Season start date in YYYYMMDD format (October 1st of season year)
        """
        target_dt = datetime.strptime(str(target_date), '%Y%m%d')

        if target_dt.month >= 10:
            season_year = target_dt.year
        else:
            season_year = target_dt.year - 1

        season_start = datetime(season_year, 10, 1)
        return season_start.strftime('%Y%m%d')

    @staticmethod
    def get_previous_season_start_date(target_date: str) -> str:
        """
        Get the start date of the previous NBA season.

        Args:
            target_date: Date in YYYYMMDD format

        Returns:
            Previous season start date in YYYYMMDD format
        """
        current_season_start = HistoricalDataLoader.get_season_start_date(target_date)
        current_season_dt = datetime.strptime(str(current_season_start), '%Y%m%d')
        previous_season_dt = datetime(current_season_dt.year - 1, 10, 1)
        return previous_season_dt.strftime('%Y%m%d')

    def _get_parquet_path(self, data_type: str, date: str = None) -> str:
        """
        Get parquet file path pattern for data type.

        Args:
            data_type: Type of data (e.g., 'dfs_salaries', 'betting_odds')
            date: Optional date in YYYYMMDD format for specific file

        Returns:
            File path pattern for DuckDB to read
        """
        base_path = self.data_dir / data_type
        
        if date:
            # Convert date YYYYMMDD to path structure YYYY/MM/DD.parquet
            year = date[:4]
            month = date[4:6]
            day = date[6:8]
            return str(base_path / year / month / f"{day}.parquet")
        else:
            # Use wildcard pattern for all files
            return str(base_path / "**" / "*.parquet")

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
        logger.info(f"Loading slate data for {date}")

        available_types = ['dfs_salaries', 'games', 'betting_odds', 'injuries']
        types_to_load = data_types or available_types

        slate_data = {
            'date': date
        }

        for data_type in tqdm(types_to_load, desc="Loading slate data", leave=False):
            try:
                path = self._get_parquet_path(data_type, date)
                
                if Path(path).exists():
                    query = f"""
                        SELECT * FROM read_parquet('{path}')
                    """
                    data = self.conn.execute(query).df()

                    # Expand DFS salaries from nested structure
                    if data_type == 'dfs_salaries' and not data.empty and 'draftkings' in data.columns:
                        dk_data = data['draftkings'].iloc[0]
                        if isinstance(dk_data, (list, tuple)) or hasattr(dk_data, '__iter__'):
                            # Convert array of dicts to DataFrame
                            expanded_data = []
                            for player in dk_data:
                                if isinstance(player, dict):
                                    player_data = player.copy()
                                    player_data['gameDate'] = date
                                    # Convert salary to int
                                    if 'salary' in player_data:
                                        player_data['salary'] = int(player_data['salary'])
                                    expanded_data.append(player_data)

                            if expanded_data:
                                data = pd.DataFrame(expanded_data)
                                logger.debug(f"Expanded {len(data)} DraftKings players from nested structure")
                else:
                    data = pd.DataFrame()

                slate_data[data_type] = data
                logger.debug(f"Loaded {len(data)} rows for {data_type}")

            except Exception as e:
                logger.warning(f"Failed to load {data_type} for {date}: {e}")
                slate_data[data_type] = pd.DataFrame()

        logger.info(
            f"Loaded slate data: "
            f"{len(slate_data.get('dfs_salaries', []))} salaries, "
            f"{len(slate_data.get('games', []))} games"
        )

        return slate_data

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
        logger.info(f"Loading historical data from {start_date} to {end_date}")

        available_types = ['dfs_salaries', 'games', 'betting_odds', 'injuries', 'player_logs_extracted']
        types_to_load = data_types or available_types

        historical_data = {}

        for data_type in tqdm(types_to_load, desc="Loading historical data", leave=False):
            try:
                path_pattern = self._get_parquet_path(data_type)
                
                # Build date filter based on common date column names
                date_col = self._get_date_column(data_type)
                
                query = f"""
                    SELECT * FROM read_parquet('{path_pattern}', hive_partitioning=1)
                    WHERE {date_col} >= '{start_date}' AND {date_col} <= '{end_date}'
                """
                
                data = self.conn.execute(query).df()
                historical_data[data_type] = data

                if not data.empty:
                    logger.info(
                        f"Loaded {len(data)} rows for {data_type} "
                        f"({data_type}: {start_date} to {end_date})"
                    )
                else:
                    logger.warning(f"No data found for {data_type} in date range")

            except Exception as e:
                logger.error(f"Failed to load {data_type}: {e}")
                historical_data[data_type] = pd.DataFrame()

        return historical_data

    def _get_date_column(self, data_type: str) -> str:
        """
        Get the appropriate date column name for a data type.

        Args:
            data_type: Type of data

        Returns:
            Name of the date column
        """
        date_columns = {
            'dfs_salaries': 'gameDate',
            'games': 'gameDate',
            'betting_odds': 'gameDate',
            'injuries': 'injDate',
            'player_logs_extracted': 'gameDate',
            'player_logs': 'gameDate'
        }
        return date_columns.get(data_type, 'gameDate')

    def load_historical_player_logs(
        self,
        start_date: str = None,
        end_date: str = None,
        num_seasons: int = 2,
        player_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load player game logs for training with strict temporal ordering.

        Loads data for current season and previous seasons. Default is current
        season + 1 previous season (total 2 seasons).

        Args:
            start_date: Start date in YYYYMMDD format (optional, overrides num_seasons)
            end_date: End date in YYYYMMDD format (exclusive)
            num_seasons: Number of seasons to load (default 2: current + previous)
                        Only used if start_date is not provided
            player_ids: Optional list of player IDs to filter for. If provided, only
                       loads data for these specific players.

        Returns:
            DataFrame with player logs before end_date

        Raises:
            ValueError: If lookahead bias detected or missing required parameters
        """
        if end_date is None:
            raise ValueError("end_date is required")

        if start_date is not None:
            logger.info(f"Loading historical player logs (from {start_date} to {end_date})")
        elif num_seasons == 1:
            start_date = self.get_season_start_date(end_date)
            logger.info(f"Loading historical player logs (up to {end_date}, current season from {start_date})")
        else:
            start_date = self.get_previous_season_start_date(end_date)
            current_season_start = self.get_season_start_date(end_date)
            logger.info(
                f"Loading historical player logs (up to {end_date}, "
                f"{num_seasons} seasons from {start_date}, "
                f"current season starts {current_season_start})"
            )

        if player_ids:
            logger.info(f"Filtering for {len(player_ids)} specific players")

        try:
            path_pattern = self._get_parquet_path('player_logs_extracted')
            date_col = 'gameDate'
            
            # Build query with optional player filter
            player_filter = ""
            if player_ids:
                # Create a comma-separated list of quoted player IDs
                player_list = "', '".join(player_ids)
                player_filter = f" AND playerID IN ('{player_list}')"
            
            query = f"""
                SELECT * FROM read_parquet('{path_pattern}', hive_partitioning=1)
                WHERE {date_col} >= '{start_date}' 
                  AND {date_col} < '{end_date}'{player_filter}
            """
            
            df = self.conn.execute(query).df()

            if df.empty:
                logger.warning(f"No historical data found for date range {start_date} to {end_date}")
                return pd.DataFrame()

            if 'gameDate' in df.columns:
                df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
                max_date_in_data = df['gameDate'].max()
                end_date_dt = datetime.strptime(str(end_date), '%Y%m%d')

                if max_date_in_data >= end_date_dt:
                    logger.error(f"LOOKAHEAD BIAS DETECTED: Data contains dates >= {end_date}")
                    raise ValueError(
                        f"Lookahead bias: max date in data ({max_date_in_data}) >= end_date ({end_date})"
                    )

                logger.info(
                    f"Loaded {len(df)} player logs from {df['gameDate'].min()} to {df['gameDate'].max()}"
                )

            return df

        except Exception as e:
            logger.error(f"Failed to load historical player logs: {str(e)}")
            raise

    def load_slate_injuries(
        self,
        date: str
    ) -> pd.DataFrame:
        """
        Load injury data for a specific slate date.

        Args:
            date: Date in YYYYMMDD format

        Returns:
            DataFrame with injury data for the slate date.
            Columns: designation, injDate, injReturnDate, playerID, description
        """
        logger.info(f"Loading injury data for {date}")

        try:
            path_pattern = self._get_parquet_path('injuries')
            
            # Injuries might span date ranges, so look for records where date falls within range
            query = f"""
                SELECT * FROM read_parquet('{path_pattern}', hive_partitioning=1)
                WHERE injDate <= '{date}' 
                  AND (injReturnDate IS NULL OR injReturnDate >= '{date}')
            """
            
            injuries = self.conn.execute(query).df()

            if not injuries.empty:
                logger.info(f"Loaded {len(injuries)} injury records for {date}")
            else:
                logger.warning(f"No injury data found for {date}")

            return injuries

        except Exception as e:
            logger.error(f"Failed to load injuries for {date}: {e}")
            return pd.DataFrame()

    def load_slate_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        Get list of dates with games in date range.

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format

        Returns:
            List of dates in YYYYMMDD format
        """
        try:
            path_pattern = self._get_parquet_path('games')
            
            query = f"""
                SELECT DISTINCT gameDate 
                FROM read_parquet('{path_pattern}', hive_partitioning=1)
                WHERE gameDate >= '{start_date}' AND gameDate <= '{end_date}'
                ORDER BY gameDate
            """
            
            schedule_data = self.conn.execute(query).df()

            if not schedule_data.empty and 'gameDate' in schedule_data.columns:
                slate_dates = schedule_data['gameDate'].tolist()
                logger.info(f"Found {len(slate_dates)} slate dates from {start_date} to {end_date}")
                return slate_dates
            else:
                logger.warning("No gameDate column found in schedule data")
                return []

        except Exception as e:
            logger.error(f"Failed to load slate dates: {str(e)}")
            return []

    def __del__(self):
        """Close DuckDB connection on cleanup"""
        if hasattr(self, 'conn'):
            self.conn.close()
