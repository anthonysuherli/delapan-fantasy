import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json
from .base import BaseStorage

logger = logging.getLogger(__name__)


class SQLiteStorage(BaseStorage):
    """SQLite-based storage adapter implementing BaseStorage interface"""

    def __init__(self, db_path: str = "nba_dfs.db"):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file

        Raises:
            FileNotFoundError: If database file does not exist
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        logger.info(f"Initialized SQLiteStorage with database: {self.db_path}")

    @staticmethod
    def _reconstruct_positions(primary_pos: str) -> list:
        """
        Reconstruct position eligibility from primary position.

        Note: Original allValidPositions data was corrupted in database.
        This reconstructs a conservative estimate based on primary position.
        Players may have additional position eligibility not captured here.

        Args:
            primary_pos: Primary position (PG, SG, SF, PF, C)

        Returns:
            List of eligible positions
        """
        if pd.isna(primary_pos):
            return []

        primary_pos = str(primary_pos).strip().upper()

        position_map = {
            'PG': ['PG'],
            'SG': ['SG'],
            'SF': ['SF'],
            'PF': ['PF'],
            'C': ['C'],
            'G': ['PG', 'SG'],
            'F': ['SF', 'PF']
        }

        return position_map.get(primary_pos, [primary_pos])

    def save(self, data: Any, data_type: str, identifier: str, **kwargs) -> None:
        """
        Save data to SQLite database.

        Args:
            data: Data to save (dict or DataFrame)
            data_type: Type of data (e.g., 'betting_odds', 'dfs_salaries')
            identifier: Unique identifier (e.g., date string)
            **kwargs: Additional parameters

        Note:
            SQLite storage primarily used for reading existing data.
            Write operations require appropriate table schemas.
        """
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        table_map = {
            'box_scores': 'player_logs_extracted',
            'dfs_salaries': 'dfs_salaries',
            'schedule': 'games',
            'injuries': 'injuries'
        }

        table_name = table_map.get(data_type, data_type)

        for col in df.columns:
            if df[col].dtype == 'object' and len(df) > 0:
                first_val = df[col].iloc[0]
                if isinstance(first_val, (list, dict, np.ndarray)):
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x.tolist() if isinstance(x, np.ndarray) else x) if x is not None else None
                    )
                    logger.debug(f"Serialized {col} to JSON for SQLite storage")

        try:
            conn = sqlite3.connect(str(self.db_path))
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            logger.info(f"Saved {len(df)} rows to {table_name}")
        except Exception as e:
            logger.error(f"Failed to save to SQLite: {e}")
            raise

    def load(self, data_type: str, filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Load data from SQLite database with optional filters.

        Args:
            data_type: Type of data to load
            filters: Optional filters (e.g., {'start_date': '20240101', 'end_date': '20240331'})

        Returns:
            DataFrame containing loaded data
        """
        table_map = {
            'box_scores': 'player_logs_extracted',
            'dfs_salaries': 'dfs_salaries',
            'schedule': 'games',
            'injuries': 'injuries',
            'betting_odds': 'betting_odds'
        }

        date_column_map = {
            'box_scores': 'gameDate',
            'schedule': 'gameDate',
            'dfs_salaries': 'date',
            'injuries': 'injDate',
            'betting_odds': 'date'
        }

        table_name = table_map.get(data_type, data_type)
        date_column = date_column_map.get(data_type, 'date')

        try:
            conn = sqlite3.connect(str(self.db_path))

            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                logger.debug(f"Table {table_name} does not exist, returning empty DataFrame")
                conn.close()
                return pd.DataFrame()

            if filters and ('start_date' in filters or 'end_date' in filters):
                start_date = filters.get('start_date')
                end_date = filters.get('end_date')

                if start_date and end_date:
                    if start_date == end_date:
                        query = f"SELECT * FROM {table_name} WHERE {date_column} = ?"
                        df = pd.read_sql_query(query, conn, params=(start_date,))
                    else:
                        query = f"SELECT * FROM {table_name} WHERE {date_column} >= ? AND {date_column} < ?"
                        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
                elif start_date:
                    query = f"SELECT * FROM {table_name} WHERE {date_column} >= ?"
                    df = pd.read_sql_query(query, conn, params=(start_date,))
                elif end_date:
                    query = f"SELECT * FROM {table_name} WHERE {date_column} < ?"
                    df = pd.read_sql_query(query, conn, params=(end_date,))
                else:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            else:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

            conn.close()

            if not df.empty:
                if 'allValidPositions' in df.columns:
                    def deserialize_position(val, pos):
                        if isinstance(val, bytes):
                            return self._reconstruct_positions(pos)
                        elif isinstance(val, str):
                            try:
                                return json.loads(val) if val else []
                            except (json.JSONDecodeError, TypeError):
                                return self._reconstruct_positions(pos)
                        elif isinstance(val, list):
                            return val
                        else:
                            return self._reconstruct_positions(pos)

                    df['allValidPositions'] = df.apply(
                        lambda row: deserialize_position(row['allValidPositions'], row['pos']),
                        axis=1
                    )
                    logger.debug("Processed allValidPositions column (JSON/binary/list)")

                logger.debug(f"Loaded {len(df)} rows from {table_name}")
            else:
                logger.debug(f"No data found in {table_name} for filters: {filters}")

            return df

        except Exception as e:
            logger.warning(f"Failed to load {data_type} from SQLite: {e}")
            return pd.DataFrame()

    def exists(self, data_type: str, identifier: str) -> bool:
        """
        Check if data exists in SQLite database.

        Args:
            data_type: Type of data
            identifier: Unique identifier (e.g., date string)

        Returns:
            True if data exists
        """
        table_map = {
            'box_scores': 'player_logs_extracted',
            'dfs_salaries': 'dfs_salaries',
            'schedule': 'games',
            'injuries': 'injuries'
        }

        date_column_map = {
            'box_scores': 'gameDate',
            'schedule': 'gameDate',
            'dfs_salaries': 'date',
            'injuries': 'injDate',
            'betting_odds': 'date'
        }

        table_name = table_map.get(data_type, data_type)
        date_column = date_column_map.get(data_type, 'date')

        try:
            conn = sqlite3.connect(str(self.db_path))
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {date_column} = ?"
            result = pd.read_sql_query(query, conn, params=(identifier,))
            conn.close()
            return result.iloc[0, 0] > 0
        except Exception as e:
            logger.debug(f"Error checking existence for {data_type}/{identifier}: {e}")
            return False
