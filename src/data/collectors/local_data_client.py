import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from src.config.paths import DATA_DIR

logger = logging.getLogger(__name__)


class LocalDataClient:
    """
    Local data client for accessing historical player data from parquet files.
    Uses DuckDB for querying parquet files efficiently.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = DATA_DIR

        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect(':memory:')
        self.request_count = 0
        self.rate_limit = 999999

    def get_player_game_logs(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Build path to player logs parquet files
            path_pattern = str(self.data_dir / 'player_logs_extracted' / '**' / '*.parquet')

            # Build query with player filter
            query = f"""
                SELECT * FROM read_parquet('{path_pattern}', hive_partitioning=1)
                WHERE playerID = '{player_id}'
            """

            # Add season filter if provided
            if season:
                query += f" AND gameDate LIKE '{season}%'"

            query += " ORDER BY gameDate DESC"

            df = self.conn.execute(query).df()

            self.request_count += 1
            logger.info(f"Loaded {len(df)} game logs for player {player_id}, season {season}")

            return {
                'statusCode': 200,
                'body': df.to_dict('records') if not df.empty else []
            }

        except Exception as e:
            logger.error(f"Failed to load game logs for player {player_id}: {str(e)}")
            return {
                'statusCode': 500,
                'body': []
            }

    def get_request_count(self) -> int:
        return self.request_count

    def get_remaining_requests(self) -> int:
        return self.rate_limit - self.request_count

    def __del__(self):
        """Close DuckDB connection on cleanup"""
        if hasattr(self, 'conn'):
            self.conn.close()
