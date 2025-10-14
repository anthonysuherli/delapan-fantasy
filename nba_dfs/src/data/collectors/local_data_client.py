import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from src.config.paths import DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)


class LocalDataClient:

    def __init__(self, db_path: str = None, data_dir: str = None):
        if db_path is None:
            db_path = DB_PATH
        if data_dir is None:
            data_dir = DATA_DIR

        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.request_count = 0
        self.rate_limit = 999999

    def get_player_game_logs(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT * FROM player_logs
                WHERE playerID = ?
            """
            params = [player_id]

            if season:
                query += " AND gameDate LIKE ?"
                params.append(f"{season}%")

            query += " ORDER BY gameDate DESC"

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

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
