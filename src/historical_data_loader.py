import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HistoricalDataLoader:

    def __init__(self, db_path: str = "nba_dfs.db"):
        self.db_path = db_path

        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        logger.info(f"Initialized HistoricalDataLoader with database: {self.db_path}")

    def load_slate_dates(self, start_date: str, end_date: str) -> List[str]:
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT DISTINCT gameDate
                FROM games
                WHERE gameDate >= ? AND gameDate <= ?
                ORDER BY gameDate
            """
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()

            slate_dates = df['gameDate'].tolist()
            logger.info(f"Found {len(slate_dates)} slate dates from {start_date} to {end_date}")

            return slate_dates

        except Exception as e:
            logger.error(f"Failed to load slate dates: {str(e)}")
            return []

    def load_slate_data(self, date: str) -> Dict[str, Any]:
        logger.info(f"Loading slate data for {date}")

        slate_data = {
            'date': date,
            'salaries': self._load_dfs_salaries(date),
            'schedule': self._load_schedule(date),
            'odds': self._load_betting_odds(date),
            'injuries': self._load_injuries(date)
        }

        logger.info(f"Loaded slate data: {len(slate_data['salaries'])} salaries, {len(slate_data['schedule'])} games")

        return slate_data

    def load_historical_player_logs(
        self,
        end_date: str,
        lookback_days: int
    ) -> pd.DataFrame:
        logger.info(f"Loading historical player logs (up to {end_date}, lookback: {lookback_days} days)")

        end_dt = datetime.strptime(end_date, '%Y%m%d')
        start_dt = end_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime('%Y%m%d')

        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT *
                FROM player_logs_extracted
                WHERE gameDate >= ? AND gameDate < ?
                ORDER BY gameDate, playerID
            """
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()

            if df.empty:
                logger.warning(f"No historical data found for date range {start_date} to {end_date}")
                return pd.DataFrame()

            max_date_in_data = df['gameDate'].max()
            if max_date_in_data >= end_date:
                logger.error(f"LOOKAHEAD BIAS DETECTED: Data contains dates >= {end_date}")
                raise ValueError(f"Lookahead bias: max date in data ({max_date_in_data}) >= end_date ({end_date})")

            logger.info(f"Loaded {len(df)} player logs from {df['gameDate'].min()} to {df['gameDate'].max()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load historical player logs: {str(e)}")
            return pd.DataFrame()

    def _load_dfs_salaries(self, date: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)

            games_query = """
                SELECT away, home
                FROM games
                WHERE gameDate = ?
            """
            games_df = pd.read_sql_query(games_query, conn, params=(date,))

            if games_df.empty:
                conn.close()
                logger.warning(f"No games found for {date}")
                return pd.DataFrame()

            teams_playing = set(games_df['away'].tolist() + games_df['home'].tolist())

            salaries_query = """
                SELECT *
                FROM dfs_salaries
                WHERE platform = 'DraftKings'
            """
            all_salaries = pd.read_sql_query(salaries_query, conn)
            conn.close()

            if all_salaries.empty:
                logger.warning(f"No DFS salaries found in database")
                return pd.DataFrame()

            slate_salaries = all_salaries[all_salaries['team'].isin(teams_playing)].copy()

            if 'salary' in slate_salaries.columns:
                slate_salaries['salary'] = pd.to_numeric(slate_salaries['salary'], errors='coerce')
            if 'dfs_salary' in slate_salaries.columns:
                slate_salaries['salary'] = pd.to_numeric(slate_salaries['dfs_salary'], errors='coerce')

            logger.debug(f"Loaded {len(slate_salaries)} DFS salaries for {date}")
            return slate_salaries

        except Exception as e:
            logger.warning(f"Failed to load DFS salaries for {date}: {str(e)}")
            return pd.DataFrame()

    def _load_schedule(self, date: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT *
                FROM games
                WHERE gameDate = ?
            """
            df = pd.read_sql_query(query, conn, params=(date,))
            conn.close()

            logger.debug(f"Loaded {len(df)} games for {date}")
            return df

        except Exception as e:
            logger.warning(f"Failed to load schedule for {date}: {str(e)}")
            return pd.DataFrame()

    def _load_betting_odds(self, date: str) -> pd.DataFrame:
        logger.debug(f"Betting odds not yet implemented for {date}")
        return pd.DataFrame()

    def _load_injuries(self, date: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT *
                FROM injuries
                WHERE gameDate = ?
            """
            df = pd.read_sql_query(query, conn, params=(date,))
            conn.close()

            logger.debug(f"Loaded {len(df)} injury reports for {date}")
            return df

        except Exception as e:
            logger.debug(f"No injury data for {date}: {str(e)}")
            return pd.DataFrame()
