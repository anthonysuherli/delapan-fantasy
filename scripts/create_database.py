import os
import sqlite3
from pathlib import Path
import pandas as pd
from typing import List, Dict


class DatabaseBuilder:
    def __init__(self, data_dir: str, db_path: str):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def load_parquet_files(self, subdirectory: str) -> pd.DataFrame:
        parquet_dir = self.data_dir / subdirectory
        if not parquet_dir.exists():
            print(f"Directory {parquet_dir} does not exist")
            return pd.DataFrame()

        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"No parquet files found in {parquet_dir}")
            return pd.DataFrame()

        print(f"Loading {len(parquet_files)} files from {subdirectory}...")
        dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined_df)} rows from {subdirectory}")
        return combined_df

    def load_player_logs(self) -> pd.DataFrame:
        parquet_dir = self.data_dir / "player_logs"
        if not parquet_dir.exists():
            return pd.DataFrame()

        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        print(f"Loading {len(parquet_files)} player log files...")
        dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                if 'playerStats' in df.columns:
                    player_stats = df['playerStats'].apply(pd.Series)
                    if not player_stats.empty:
                        player_stats['gameID'] = df['gameID'].iloc[0]
                        player_stats['gameDate'] = df['gameDate'].iloc[0]
                        dfs.append(player_stats)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined_df)} player log rows")
        return combined_df

    def load_games_with_team_stats(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        parquet_dir = self.data_dir / "games"
        if not parquet_dir.exists():
            return pd.DataFrame(), pd.DataFrame()

        parquet_files = list(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame(), pd.DataFrame()

        print(f"Loading {len(parquet_files)} game files...")
        game_dfs = []

        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                game_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        games_df = pd.concat(game_dfs, ignore_index=True) if game_dfs else pd.DataFrame()

        print(f"Loaded {len(games_df)} game rows")

        return games_df, pd.DataFrame()

    def create_tables(self):
        print("Creating database tables...")

        dfs_salaries_df = self.load_parquet_files("dfs_salaries")
        if not dfs_salaries_df.empty:
            dfs_salaries_df.to_sql('dfs_salaries', self.conn, if_exists='replace', index=False)
            print(f"Created dfs_salaries table with {len(dfs_salaries_df)} rows")

        games_df, _ = self.load_games_with_team_stats()
        if not games_df.empty:
            games_df.to_sql('games', self.conn, if_exists='replace', index=False)
            print(f"Created games table with {len(games_df)} rows")

        player_logs_df = self.load_player_logs()
        if not player_logs_df.empty:
            player_logs_df.to_sql('player_logs', self.conn, if_exists='replace', index=False)
            print(f"Created player_logs table with {len(player_logs_df)} rows")

        injuries_df = self.load_parquet_files("injuries")
        if not injuries_df.empty:
            injuries_df.to_sql('injuries', self.conn, if_exists='replace', index=False)
            print(f"Created injuries table with {len(injuries_df)} rows")

        self.conn.commit()
        print("All tables created successfully")

    def create_indexes(self):
        print("Creating indexes...")
        cursor = self.conn.cursor()

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_dfs_salaries_playerID ON dfs_salaries(playerID)",
            "CREATE INDEX IF NOT EXISTS idx_games_gameID ON games(gameID)",
            "CREATE INDEX IF NOT EXISTS idx_games_gameDate ON games(gameDate)",
            "CREATE INDEX IF NOT EXISTS idx_player_logs_playerID ON player_logs(playerID)",
            "CREATE INDEX IF NOT EXISTS idx_player_logs_gameID ON player_logs(gameID)",
            "CREATE INDEX IF NOT EXISTS idx_player_logs_gameDate ON player_logs(gameDate)",
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                print(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                print(f"Error creating index: {e}")

        self.conn.commit()
        print("Indexes created successfully")

    def build(self):
        self.connect()
        try:
            self.create_tables()
            self.create_indexes()
        finally:
            self.close()


def main():
    data_dir = r"C:\Users\antho\OneDrive\Documents\Repositories\delapan-fantasy\data"
    db_path = r"C:\Users\antho\OneDrive\Documents\Repositories\delapan-fantasy\nba_dfs.db"

    builder = DatabaseBuilder(data_dir, db_path)
    builder.build()

    print(f"\nDatabase created at: {db_path}")


if __name__ == "__main__":
    main()
