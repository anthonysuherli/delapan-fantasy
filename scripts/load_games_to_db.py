import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "nba_dfs.db"
GAMES_DIR = Path("data/games")


def create_games_table(conn):
    """Create games table if it doesn't exist."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            gameID TEXT PRIMARY KEY,
            gameDate TEXT NOT NULL,
            teamIDAway TEXT,
            away TEXT,
            teamIDHome TEXT,
            home TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_date
        ON games(gameDate)
    """)

    conn.commit()
    print("Games table created/verified")


def load_games_from_parquet(conn):
    """Load all games from parquet files into database."""

    parquet_files = sorted(GAMES_DIR.glob("**/**.parquet"))

    print(f"\nFound {len(parquet_files)} parquet files to process")
    print("=" * 80)

    total_games_inserted = 0
    total_games_skipped = 0
    files_processed = 0

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)

            if df.empty:
                continue

            required_cols = ['gameID', 'gameDate', 'away', 'home']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {file_path.relative_to(GAMES_DIR)}: missing required columns")
                continue

            games_in_file = 0
            skipped_in_file = 0

            for _, row in df.iterrows():
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO games
                        (gameID, gameDate, teamIDAway, away, teamIDHome, home)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        row['gameID'],
                        row['gameDate'],
                        row.get('teamIDAway', None),
                        row['away'],
                        row.get('teamIDHome', None),
                        row['home']
                    ))

                    if cursor.rowcount > 0:
                        games_in_file += 1
                    else:
                        skipped_in_file += 1

                except Exception as e:
                    print(f"  Error inserting game {row.get('gameID', 'unknown')}: {str(e)}")
                    skipped_in_file += 1

            conn.commit()

            total_games_inserted += games_in_file
            total_games_skipped += skipped_in_file
            files_processed += 1

            if games_in_file > 0:
                print(f"[{files_processed:3d}] {file_path.relative_to(GAMES_DIR)}: {games_in_file} inserted, {skipped_in_file} skipped")

        except Exception as e:
            print(f"Error processing {file_path.relative_to(GAMES_DIR)}: {str(e)}")
            continue

    print("\n" + "=" * 80)
    print("LOAD COMPLETE")
    print("=" * 80)
    print(f"Files processed:     {files_processed}")
    print(f"Games inserted:      {total_games_inserted}")
    print(f"Games skipped:       {total_games_skipped} (duplicates)")
    print("=" * 80)


def verify_data(conn):
    """Verify loaded data."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM games")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(gameDate), MAX(gameDate) FROM games")
    min_date, max_date = cursor.fetchone()

    cursor.execute("SELECT COUNT(DISTINCT gameDate) FROM games")
    unique_dates = cursor.fetchone()[0]

    print("\n" + "=" * 80)
    print("DATABASE VERIFICATION")
    print("=" * 80)
    print(f"Total games in DB:   {total}")
    print(f"Date range:          {min_date} to {max_date}")
    print(f"Unique dates:        {unique_dates}")

    cursor.execute("""
        SELECT gameDate, COUNT(*) as games
        FROM games
        WHERE gameDate >= '20241022'
        ORDER BY gameDate DESC
        LIMIT 10
    """)

    print("\nRecent game dates (2024-2025 season):")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} games")

    print("=" * 80)


def main():
    if not GAMES_DIR.exists():
        print(f"ERROR: Games directory not found: {GAMES_DIR}")
        return 1

    print("NBA Games Database Loader")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Source: {GAMES_DIR}")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)

    try:
        create_games_table(conn)

        load_games_from_parquet(conn)

        verify_data(conn)

    finally:
        conn.close()

    print("\nDatabase update complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
