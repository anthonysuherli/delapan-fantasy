import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "nba_dfs.db"
GAMES_DIR = Path("C:\\Users\\antho\\OneDrive\\Documents\\Repositories\\delapan-fantasy\\data\\games")
PLAYER_LOGS_DIR = Path("C:\\Users\\antho\\OneDrive\\Documents\\Repositories\\delapan-fantasy\\data\\player_logs")


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


def create_player_logs_table(conn):
    """Create player_logs_extracted table if it doesn't exist."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_logs_extracted (
            playerID TEXT,
            longName TEXT,
            team TEXT,
            teamAbv TEXT,
            teamID TEXT,
            gameID TEXT,
            gameDate TEXT,
            pos TEXT,
            mins TEXT,
            pts TEXT,
            reb TEXT,
            ast TEXT,
            stl TEXT,
            blk TEXT,
            TOV TEXT,
            PF TEXT,
            fga TEXT,
            fgm TEXT,
            fgp TEXT,
            fta TEXT,
            ftm TEXT,
            ftp TEXT,
            tptfga TEXT,
            tptfgm TEXT,
            tptfgp TEXT,
            OffReb TEXT,
            DefReb TEXT,
            fantasyPoints TEXT,
            fantasyPts TEXT,
            plusMinus TEXT,
            usage TEXT,
            tech TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (playerID, gameID)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_player_logs_date
        ON player_logs_extracted(gameDate)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_player_logs_player
        ON player_logs_extracted(playerID)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_player_logs_team
        ON player_logs_extracted(teamAbv)
    """)

    conn.commit()
    print("Player logs table created/verified")


def flatten_player_stats(player_stats_dict):
    """Extract nested playerStats dictionary into flat structure."""
    if not isinstance(player_stats_dict, dict):
        return {}

    flattened = {}
    for key, value in player_stats_dict.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[nested_key] = nested_value
        else:
            flattened[key] = value

    return flattened


def load_player_logs_from_parquet(conn):
    """Load and flatten player logs from parquet files into database."""

    parquet_files = sorted(PLAYER_LOGS_DIR.glob("**/*.parquet"))

    print(f"\nFound {len(parquet_files)} player log parquet files to process")
    print("=" * 80)

    total_logs_inserted = 0
    total_logs_skipped = 0
    files_processed = 0

    for file_path in parquet_files:
        try:
            try:
                df = pd.read_parquet(file_path, engine='pyarrow')
            except Exception:
                df = pd.read_parquet(file_path, engine='fastparquet')

            if df.empty:
                continue

            if 'playerStats' not in df.columns:
                continue

            logs_in_file = 0
            skipped_in_file = 0

            for _, row in df.iterrows():
                try:
                    player_stats = row.get('playerStats', {})

                    if isinstance(player_stats, str):
                        import json
                        player_stats = json.loads(player_stats)

                    if not isinstance(player_stats, dict):
                        skipped_in_file += 1
                        continue

                    flattened = flatten_player_stats(player_stats)

                    if not flattened.get('playerID'):
                        skipped_in_file += 1
                        continue

                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO player_logs_extracted
                        (playerID, longName, team, teamAbv, teamID, gameID, gameDate,
                         pos, mins, pts, reb, ast, stl, blk, TOV, PF,
                         fga, fgm, fgp, fta, ftm, ftp, tptfga, tptfgm, tptfgp,
                         OffReb, DefReb, fantasyPoints, fantasyPts, plusMinus, usage, tech)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        flattened.get('playerID', ''),
                        flattened.get('longName', ''),
                        flattened.get('team', ''),
                        flattened.get('teamAbv', row.get('teamAbv', '')),
                        flattened.get('teamID', row.get('teamID', '')),
                        row.get('gameID', ''),
                        row.get('gameDate', ''),
                        flattened.get('pos', ''),
                        flattened.get('mins', ''),
                        flattened.get('pts', ''),
                        flattened.get('reb', ''),
                        flattened.get('ast', ''),
                        flattened.get('stl', ''),
                        flattened.get('blk', ''),
                        flattened.get('TOV', ''),
                        flattened.get('PF', ''),
                        flattened.get('fga', ''),
                        flattened.get('fgm', ''),
                        flattened.get('fgp', ''),
                        flattened.get('fta', ''),
                        flattened.get('ftm', ''),
                        flattened.get('ftp', ''),
                        flattened.get('tptfga', ''),
                        flattened.get('tptfgm', ''),
                        flattened.get('tptfgp', ''),
                        flattened.get('OffReb', ''),
                        flattened.get('DefReb', ''),
                        flattened.get('fantasyPoints', ''),
                        flattened.get('fantasyPts', ''),
                        flattened.get('plusMinus', ''),
                        flattened.get('usage', ''),
                        flattened.get('tech', '')
                    ))

                    if cursor.rowcount > 0:
                        logs_in_file += 1
                    else:
                        skipped_in_file += 1

                except Exception:
                    skipped_in_file += 1

            conn.commit()

            total_logs_inserted += logs_in_file
            total_logs_skipped += skipped_in_file
            files_processed += 1

            if logs_in_file > 0:
                print(f"[{files_processed:3d}] {file_path.relative_to(PLAYER_LOGS_DIR)}: {logs_in_file} inserted, {skipped_in_file} skipped")

        except Exception:
            continue

    print("\n" + "=" * 80)
    print("PLAYER LOGS LOAD COMPLETE")
    print("=" * 80)
    print(f"Files processed:     {files_processed}")
    print(f"Logs inserted:       {total_logs_inserted}")
    print(f"Logs skipped:        {total_logs_skipped} (duplicates)")
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
    print("DATABASE VERIFICATION - GAMES")
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

    cursor.execute("SELECT COUNT(*) FROM player_logs_extracted")
    total_logs = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(gameDate), MAX(gameDate) FROM player_logs_extracted")
    log_min_date, log_max_date = cursor.fetchone()

    cursor.execute("SELECT COUNT(DISTINCT playerID) FROM player_logs_extracted")
    unique_players = cursor.fetchone()[0]

    print("\n" + "=" * 80)
    print("DATABASE VERIFICATION - PLAYER LOGS")
    print("=" * 80)
    print(f"Total player logs:   {total_logs}")
    print(f"Date range:          {log_min_date} to {log_max_date}")
    print(f"Unique players:      {unique_players}")
    print("=" * 80)


def main():
    if not PLAYER_LOGS_DIR.exists():
        print(f"ERROR: Player logs directory not found: {PLAYER_LOGS_DIR}")
        return 1

    print("NBA Player Logs Extractor")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Player logs source: {PLAYER_LOGS_DIR}")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)

    try:
        create_player_logs_table(conn)

        load_player_logs_from_parquet(conn)

        verify_data(conn)

    finally:
        conn.close()

    print("\nDatabase update complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
