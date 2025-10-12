"""
Script to collect NBA depth charts from Tank01 API.

Depth charts show player rotation priority for each position on each team.
This is valuable for:
- Understanding playing time expectations
- Identifying injury opportunities
- Projecting usage changes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
from dotenv import load_dotenv
import os

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.data.collectors.tank01_client import Tank01Client
from src.data.storage.parquet_storage import ParquetStorage
from src.data.storage.sqlite_storage import SQLiteStorage


def collect_depth_charts(
    client: Tank01Client,
    storage,
    team_abv: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Collect depth chart data from Tank01 API.

    Args:
        client: Tank01Client instance
        storage: Storage instance (ParquetStorage or SQLiteStorage)
        team_abv: Optional team abbreviation to filter by
        verbose: Print progress messages

    Returns:
        DataFrame with depth chart data
    """
    if verbose:
        print(f"Collecting depth charts...")
        if team_abv:
            print(f"  Filtering by team: {team_abv}")

    try:
        # Get depth charts data
        params = {}
        if team_abv:
            params['team_abv'] = team_abv

        response = client.get_depth_charts(**params)

        if response.get('statusCode') != 200:
            print(f"Error: API returned status code {response.get('statusCode')}")
            return pd.DataFrame()

        data = response.get('body', {})

        if not data:
            print("No depth chart data returned")
            return pd.DataFrame()

        # Process depth charts into DataFrame
        depth_chart_records = []
        collection_date = datetime.now().strftime('%Y%m%d')

        for team, positions in data.items():
            if isinstance(positions, dict):
                for position, players in positions.items():
                    if isinstance(players, list):
                        for depth_order, player_info in enumerate(players, 1):
                            if isinstance(player_info, dict):
                                record = {
                                    'team': team,
                                    'position': position,
                                    'depth_order': depth_order,
                                    'playerID': player_info.get('playerID'),
                                    'playerName': player_info.get('longName', player_info.get('name')),
                                    'collection_date': collection_date
                                }
                                depth_chart_records.append(record)
                            elif isinstance(player_info, str):
                                # Sometimes API returns just player names
                                record = {
                                    'team': team,
                                    'position': position,
                                    'depth_order': depth_order,
                                    'playerID': None,
                                    'playerName': player_info,
                                    'collection_date': collection_date
                                }
                                depth_chart_records.append(record)

        df = pd.DataFrame(depth_chart_records)

        if df.empty:
            print("No depth chart records created")
            return df

        if verbose:
            print(f"  Collected {len(df)} depth chart entries")
            print(f"  Teams: {df['team'].nunique()}")
            print(f"  Positions: {sorted(df['position'].unique())}")

        # Save to storage
        try:
            if isinstance(storage, ParquetStorage):
                filename = f"depth_charts_{collection_date}.parquet"
                if team_abv:
                    filename = f"depth_charts_{team_abv}_{collection_date}.parquet"

                storage.save(
                    'depth_charts',
                    df,
                    {'collection_date': collection_date}
                )
                if verbose:
                    print(f"  Saved to Parquet: {filename}")

            elif isinstance(storage, SQLiteStorage):
                # Create table if doesn't exist
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS depth_charts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    position TEXT NOT NULL,
                    depth_order INTEGER NOT NULL,
                    playerID TEXT,
                    playerName TEXT,
                    collection_date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team, position, depth_order, collection_date)
                )
                """
                storage.conn.execute(create_table_sql)

                # Insert data
                df['created_at'] = datetime.now()
                df.to_sql('depth_charts', storage.conn, if_exists='append', index=False)
                storage.conn.commit()

                if verbose:
                    print(f"  Saved {len(df)} records to SQLite")

        except Exception as e:
            print(f"Error saving data: {e}")

        return df

    except Exception as e:
        print(f"Error collecting depth charts: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Collect NBA depth charts')
    parser.add_argument('--team', type=str, help='Team abbreviation (e.g., LAL, BOS)')
    parser.add_argument('--all-teams', action='store_true', help='Collect all teams')
    parser.add_argument('--storage', type=str, choices=['parquet', 'sqlite'],
                       default='parquet', help='Storage type')
    parser.add_argument('--db-path', type=str, default='nba_dfs.db',
                       help='SQLite database path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('TANK01_API_KEY')

    if not api_key:
        print("Error: TANK01_API_KEY not found in environment variables")
        sys.exit(1)

    # Initialize client
    client = Tank01Client(api_key=api_key)

    # Initialize storage
    if args.storage == 'parquet':
        storage = ParquetStorage()
    else:
        db_path = repo_root / args.db_path
        storage = SQLiteStorage(str(db_path))

    print("=" * 50)
    print("NBA Depth Charts Collection")
    print("=" * 50)
    print(f"Storage: {args.storage}")
    if args.storage == 'sqlite':
        print(f"Database: {args.db_path}")
    print()

    # Collect data
    if args.all_teams or (not args.team):
        # Collect all teams
        df = collect_depth_charts(
            client=client,
            storage=storage,
            team_abv=None,
            verbose=args.verbose
        )
    else:
        # Collect specific team
        df = collect_depth_charts(
            client=client,
            storage=storage,
            team_abv=args.team,
            verbose=args.verbose
        )

    # Display summary
    if not df.empty:
        print("\n" + "=" * 50)
        print("Collection Summary")
        print("=" * 50)
        print(f"Total records: {len(df)}")
        print(f"Teams: {df['team'].nunique()}")
        print(f"Unique players: {df['playerName'].nunique()}")
        print(f"\nPosition distribution:")
        print(df['position'].value_counts())

        print(f"\nTop depth chart entries (first 10):")
        print(df[['team', 'position', 'depth_order', 'playerName']].head(10))

        print(f"\nAPI requests used: {client.get_request_count()}")
        print(f"Remaining requests: {client.get_remaining_requests()}")
    else:
        print("\nNo data collected")


if __name__ == '__main__':
    main()