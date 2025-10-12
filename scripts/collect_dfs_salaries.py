"""
Collect DFS salaries, betting odds, and projections for a date range.

Usage:
    python scripts/collect_dfs_salaries.py --start-date 20241201 --end-date 20241231
    python scripts/collect_dfs_salaries.py --start-date 20241201 --end-date 20241231 --include-betting-odds
    python scripts/collect_dfs_salaries.py --start-date 20241201 --end-date 20241231 --include-all
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors.tank01_client import Tank01Client
from src.data.storage_utils import get_partitioned_path


def save_dfs_salaries(data: dict, date_str: str, base_dir: str = "./data"):
    if "body" not in data:
        return False

    body = data["body"]

    all_players = []
    for platform in ['draftkings']:
        if platform in body and body[platform]:
            platform_data = body[platform]
            if isinstance(platform_data, list):
                for player in platform_data:
                    player_copy = player.copy() if isinstance(player, dict) else {}
                    player_copy['platform'] = 'DraftKings'
                    player_copy['date'] = date_str
                    all_players.append(player_copy)

    if not all_players:
        return False

    df = pd.DataFrame(all_players)
    if df.empty:
        return False

    output_path = get_partitioned_path(
        base_dir=f"{base_dir}/dfs_salaries",
        date=date_str
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return True


def save_betting_odds(data: dict, date_str: str, base_dir: str = "./data/inputs"):
    if "body" not in data:
        return False

    df = pd.json_normalize(data["body"])
    if df.empty:
        return False

    output_path = get_partitioned_path(
        base_dir=f"{base_dir}/betting_odds",
        date=date_str
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return True


def save_projections(data: dict, date_str: str, base_dir: str = "./data/inputs"):
    if "body" not in data:
        return False

    df = pd.json_normalize(data["body"])
    if df.empty:
        return False

    output_path = get_partitioned_path(
        base_dir=f"{base_dir}/projections",
        date=date_str
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return True


def check_existing_files(date_str: str, include_betting: bool, include_projections: bool) -> bool:
    dfs_path = get_partitioned_path(base_dir="./data/inputs/dfs_salaries", date=date_str)

    if not dfs_path.exists():
        return False

    if include_betting:
        betting_path = get_partitioned_path(base_dir="./data/inputs/betting_odds", date=date_str)
        if not betting_path.exists():
            return False

    if include_projections:
        projections_path = get_partitioned_path(base_dir="./data/inputs/projections", date=date_str)
        if not projections_path.exists():
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Collect DFS salaries and optionally betting odds and projections'
    )
    parser.add_argument(
        '--start-date',
        required=True,
        help='Start date in YYYYMMDD format'
    )
    parser.add_argument(
        '--end-date',
        required=True,
        help='End date in YYYYMMDD format'
    )
    parser.add_argument(
        '--include-betting-odds',
        action='store_true',
        help='Also collect betting odds data'
    )
    parser.add_argument(
        '--include-projections',
        action='store_true',
        help='Also collect projection data'
    )
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Collect salaries, betting odds, and projections'
    )
    parser.add_argument(
        '--skip-no-games',
        action='store_true',
        default=True,
        help='Skip dates with no games (default: True)'
    )

    args = parser.parse_args()

    if args.include_all:
        args.include_betting_odds = True
        args.include_projections = True

    load_dotenv()
    api_key = os.getenv('TANK01_API_KEY')
    if not api_key:
        raise ValueError("TANK01_API_KEY not found in environment")

    client = Tank01Client(api_key=api_key)

    start_dt = datetime.strptime(args.start_date, '%Y%m%d')
    end_dt = datetime.strptime(args.end_date, '%Y%m%d')
    total_dates = (end_dt - start_dt).days + 1

    print("DFS Data Collection")
    print("=" * 80)
    print(f"Date range: {args.start_date} to {args.end_date} ({total_dates} days)")
    print(f"Collecting: salaries", end="")
    if args.include_betting_odds:
        print(", betting odds", end="")
    if args.include_projections:
        print(", projections", end="")
    print()
    print(f"API requests remaining: {client.get_remaining_requests()}")
    print("=" * 80)

    dates_processed = 0
    dates_skipped = 0
    dates_no_games = 0

    date_range = [start_dt + timedelta(days=x) for x in range(total_dates)]

    for current_date in tqdm(date_range, desc="Collecting DFS data"):
        date_str = current_date.strftime('%Y%m%d')

        if check_existing_files(date_str, args.include_betting_odds, args.include_projections):
            dates_skipped += 1
            continue

        if args.skip_no_games:
            schedule_response = client.get_daily_schedule(date_str)
            games = schedule_response.get('body', [])

            if not games or len(games) == 0:
                dates_no_games += 1
                continue

        dfs_response = client.get_dfs_salaries(date=date_str)
        if dfs_response.get('statusCode') != 200:
            continue

        saved_salaries = save_dfs_salaries(dfs_response, date_str)

        if args.include_betting_odds:
            betting_response = client.get_betting_odds(game_date=date_str)
            saved_betting = save_betting_odds(betting_response, date_str)

        if args.include_projections:
            projections_response = client.get_projections(num_of_days=7)
            saved_projections = save_projections(projections_response, date_str)

        dates_processed += 1

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Dates processed:     {dates_processed}")
    print(f"Dates skipped:       {dates_skipped} (already existed)")
    print(f"Dates no games:      {dates_no_games}")
    print(f"API requests used:   {client.request_count}")
    print(f"Remaining requests:  {client.get_remaining_requests()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
