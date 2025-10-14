import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from src.data.collectors.tank01_client import Tank01Client
from src.data.storage_utils import get_partitioned_path

load_dotenv()


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


def main():
    parser = argparse.ArgumentParser(description='Collect NBA betting odds data')
    parser.add_argument('--start-date', required=True, help='Start date in YYYYMMDD format')
    parser.add_argument('--end-date', required=True, help='End date in YYYYMMDD format')
    parser.add_argument('--output-dir', default='./data/inputs', help='Base output directory (default: ./data/inputs)')
    parser.add_argument('--skip-no-games', action='store_true', default=True, help='Skip dates with no games (default: True)')

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv('TANK01_API_KEY')
    if not api_key:
        raise ValueError("TANK01_API_KEY not found in environment")

    client = Tank01Client(api_key=api_key)

    start_dt = datetime.strptime(args.start_date, '%Y%m%d')
    end_dt = datetime.strptime(args.end_date, '%Y%m%d')
    total_dates = (end_dt - start_dt).days + 1

    print("Betting Odds Collection")
    print("=" * 80)
    print(f"Date range: {args.start_date} to {args.end_date} ({total_dates} days)")
    print(f"Output directory: {args.output_dir}")
    print(f"API requests remaining: {client.get_remaining_requests()}")
    print("=" * 80)

    dates_processed = 0
    dates_skipped = 0
    dates_no_games = 0

    date_range = [start_dt + timedelta(days=x) for x in range(total_dates)]

    for current_date in tqdm(date_range, desc="Collecting betting odds"):
        date_str = current_date.strftime('%Y%m%d')

        output_path = get_partitioned_path(base_dir=f"{args.output_dir}/betting_odds", date=date_str)
        if output_path.exists():
            dates_skipped += 1
            continue

        if args.skip_no_games:
            schedule_response = client.get_daily_schedule(date_str)
            games = schedule_response.get('body', [])

            if not games or len(games) == 0:
                dates_no_games += 1
                continue

        betting_response = client.get_betting_odds(game_date=date_str)
        if betting_response.get('statusCode') != 200:
            continue

        saved = save_betting_odds(betting_response, date_str, args.output_dir)
        if saved:
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
