import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from src.data.collectors.tank01_client import Tank01Client

load_dotenv()

def collect_games(client, start_date, end_date, output_dir):
    """
    Collect all games from start_date to end_date and save as parquet in partitioned structure.

    Output format: data/games/YYYY/MM/DD.parquet
    Each parquet contains a DataFrame with all games for that date.

    Args:
        client: Tank01Client instance
        start_date: Start date string (YYYYMMDD)
        end_date: End date string (YYYYMMDD)
        output_dir: Base directory for saving (e.g., 'data/games')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')

    current_date = start_dt
    total_dates = (end_dt - start_dt).days + 1
    games_collected = 0
    dates_with_games = 0
    dates_without_games = 0

    print(f"Collecting NBA games from {start_date} to {end_date}")
    print(f"Total dates to check: {total_dates}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Format: Parquet DataFrames")
    print("=" * 80)

    date_range = [start_dt + timedelta(days=x) for x in range(total_dates)]

    for current_date in tqdm(date_range, desc="Collecting games"):
        date_str = current_date.strftime('%Y%m%d')

        try:
            response = client.get_daily_schedule(date_str)

            if response.get('statusCode') == 200:
                games = response.get('body', [])

                if games:
                    year = date_str[:4]
                    month = date_str[4:6]
                    day = date_str[6:8]

                    date_dir = output_path / year / month
                    date_dir.mkdir(parents=True, exist_ok=True)

                    df = pd.DataFrame(games)

                    df['gameDate'] = date_str

                    file_path = date_dir / f"{day}.parquet"
                    df.to_parquet(file_path, index=False)

                    games_collected += len(games)
                    dates_with_games += 1
                else:
                    dates_without_games += 1

        except Exception as e:
            pass

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total dates checked:     {total_dates}")
    print(f"Dates with games:        {dates_with_games}")
    print(f"Dates without games:     {dates_without_games}")
    print(f"Total games collected:   {games_collected}")
    print(f"API requests made:       {client.request_count}")
    print(f"Remaining requests:      {client.get_remaining_requests()}")
    print(f"Output directory:        {output_path.absolute()}")
    print("=" * 80)


def main():
    api_key = os.getenv('TANK01_API_KEY')

    if not api_key:
        print("ERROR: TANK01_API_KEY not found in .env file")
        return 1

    client = Tank01Client(api_key=api_key)

    print("NBA Games Collection Script (Parquet Format)")
    print("=" * 80)
    print(f"API Key: {api_key[:10]}...")
    print(f"Rate limit: {client.rate_limit} requests/month")
    print(f"Current usage: {client.request_count}")
    print("=" * 80)

    print("\nCollecting 2024-2025 NBA Season Games")
    print("-" * 80)

    season_2024_start = '20241022'
    season_2024_end = '20241231'

    season_2025_start = '20250101'
    season_2025_end = '20250620'

    print(f"\nPhase 1: October - December 2024 ({season_2024_start} to {season_2024_end})")
    collect_games(
        client=client,
        start_date=season_2024_start,
        end_date=season_2024_end,
        output_dir='data/games'
    )

    if client.get_remaining_requests() < 200:
        print("\nWARNING: Low API requests remaining. Stopping before Phase 2.")
        print(f"Remaining requests: {client.get_remaining_requests()}")
        return 0

    print(f"\n\nPhase 2: January - June 2025 ({season_2025_start} to {season_2025_end})")
    collect_games(
        client=client,
        start_date=season_2025_start,
        end_date=season_2025_end,
        output_dir='data/games'
    )

    print("\n" + "=" * 80)
    print("ALL PHASES COMPLETE")
    print("=" * 80)
    print(f"Total API requests used: {client.request_count}")
    print(f"Remaining requests: {client.get_remaining_requests()}")
    print("\nData structure:")
    print("  data/games/YYYY/MM/DD.parquet")
    print("  Each parquet file contains a DataFrame of all games for that day")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
