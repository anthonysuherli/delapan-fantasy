import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors.tank01_client import Tank01Client
from src.data.storage_utils import get_partitioned_path

load_dotenv()

def save_dfs_salaries(data: dict, date_str: str, base_dir: str = "./data/inputs"):
    if "body" not in data:
        return

    df = pd.json_normalize(data["body"])
    output_path = get_partitioned_path(
        base_dir=f"{base_dir}/dfs_salaries",
        date=date_str
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

def save_betting_odds(data: dict, date_str: str, base_dir: str = "./data/inputs"):
    if "body" not in data:
        return

    df = pd.json_normalize(data["body"])
    output_path = get_partitioned_path(
        base_dir=f"{base_dir}/betting_odds",
        date=date_str
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

def save_projections(data: dict, date_str: str, base_dir: str = "./data/inputs"):
    if "body" not in data:
        return

    df = pd.json_normalize(data["body"])
    output_path = get_partitioned_path(
        base_dir=f"{base_dir}/projections",
        date=date_str
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

def get_game_count(client, date_str: str) -> int:
    schedule = client.get_daily_schedule(date_str)
    return schedule['body']


def fetch_dfs_salaries_for_seasons():
    api_key = os.getenv("TANK01_API_KEY")
    if not api_key:
        raise ValueError("TANK01_API_KEY not found in environment")

    client = Tank01Client(api_key=api_key)
    # Convert season_ranges into list of (start, end) date objects
    season_ranges = [
        # ("20221018", "20230410"),
        # ("20231024", "20240415"),
        ("20241022", "20250414")
    ]
    season_ranges = [
        (
            datetime.strptime(start, "%Y%m%d"),
            datetime.strptime(end, "%Y%m%d")
        )
        for start, end in season_ranges
    ]

    for season_start, season_end in season_ranges:
        print(f"Fetching DFS salaries from {season_start} to {season_end}")

        current_date = season_start
        end_date = season_end
        print(f"Season from {current_date.date()} to {end_date.date()}")
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")

            dfs_path = get_partitioned_path(base_dir="./data/inputs/dfs_salaries", date=date_str)
            betting_path = get_partitioned_path(base_dir="./data/inputs/betting_odds", date=date_str)
            projections_path = get_partitioned_path(base_dir="./data/inputs/projections", date=date_str)

            if dfs_path.exists() and betting_path.exists() and projections_path.exists():
                print(f"Files exist for {date_str}, skipping...")
                current_date += timedelta(days=1)
                continue

            games = pd.DataFrame(get_game_count(client, date_str))
            print(games.head(3))
            if len(games) == 0:
                print(f"No games on {date_str}, skipping...")
                current_date += timedelta(days=1)
                continue

            print(f"Date: {date_str}, Game Count: {len(games)}")

            if not dfs_path.exists():
                dfs_response = client.get_dfs_salaries(date=date_str)
                dfs_salaries = pd.json_normalize(dfs_response.get("body", []))
                print(f"Fetched DFS salaries for {date_str}, {dfs_salaries.shape[0]} rows")
                print(dfs_salaries.head(3))
                save_dfs_salaries(dfs_response, date_str)

            if not betting_path.exists():
                betting_response = client.get_betting_odds(game_date=date_str)
                betting_odds = pd.json_normalize(betting_response.get("body", []))
                print(f"Fetched betting odds for {date_str}, {betting_odds.shape[0]} rows")
                print(betting_odds.head(3))
                save_betting_odds(betting_response, date_str)

            if not projections_path.exists():
                projections_response = client.get_projections(num_of_days=7)
                projections = pd.json_normalize(projections_response.get("body", []))
                print(f"Fetched projections for {date_str}, {projections.shape[0]} rows")
                print(projections.head(3))
                save_projections(projections_response, date_str)

            current_date += timedelta(days=1)   
            
    print(f"Total requests made: {client.get_request_count()}")

if __name__ == "__main__":
    fetch_dfs_salaries_for_seasons()
