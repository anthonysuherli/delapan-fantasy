import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from src.data import Tank01Client, CSVStorage

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_date_range(start_date: str, end_date: str):
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    current = start
    while current <= end:
        yield current.strftime('%Y%m%d')
        current += timedelta(days=1)


def collect_schedules(
    client: Tank01Client,
    storage: CSVStorage,
    start_date: str,
    end_date: str
):
    logger.info(f"Collecting schedules from {start_date} to {end_date}")

    dates = list(generate_date_range(start_date, end_date))
    total_dates = len(dates)
    schedules_collected = 0
    errors = []

    for idx, game_date in enumerate(dates, 1):
        try:
            logger.info(f"[{idx}/{total_dates}] Fetching schedule for {game_date}")

            schedule = client.get_schedule(game_date=game_date)
            schedule_body = schedule.get('body', {})

            if not schedule_body:
                logger.warning(f"No games found for {game_date}")
                continue

            games = []
            if isinstance(schedule_body, dict):
                games = list(schedule_body.values())
            elif isinstance(schedule_body, list):
                games = schedule_body

            storage.save_schedule(schedule, game_date=game_date)
            schedules_collected += 1
            logger.info(f"Saved schedule for {game_date} with {len(games)} games")

        except Exception as e:
            error_msg = f"Failed to collect schedule for {game_date}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

    logger.info(f"Schedule collection complete: {schedules_collected}/{total_dates} dates")
    logger.info(f"Remaining requests: {client.get_remaining_requests()}/{client.rate_limit}")

    return errors


def collect_box_scores(
    client: Tank01Client,
    storage: CSVStorage,
    start_date: str,
    end_date: str
):
    logger.info(f"Collecting box scores from {start_date} to {end_date}")

    dates = list(generate_date_range(start_date, end_date))
    total_dates = len(dates)
    total_games = 0
    errors = []

    for idx, game_date in enumerate(dates, 1):
        try:
            logger.info(f"[{idx}/{total_dates}] Processing box scores for {game_date}")

            schedule_data = storage.load_data('schedule', start_date=game_date, end_date=game_date)

            if schedule_data.empty:
                logger.warning(f"No schedule data found for {game_date}, skipping")
                continue

            game_ids = schedule_data['gameID'].unique()
            logger.info(f"Found {len(game_ids)} games for {game_date}")

            for game_id in game_ids:
                try:
                    logger.debug(f"Fetching box score for {game_id}")
                    box_score = client.get_box_score(game_id=game_id)
                    storage.save_box_scores(box_score, game_id=game_id)
                    total_games += 1
                    logger.info(f"Saved box score for {game_id}")

                except Exception as e:
                    error_msg = f"Failed to get box score for {game_id}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            if idx % 5 == 0:
                logger.info(f"Progress: {idx}/{total_dates} dates, {total_games} games collected")
                logger.info(f"Remaining requests: {client.get_remaining_requests()}/{client.rate_limit}")

        except Exception as e:
            error_msg = f"Failed to process box scores for {game_date}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

    logger.info(f"Box score collection complete: {total_games} games collected")
    logger.info(f"Requests used: {client.get_request_count()}")
    logger.info(f"Remaining requests: {client.get_remaining_requests()}/{client.rate_limit}")

    return errors


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build historical NBA game logs')
    parser.add_argument('--start-date', required=True, help='Start date in YYYYMMDD format')
    parser.add_argument('--end-date', required=True, help='End date in YYYYMMDD format')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    api_key = os.getenv('TANK01_API_KEY')
    if not api_key:
        logger.error("TANK01_API_KEY not found in .env file")
        logger.error("Please add your RapidAPI key to .env")
        return

    client = Tank01Client(api_key=api_key)
    storage = CSVStorage(base_path='./data', use_parquet=True)

    logger.info("NBA Historical Game Logs Builder")
    logger.info("=" * 80)
    logger.info(f"Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"Storage: Parquet format in ./data/")
    logger.info(f"API Rate Limit: {client.rate_limit} requests/month")

    if not args.yes:
        user_input = input(f"Proceed with collection? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("Collection cancelled")
            return

    logger.info("Step 1: Collecting schedules")
    schedule_errors = collect_schedules(
        client=client,
        storage=storage,
        start_date=args.start_date,
        end_date=args.end_date
    )

    logger.info("Step 2: Collecting box scores")
    box_score_errors = collect_box_scores(
        client=client,
        storage=storage,
        start_date=args.start_date,
        end_date=args.end_date
    )

    all_errors = schedule_errors + box_score_errors
    if all_errors:
        logger.warning(f"Total errors encountered: {len(all_errors)}")
        for error in all_errors[:10]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 10:
            logger.warning(f"  ... and {len(all_errors) - 10} more")

    logger.info("Collection complete")


if __name__ == '__main__':
    main()