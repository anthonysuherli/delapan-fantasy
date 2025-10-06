# Data Collection Scripts

Scripts for building historical NBA data datasets.

## build_historical_game_logs.py

Collects historical NBA game data including schedules and box scores for a specified date range.

### Usage

```bash
python scripts/build_historical_game_logs.py --start-date YYYYMMDD --end-date YYYYMMDD
```

### Arguments

Required:
- `--start-date`: First date to collect (format: YYYYMMDD)
- `--end-date`: Last date to collect (format: YYYYMMDD)

Optional:
- `--yes`, `-y`: Skip confirmation prompt

### Example

```bash
python scripts/build_historical_game_logs.py --start-date 20241201 --end-date 20241231
```

```bash
python scripts/build_historical_game_logs.py --start-date 20241201 --end-date 20241231 -y
```
### How It Works

Two-phase collection process:

Phase 1 - Schedule Collection:
1. Iterates through each date in range
2. Calls Tank01Client.get_schedule(game_date)
3. Saves schedule to ./data/inputs/schedule/schedule_YYYYMMDD.parquet
4. Logs progress and remaining API requests

Phase 2 - Box Score Collection:
1. Loads saved schedules from Phase 1
2. Extracts game IDs from each schedule
3. Calls Tank01Client.get_box_score(game_id) for each game
4. Saves box scores to ./data/inputs/box_scores/box_scores_GAMEID.parquet
5. Progress reports every 5 dates
6. Handles errors gracefully, continues on failure

### Data Collected

Schedule data (per date):
- gameID (awayTeam@homeTeam_YYYYMMDD)
- home/away team abbreviations
- gameTime, gameStatus
- seasonType (Regular Season, Playoffs)

Box score data (per game):
- Player stats (PTS, REB, AST, STL, BLK, TO, FG%, 3P%, FT%, MIN)
- Team stats (total points, FG%, rebounds, etc)
- Game metadata (final score, attendance, etc)

### Storage Format

- Format: Parquet (compressed, efficient for analytics)
- Location: ./data/inputs/schedule/ and ./data/inputs/box_scores/
- Filename patterns:
  - schedule_YYYYMMDD.parquet
  - box_scores_GAMEID.parquet
- Load via CSVStorage.load_data(data_type, start_date, end_date)

### Example Output

```
2024-12-15 10:30:00 - INFO - NBA Historical Game Logs Builder
2024-12-15 10:30:00 - INFO - ================================================================================
2024-12-15 10:30:00 - INFO - Date Range: 20241201 to 20241231
2024-12-15 10:30:00 - INFO - Storage: Parquet format in ./data/inputs/
2024-12-15 10:30:00 - INFO - API Rate Limit: 1000 requests/month
Proceed with collection? (y/n): y

2024-12-15 10:30:05 - INFO - Step 1: Collecting schedules
2024-12-15 10:30:05 - INFO - [1/31] Fetching schedule for 20241201
2024-12-15 10:30:06 - INFO - Saved schedule for 20241201 with 11 games
2024-12-15 10:30:06 - INFO - [2/31] Fetching schedule for 20241202
...
2024-12-15 10:35:00 - INFO - Schedule collection complete: 31/31 dates
2024-12-15 10:35:00 - INFO - Remaining requests: 969/1000

2024-12-15 10:35:00 - INFO - Step 2: Collecting box scores
2024-12-15 10:35:00 - INFO - [1/31] Processing box scores for 20241201
2024-12-15 10:35:00 - INFO - Found 11 games for 20241201
2024-12-15 10:35:01 - INFO - Saved box score for BOS@MIA_20241201
2024-12-15 10:35:02 - INFO - Saved box score for LAL@GSW_20241201
...
2024-12-15 10:35:15 - INFO - Progress: 5/31 dates, 55 games collected
2024-12-15 10:35:15 - INFO - Remaining requests: 914/1000
...
2024-12-15 10:45:00 - INFO - Box score collection complete: 341 games collected
2024-12-15 10:45:00 - INFO - Requests used: 372
2024-12-15 10:45:00 - INFO - Remaining requests: 628/1000
2024-12-15 10:45:00 - INFO - Collection complete
```

### Prerequisites

- TANK01_API_KEY in .env file
- RapidAPI subscription to Tank01 Fantasy Stats API
- Sufficient API rate limit for date range (1 request per date + 1 per game)

### API Usage Estimation

Formula:
```
Total Requests = Dates + (Dates * Games per Day)
```

Examples:
- 1 week (7 days): 7 + (7 * 11) = ~84 requests
- 1 month (30 days): 30 + (30 * 11) = ~360 requests
- 1 season (180 days): 180 + (180 * 11) = ~2,160 requests

Rate limits:
- Free tier: 1000 requests/month
- Recommendation: Start with 1-2 weeks for testing

### Error Handling

Script continues on errors:
- Missing schedules: Logs warning, skips to next date
- Failed box scores: Logs error, continues with next game
- Network errors: Logged but do not stop execution
- Error summary displayed at end

### Tips

1. Monitor API usage: Check get_remaining_requests() output
2. Start small: Test with 7-day range first
3. Schedule collection is fast (1 request per date)
4. Box scores are bulk of API usage (1 per game)
5. Use -y flag for automated/scheduled runs
6. Check ./data/inputs/ directories after completion
