# Scripts

Scripts for data collection, backtesting, and hyperparameter optimization.

## run_backtest.py

Run walk-forward backtest with per-player or slate-level models, including benchmark comparison, statistical testing, and salary tier analysis.

### Usage

```bash
python scripts/run_backtest.py --test-start YYYYMMDD --test-end YYYYMMDD
```

### Arguments

Required:
- `--test-start`: Test start date (format: YYYYMMDD)
- `--test-end`: Test end date (format: YYYYMMDD)

Optional:
- `--db-path`: Path to SQLite database (default: nba_dfs.db)
- `--num-seasons`: Number of seasons for training (default: 1)
- `--model-type`: Model type - xgboost, random_forest, or linear (default: xgboost)
- `--feature-config`: Feature configuration name (default: default_features)
- `--per-player`: Use per-player models instead of slate-level
- `--min-player-games`: Minimum games for per-player models (default: 10)
- `--min-benchmark-games`: Minimum games for benchmark (default: 5)
- `--recalibrate-days`: Recalibrate model every N days (default: 7)
- `--output-dir`: Output directory (default: data/backtest_results)
- `--no-save-models`: Do not save trained models
- `--no-save-predictions`: Do not save predictions to parquet
- `--max-depth`: XGBoost max_depth (default: 6)
- `--learning-rate`: XGBoost learning_rate (default: 0.05)
- `--n-estimators`: XGBoost n_estimators (default: 200)
- `--verbose`: Enable verbose logging

### Examples

```bash
# Single day backtest
python scripts/run_backtest.py --test-start 20250205 --test-end 20250205

# One week with per-player models
python scripts/run_backtest.py --test-start 20250201 --test-end 20250207 --per-player

# Full month with custom features
python scripts/run_backtest.py --test-start 20250201 --test-end 20250228 --feature-config base_features

# Season backtest with model saving disabled
python scripts/run_backtest.py --test-start 20250101 --test-end 20250228 --no-save-models

# Verbose logging for debugging
python scripts/run_backtest.py --test-start 20250205 --test-end 20250206 --verbose
```

### What It Does

1. Loads historical training data based on num_seasons
2. Builds features using YAML-configured feature pipeline
3. Initializes season average benchmark for comparison
4. For each test date:
   - Trains models (per-player or slate-level)
   - Generates predictions with benchmark comparison
   - Saves models, predictions, and results (if enabled)
   - Evaluates against actual results
5. Aggregates results across all slates
6. Performs statistical significance testing (paired t-test)
7. Analyzes performance by salary tier
8. Exports results to CSV and summary to TXT

### Output Files

All files saved to `--output-dir` (default: data/backtest_results/):

**Daily Results:**
- `backtest_results_START_to_END.csv`: Per-slate metrics
- `summary_START_to_END.txt`: Overall summary with statistical tests
- `tier_comparison_START_to_END.csv`: Performance by salary tier

**Predictions (if save_predictions=True):**
- `data/models/per_slate/YYYYMMDD.parquet`: Predictions for each date
- `data/models/per_slate/YYYYMMDD_with_actuals.parquet`: Predictions with actual results

**Models (if save_models=True):**
- Per-player: `data/models/per_player/PLAYERID_NAME.pkl`
- Slate-level: `data/models/per_slate/MODEL_DATE.pkl`
- Metadata: Corresponding `.json` files with training info

### Metrics Reported

**Model Performance:**
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Correlation coefficient

**Benchmark Comparison:**
- Benchmark MAPE/RMSE
- MAPE improvement (model vs benchmark)
- Statistical significance (p-value, t-statistic)
- Effect size (Cohen's d)

**Salary Tier Analysis:**
- Performance breakdown by salary tier (Low/Mid/High/Elite)
- Model vs benchmark for each tier

### Training Period Calculation

Training period is automatically calculated based on test dates:
- `num_seasons=1`: Current season start to day before test start
- `num_seasons=2`: Previous season start to day before test start

Example:
```bash
--test-start 20250205 --num-seasons 1
# Training: 20241001 to 20250204 (current season)

--test-start 20250205 --num-seasons 2
# Training: 20231001 to 20250204 (includes previous season)
```

### Prerequisites

- SQLite database with historical data (see collect_games.py)
- Feature configuration YAML (see config/features/)
- Sufficient disk space for model saving (optional)

### Performance Considerations

- Per-player models: 500+ models per slate, slower but more accurate
- Slate-level models: Single model, faster, good baseline
- Model recalibration: Lower recalibrate-days = more accurate but slower
- Prediction saving: Adds I/O overhead but enables analysis

---

## collect_games.py

Collects historical NBA game data including schedules and box scores for a specified date range.

### Usage

```bash
python scripts/collect_games.py --start-date YYYYMMDD --end-date YYYYMMDD
```

### Arguments

Required:
- `--start-date`: First date to collect (format: YYYYMMDD)
- `--end-date`: Last date to collect (format: YYYYMMDD)

Optional:
- `--yes`, `-y`: Skip confirmation prompt

### Example

```bash
python scripts/collect_games.py --start-date 20241201 --end-date 20241231
```

```bash
python scripts/collect_games.py --start-date 20241201 --end-date 20241231 -y
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
- Load via ParquetStorage.load(data_type, filters)

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

---

## collect_dfs_salaries.py

Collects DFS salary data from Tank01 API for a specified date range.

### Usage

```bash
python scripts/collect_dfs_salaries.py --start-date YYYYMMDD --end-date YYYYMMDD
```

### Arguments

Required:
- `--start-date`: First date to collect (format: YYYYMMDD)
- `--end-date`: Last date to collect (format: YYYYMMDD)

Optional:
- `--platform`: DFS platform (default: DraftKings)
- `--yes`, `-y`: Skip confirmation prompt

### Example

```bash
python scripts/collect_dfs_salaries.py --start-date 20241201 --end-date 20241231
```

### Data Collected

DFS salary data (per date):
- playerID: Unique player identifier
- longName: Full player name
- team: Team abbreviation
- teamID: Team identifier
- pos: Primary position
- allValidPositions: All eligible positions
- salary: DFS salary for the platform
- platform: DFS platform (DraftKings, FanDuel, etc)

### Storage Format

- Format: Parquet
- Location: ./data/inputs/dfs_salaries/
- Filename pattern: dfs_salaries_YYYYMMDD.parquet

### Prerequisites

- TANK01_API_KEY in .env file
- RapidAPI subscription to Tank01 Fantasy Stats API

---

## load_games_to_db.py

Loads collected game data from Parquet files into SQLite database for legacy compatibility and SQL analysis.

### Usage

```bash
python scripts/load_games_to_db.py
```

### What It Does

1. Reads Parquet files from ./data/inputs/
2. Transforms data for database schema
3. Loads data into nba_dfs.db SQLite database
4. Creates tables: player_logs, games, dfs_salaries, injuries, seasons

### Database Tables

- player_logs: Player performance statistics per game
- games: NBA game schedule with matchup information
- dfs_salaries: DFS platform salaries by date and platform
- injuries: Player injury reports and statuses
- seasons: NBA season metadata with start and end dates

Note: Parquet files are the primary data source. SQLite database is for legacy compatibility and SQL-based analysis.

See [data/data_readme.md](../data/data_readme.md) for full database schema documentation.

---

## collect_depth_charts.py

Collects NBA team depth charts showing player rotation priorities for each position.

### Usage

```bash
# Collect all teams' depth charts
python scripts/collect_depth_charts.py --all-teams

# Collect specific team
python scripts/collect_depth_charts.py --team LAL

# Save to SQLite instead of Parquet
python scripts/collect_depth_charts.py --all-teams --storage sqlite

# Verbose output
python scripts/collect_depth_charts.py --all-teams --verbose
```

### Arguments

Optional:
- `--team`: Team abbreviation (e.g., LAL, BOS)
- `--all-teams`: Collect all teams (default if no team specified)
- `--storage`: Storage type - 'parquet' (default) or 'sqlite'
- `--db-path`: SQLite database path (default: nba_dfs.db)
- `--verbose`: Enable verbose output

### Example

```bash
# Collect all teams to Parquet
python scripts/collect_depth_charts.py --all-teams --verbose

# Collect Lakers depth chart to SQLite
python scripts/collect_depth_charts.py --team LAL --storage sqlite
```

### Data Collected

Depth chart data per team/position:
- team: Team abbreviation
- position: Player position (PG, SG, SF, PF, C)
- depth_order: Rotation priority (1 = starter, 2 = backup, etc.)
- playerID: Unique player identifier
- playerName: Full player name
- collection_date: Date of collection (YYYYMMDD)

### Storage Format

Parquet storage:
- Location: ./data/inputs/depth_charts/
- Filename pattern: depth_charts_YYYYMMDD.parquet

SQLite storage:
- Table: depth_charts
- Unique constraint on (team, position, depth_order, collection_date)

### Use Cases

Depth charts are valuable for:
- **Playing Time Projection**: Starters typically play more minutes
- **Injury Impact**: Identify who benefits when a player is out
- **Usage Rate Changes**: Role changes affect shot attempts
- **DFS Strategy**: Target value plays based on rotation changes
- **Coaching Patterns**: Track rotation tendencies

### Prerequisites

- TANK01_API_KEY in .env file
- RapidAPI subscription to Tank01 Fantasy Stats API

### API Usage

- 1 API request for all teams
- 1 API request per specific team

---

## optimize_xgboost_hyperparameters.py

Bayesian hyperparameter optimization for XGBoost models using Optuna.

### Usage

```bash
python scripts/optimize_xgboost_hyperparameters.py
```

### Arguments

Optional:
- `--feature-config NAME`: Feature configuration to use (default: default_features)
- `--per-player`: Run per-player optimization (default: slate-level)
- `--n-trials INT`: Number of optimization trials (default: 30)
- `--cv-folds INT`: Cross-validation folds (default: 3)

### Example

```bash
python scripts/optimize_xgboost_hyperparameters.py --feature-config default_features --n-trials 50
```

```bash
python scripts/optimize_xgboost_hyperparameters.py --per-player --feature-config base_features
```

### What It Does

Slate-level optimization:
1. Loads historical player logs
2. Builds features using specified config
3. Performs Bayesian optimization with Optuna
4. Outputs best hyperparameters
5. Saves results to optimization_results.json

Per-player optimization:
1. Selects sample of players for optimization
2. Runs optimization for each player
3. Aggregates best parameters across players
4. Outputs per-player and aggregated results

### Hyperparameters Optimized

- max_depth: Tree depth (3-10)
- learning_rate: Step size (0.01-0.3)
- n_estimators: Number of trees (100-500)
- subsample: Row sampling ratio (0.6-1.0)
- colsample_bytree: Column sampling ratio (0.6-1.0)
- min_child_weight: Minimum sum of instance weight (1-10)
- gamma: Minimum loss reduction (0-1)

### Output

Results saved to optimization_results.json:
- best_params: Optimal hyperparameters
- best_score: Best MAE achieved
- n_trials: Number of trials run
- feature_config: Configuration used
- timestamp: When optimization was run
