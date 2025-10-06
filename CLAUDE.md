# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA DFS machine learning pipeline for DraftKings optimization. Modular design with pluggable models, features, and optimizers targeting 30% MAPE on player projections.

## Architecture

Five-layer structure (Phase 1 complete, Phases 2-5 planned):

1. Data layer (COMPLETE): Tank01 API collection, validation, Parquet storage
2. Feature layer (PLANNED): Rolling averages, EWMA, DvP, Vegas lines
3. Model layer (PLANNED): XGBoost, Random Forest, ensemble meta-learner
4. Optimization layer (PLANNED): Linear programming for cash games, genetic algorithms for GPP
5. Evaluation layer (PLANNED): Walk-forward validation, backtesting, MAPE metrics

Registry pattern planned for swapping models, features, and optimizers via configuration.

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Testing
```bash
pytest tests/
pytest tests/data/test_tank01_client.py
pytest -v tests/data/
```

### Data Collection
```bash
python scripts/build_historical_game_logs.py
```

Requires TANK01_API_KEY in .env file. RapidAPI key from Tank01 Fantasy Stats API.

Edit script to configure date range before running.

## Key Modules

### Data Collection: src/data/collectors/

Tank01Client wraps RapidAPI endpoints (src/data/collectors/tank01_client.py):
- get_dfs_salaries(date, lineup_type='DraftKings'): DFS salaries by date and platform
- get_betting_odds(date): Vegas lines, spreads, totals
- get_projections(date, lineup_type='DraftKings'): Fantasy projections
- get_schedule(date): Daily NBA schedule with game IDs
- get_injuries(): Current injury reports
- get_teams(): Team metadata (abbrev, teamID, city, name)
- get_box_score(game_id): Game box scores (requires gameID format)

Rate limit: 1000 requests/month (free tier). Client tracks usage via request_count and get_remaining_requests().

Endpoints defined in [src/data/collectors/endpoints.py](src/data/collectors/endpoints.py).

Date format: YYYYMMDD
Game ID format: awayTeamAbv@homeTeamAbv_YYYYMMDD

### Storage: src/data/storage/

BaseStorage abstract interface in [src/data/storage/base.py](src/data/storage/base.py).

CSVStorage implementation (src/data/storage/csv_storage.py):
- Stores data as Parquet or CSV in ./data/inputs/ subdirectories
- Methods: save_betting_odds, save_dfs_salaries, save_projections, save_schedule, save_injuries, save_teams, save_box_score
- load_data(data_type, start_date=None, end_date=None): Load and filter by date range
- use_parquet=True by default for efficiency

Storage structure:
```
data/
└── inputs/
    ├── betting_odds/       # betting_odds_YYYYMMDD.parquet
    ├── dfs_salaries/       # dfs_salaries_YYYYMMDD.parquet
    ├── projections/        # projections_YYYYMMDD.parquet
    ├── schedule/           # schedule_YYYYMMDD.parquet
    ├── injuries/           # injuries_YYYYMMDD.parquet
    ├── teams/              # teams.parquet
    └── box_scores/         # box_scores_GAMEID.parquet
```


## Implementation Status

Phase 1 complete: Data collection via Tank01 API, CSV/Parquet storage.

Phases 2-5 planned: Feature pipeline, ML models, optimization, evaluation framework.

## Configuration

API key required from RapidAPI Tank01 Fantasy Stats:
- Sign up at rapidapi.com
- Subscribe to Tank01 Fantasy Stats API
- Copy key to .env as TANK01_API_KEY

## Testing Strategy

Unit tests in tests/ mirror src/ structure.
Test fixtures mock API responses.
Tests validate rate limiting, retries, error handling.

Run tests:
```bash
pytest tests/ -v
```

## Data Patterns

All Tank01 responses follow structure:
```python
{
    'statusCode': 200,
    'body': {...}  # Actual data
}
```

Storage flattens body to DataFrame for analysis.

## Development Workflow

### Adding New Endpoints

1. Add endpoint to src/data/collectors/endpoints.py
2. Implement method in Tank01Client (src/data/collectors/tank01_client.py)
3. Add storage method to CSVStorage (src/data/storage/csv_storage.py)
4. Write unit tests in tests/data/
5. Run pytest

### Building Historical Datasets

1. Edit scripts/build_historical_game_logs.py to set date range
2. Run script: python scripts/build_historical_game_logs.py
3. Monitor API usage via client.get_remaining_requests()
4. Verify data in ./data/inputs/ subdirectories

### Next Development: Phase 2 (Feature Pipeline)

Create feature engineering layer:
1. Implement src/features/base.py (FeatureBase abstract class)
2. Create src/features/registry.py (FeatureRegistry for pluggable features)
3. Add feature implementations (box_score.py, advanced.py, opponent.py, vegas.py)
4. Build FeaturePipeline for composable transformations
5. Write tests for each feature calculator
