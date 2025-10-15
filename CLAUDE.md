# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA DFS machine learning pipeline for DraftKings optimization. Per-player XGBoost models with YAML-configured features targeting 30% MAPE on elite player projections.

## Architecture

Five-layer structure (All layers implemented):

1. Data layer: Tank01 API, Parquet storage (date-partitioned), historical loaders with temporal validation, caching
2. Feature layer: YAML-configured pipelines, rolling stats, EWMA, 147 features from 21 box score statistics
3. Model layer: Per-player XGBoost, Bayesian hyperparameter optimization, model serialization, Random Forest baseline
4. Optimization layer: Linear programming via PuLP, DraftKings constraints (8 players, $50k cap)
5. Evaluation layer: Walk-forward backtesting, MAPE/RMSE/MAE/Correlation metrics, salary tier analysis

Registry pattern across layers for component hot-swapping. Configuration-driven design for reproducibility.

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
python scripts/collect_games.py --start-date 20241201 --end-date 20241231
python scripts/collect_dfs_salaries.py --start-date 20241201 --end-date 20241231
```

### Backtesting
```bash
python scripts/run_backtest.py --test-start 20250205 --test-end 20250206
python scripts/run_backtest.py --test-start 20250201 --test-end 20250228 --per-player
```

Requires TANK01_API_KEY in .env file. RapidAPI key from Tank01 Fantasy Stats API.

See [docs/SCRIPTS_GUIDE.md](docs/SCRIPTS_GUIDE.md) for complete scripts documentation.
See [scripts/README.md](scripts/README.md) for additional details.

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

LocalDataClient (src/data/collectors/local_data_client.py):
- Local data access without API calls
- Used for backtesting and historical analysis

Cache (src/data/collectors/cache.py):
- Caches API responses to reduce rate limit usage
- Automatic cache invalidation

Rate limit: 1000 requests/month (free tier). Client tracks usage via request_count and get_remaining_requests().

Endpoints defined in [src/data/collectors/endpoints.py](src/data/collectors/endpoints.py).

Date format: YYYYMMDD
Game ID format: awayTeamAbv@homeTeamAbv_YYYYMMDD

### Storage: src/data/storage/

BaseStorage abstract interface in [src/data/storage/base.py](src/data/storage/base.py).

ParquetStorage implementation (src/data/storage/parquet_storage.py):
- Stores data as Parquet in ./data/inputs/ subdirectories
- Methods: save(data_type, data, metadata), load(data_type, filters)
- Efficient columnar storage with compression
- Supports date range filtering

Versioning (src/data/storage/versioning.py):
- Dataset version control
- Track schema changes and data lineage

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

### Data Loaders: src/data/loaders/

HistoricalDataLoader (src/data/loaders/historical_loader.py):
- load_slate_data(date): Load all data for a specific slate
- load_historical_data(start_date, end_date): Load data across date range
- load_historical_player_logs(end_date, lookback_days): Load player logs with temporal validation
- load_slate_dates(start_date, end_date): Get dates with games in range
- Prevents lookahead bias in training data

OptimizedHistoricalDataLoader (src/data/loaders/optimized_historical_loader.py):
- Drop-in replacement for HistoricalDataLoader with 2-5x speedup
- Parallel file reading with TensorFlow or PyTorch backends
- Automatic backend selection (loader_type='auto')
- Prefetching and caching for improved throughput
- Identical API to HistoricalDataLoader for backward compatibility

TensorFlowDataLoader (src/data/loaders/tensorflow_loader.py):
- Parallel file reading with tf.data.Dataset.interleave
- Prefetching with tf.data.AUTOTUNE
- GPU-ready tensor outputs with caching
- create_cached_dataset(): Create optimized dataset for model training

PyTorchDataLoader (src/data/loaders/pytorch_loader.py):
- Multi-worker DataLoader with ThreadPoolExecutor
- Pin memory for faster GPU transfer
- ParquetDataset and StreamingParquetDataset classes
- create_dataloader(): Create DataLoader with parallel workers

### Features: src/features/

FeatureTransformer base class (src/features/base.py):
- Abstract interface with fit(), transform(), fit_transform() methods
- State management with is_fitted property

FeatureRegistry (src/features/registry.py):
- register(name, transformer_class): Register transformer
- create(name, **kwargs): Instantiate transformer by name
- list_transformers(): List registered transformers

FeaturePipeline (src/features/pipeline.py):
- add(transformer): Add transformer to pipeline
- fit(data): Fit all transformers
- transform(data): Apply all transformers sequentially
- fit_transform(data): Fit and transform in one step

RollingStatsTransformer (src/features/transformers/rolling_stats.py):
- Calculate rolling averages for windows [3, 5, 10]
- Stats: 21 box score columns (pts, reb, ast, stl, blk, mins, fpts, usage, shooting%, etc)
- Generates rolling mean, std for each window
- Grouped by playerID, sorted by gameDate

EWMATransformer (src/features/transformers/ewma.py):
- Exponentially weighted moving averages
- Default span: 5 games
- Applied to all 21 statistics

FeatureConfig (src/utils/feature_config.py):
- Load feature configurations from YAML files
- Available configs: default_features.yaml (21 stats), base_features.yaml (6 stats)
- Build pipelines from configuration
- Supports configuration versioning

### Models: src/models/

BaseModel abstract class (src/models/base.py):
- train(X, y): Train model
- predict(X): Generate predictions
- save(path), load(path): Model serialization
- is_trained property

ModelRegistry (src/models/registry.py):
- register(name, model_class): Register model
- create(name, config): Instantiate model by name

XGBoostModel (src/models/xgboost_model.py):
- Gradient boosting regression
- Default config: max_depth=6, learning_rate=0.05, n_estimators=200
- get_feature_importance(): SHAP-compatible importance scores

RandomForestModel (src/models/random_forest_model.py):
- Ensemble of decision trees
- Baseline model for comparison

### Optimization: src/optimization/

BaseOptimizer (src/optimization/base.py):
- optimize(projections, num_lineups): Generate lineups
- validate_lineup(lineup): Check constraints

BaseConstraint (src/optimization/base.py):
- is_satisfied(lineup): Validate constraint

LinearProgramOptimizer (src/optimization/optimizers/linear_program.py):
- Uses PuLP for integer linear programming
- Maximizes projected points subject to salary cap
- salary_cap: Default $50,000 for DraftKings

DraftKings constraints (src/optimization/constraints/draftkings.py):
- Salary cap $50,000
- Exactly 8 players
- Position requirements: PG/SG/SF/PF/C/G/F/UTIL
- Min 2 teams, min 2 games

OptimizerRegistry (src/optimization/registry.py):
- register(name, optimizer_class): Register optimizer
- create(name, **kwargs): Instantiate optimizer

### Evaluation: src/evaluation/

WalkForwardBacktest (src/walk_forward_backtest.py):
- run(): Execute walk-forward backtest across date range
- Automated model recalibration every N days
- Per-player or per-slate model support
- Benchmark comparison with statistical testing
- Model and prediction persistence
- Returns aggregated results with daily breakdown

Validator (src/evaluation/backtest/validator.py):
- validate(train_data, test_data): Single validation iteration
- walk_forward_validate(data_splits): Walk-forward across splits
- Returns dict of metric_name: value

Benchmarks (src/evaluation/benchmarks/):
- SeasonAverageBenchmark: Baseline using player season averages
- compare_with_model(): Head-to-head comparison with statistical tests
- compare_by_salary_tier(): Performance breakdown by salary bins

Metrics (src/evaluation/metrics/accuracy.py):
- MAPEMetric: Mean Absolute Percentage Error (target <30%)
- RMSEMetric: Root Mean Squared Error
- MAEMetric: Mean Absolute Error
- CorrelationMetric: Pearson correlation coefficient

MetricRegistry (src/evaluation/metrics/registry.py):
- register(name, metric_class): Register metric
- create(name): Instantiate metric

### Utilities: src/utils/

Logging (src/utils/logging.py):
- Centralized logging configuration
- File and console handlers
- Log level management

ConfigLoader (src/utils/config_loader.py):
- Load YAML configuration files
- Merge configs from multiple sources
- Environment variable substitution

IO utilities (src/utils/io.py):
- File I/O helpers
- Path management
- Data serialization

Paths (src/config/paths.py):
- Centralized path definitions
- Data directories
- Output directories


## Implementation Status

All five layers implemented with working end-to-end pipeline. Walk-forward backtesting framework operational with benchmark comparison.

Current notebooks:
- backtest_1d_by_player.ipynb: Per-player model training for single date
- backtest_1d_by_slate.ipynb: Slate-level model baseline
- backtest_season.ipynb: Season-long walk-forward validation using WalkForwardBacktest
- benchmark_comparison.ipynb: Model vs benchmark statistical comparison

Walk-forward backtesting features:
- WalkForwardBacktest framework (src/walk_forward_backtest.py)
- Automated model recalibration (default: 7 days)
- SeasonAverageBenchmark baseline comparison
- Statistical significance testing (paired t-test, Cohen's d)
- Salary tier performance analysis
- Model and prediction persistence with metadata
- Training inputs saved for reproducibility

Performance benchmarks (2025-02-05):
- Elite players ($8k+): 32.9% MAPE (near 30% target)
- Overall: 81.18% MAPE, 0.728 correlation
- Coverage: 96.4% of players with models
- Issues: Low-output player MAPE inflation, missing contextual features

Active development: Injury filtering, contextual features, multi-slate statistical validation

## Configuration

### API Key

API key required from RapidAPI Tank01 Fantasy Stats:
- Sign up at rapidapi.com
- Subscribe to Tank01 Fantasy Stats API
- Copy key to .env as TANK01_API_KEY

### Configuration Files

YAML configuration files in config/ directory:

config/data.yaml:
- Data source configuration
- Storage paths
- Date ranges

config/features/*.yaml:
- default_features.yaml: Full feature set (21 statistics, 147 features)
- base_features.yaml: Minimal set (6 core statistics for fast iteration)
- Pipeline definitions with rolling windows and EWMA transformers
- Feature statistics and hyperparameters

config/models/*.yaml:
- Model hyperparameters
- XGBoost, Random Forest configs
- Training parameters

config/experiments/*.yaml:
- Backtest configurations
- Walk-forward validation settings
- Experiment tracking

Example model config (config/models/xgboost_default.yaml):
```yaml
model_type: xgboost
hyperparameters:
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 200
  min_child_weight: 5
  subsample: 0.8
  colsample_bytree: 0.8
  objective: reg:squarederror
  random_state: 42
```

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

### Adding New Data Endpoints

1. Add endpoint to src/data/collectors/endpoints.py
2. Implement method in Tank01Client (src/data/collectors/tank01_client.py)
3. Add support in ParquetStorage (src/data/storage/parquet_storage.py)
4. Write unit tests in tests/data/collectors/
5. Run pytest

### Building Historical Datasets

1. Collect game data:
   python scripts/collect_games.py --start-date YYYYMMDD --end-date YYYYMMDD
2. Collect DFS salaries:
   python scripts/collect_dfs_salaries.py --start-date YYYYMMDD --end-date YYYYMMDD
3. Monitor API usage via client.get_remaining_requests()
4. Verify data in ./data/inputs/ subdirectories
5. Optional: Load to SQLite: python scripts/load_games_to_db.py

### Adding New Features

1. Create transformer class extending FeatureTransformer in src/features/transformers/
2. Implement fit() and transform() methods
3. Register transformer in FeatureRegistry
4. Add to FeaturePipeline in config/features/*.yaml
5. Write tests in tests/features/
6. Run pytest

### Adding New Models

1. Create model class extending BaseModel in src/models/
2. Implement train(), predict(), save(), load() methods
3. Register model in ModelRegistry
4. Add configuration in config/models/*.yaml
5. Write tests in tests/models/
6. Run pytest

### Running Backtests

Current notebooks:
1. backtest_1d_by_player.ipynb: Single-day per-player model backtest
2. backtest_1d_by_slate.ipynb: Single-day slate-level baseline
3. backtest_season.ipynb: Season-long walk-forward validation

Workflow:
1. Load historical data via HistoricalDataLoader (temporal validation)
2. Load feature config: feature_config = load_feature_config('default_features')
3. Build features: pipeline = feature_config.build_pipeline(FeaturePipeline)
4. Train per-player XGBoost models (500+ models per slate)
5. Generate projections
6. Calculate metrics (MAPE, RMSE, MAE, Correlation)
7. Analyze errors by salary tier
8. Optional: Optimize lineups with LinearProgramOptimizer

## Usage Examples

### Feature Pipeline (Configuration-Driven)

```python
from src.utils.feature_config import load_feature_config
from src.features.pipeline import FeaturePipeline

feature_config = load_feature_config('default_features')
pipeline = feature_config.build_pipeline(FeaturePipeline)

features = pipeline.fit_transform(training_data)
test_features = pipeline.transform(test_data)
```

### Feature Pipeline (Manual)

```python
from src.features.pipeline import FeaturePipeline
from src.features.transformers.rolling_stats import RollingStatsTransformer
from src.features.transformers.ewma import EWMATransformer

pipeline = FeaturePipeline()
pipeline.add(RollingStatsTransformer(windows=[3, 5, 10], stats=['pts', 'reb', 'ast'], include_std=True))
pipeline.add(EWMATransformer(span=5, stats=['pts', 'reb', 'ast']))

features = pipeline.fit_transform(training_data)
test_features = pipeline.transform(test_data)
```

### Model Training

```python
from src.models.xgboost_model import XGBoostModel

config = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200
}

model = XGBoostModel(config)
model.train(X_train, y_train)
predictions = model.predict(X_test)

model.save('models/xgboost_player_123.pkl')
```

### Lineup Optimization

```python
from src.optimization.optimizers.linear_program import LinearProgramOptimizer
from src.optimization.constraints.draftkings import DraftKingsConstraints

constraints = DraftKingsConstraints()
optimizer = LinearProgramOptimizer(
    constraints=[constraints],
    salary_cap=50000
)

lineups = optimizer.optimize(projections_df, num_lineups=1)
```

### Walk-Forward Validation

```python
from src.walk_forward_backtest import WalkForwardBacktest

backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20241001',
    train_end='20241130',
    test_start='20241201',
    test_end='20241215',
    per_player_models=True,
    model_type='xgboost',
    feature_config='default_features',
    recalibrate_days=7,
    save_models=True,
    save_predictions=True
)

results = backtest.run()
```

### Data Loading

```python
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.data.storage.parquet_storage import ParquetStorage

storage = ParquetStorage()
loader = HistoricalDataLoader(storage)

slate_data = loader.load_slate_data('20241215')
historical_data = loader.load_historical_data('20241201', '20241231')
player_logs = loader.load_historical_player_logs('20241215', lookback_days=365)
```
