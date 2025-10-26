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

## Modular Design

All components use **registry patterns** for hot-swapping:

### Feature Sets (Config-Driven)
- `base_features.yaml`: 6 core stats + efficiency metrics (eFG%, TS%, FTR, AST/TO, GameScore)
- `default_features.yaml`: 21 statistics, 147 features with rolling windows
- Create custom feature sets in `config/features/` - automatically loaded

### Models (Registry Pattern)
Registered models in `src/models/registry.py`:
- `xgboost`: Gradient boosting (primary model)
- `random_forest`: Random forest baseline
- `stacking`: Multi-model ensemble with meta-learner
- `bagging`: Bootstrap aggregating for variance reduction
- `minutes`: Minutes projection model (filters low-minute players)
- `quantile`: Quantile regression for floor/median/ceiling predictions

**Adding new models:**
```python
# 1. Create model class extending BaseModel
class MyModel(BaseModel):
    def train(self, X, y): ...
    def predict(self, X): ...

# 2. Register in src/models/registry.py
from .my_model import MyModel
registry.register('my_model', MyModel)

# 3. Use via CLI
python scripts/predict_slate.py --model my_model_config
```

### Optimizers (Registry Pattern)
Registered in `src/optimization/registry.py`:
- `linear_program`: Integer linear programming via PuLP (cash game optimization)
- `gpp_genetic`: Genetic algorithm for GPP tournaments (ceiling optimization with ownership)
- `lineup_generator`: pydfs-lineup-optimizer with DraftKings constraints

### Ensemble Models

**Stacking** (`config/models/stacked_xgb_rf.yaml`):
- Combines XGBoost + Random Forest base models
- XGBoost meta-learner on base predictions
- 5-fold CV for out-of-fold training
- Usage: `--model stacked_xgb_rf`

**Bagging** (`config/models/bagged_xgboost.yaml`):
- 10 XGBoost models on bootstrap samples
- Variance reduction through averaging
- 80% data sampling per model
- Usage: `--model bagged_xgboost`

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

### Slate Prediction (Daily Workflow)

**Predict full slate with analysis:**
```bash
python scripts/predict_slate.py --date 20250210 --analyze
```

**Swap feature sets:**
```bash
python scripts/predict_slate.py --date 20250210 --features default_features  # 147 features
python scripts/predict_slate.py --date 20250210 --features base_features     # Minimal set
```

**Swap model types:**
```bash
python scripts/predict_slate.py --date 20250210 --model xgboost_default
python scripts/predict_slate.py --date 20250210 --model stacked_xgb_rf       # Ensemble
python scripts/predict_slate.py --date 20250210 --model bagged_xgboost       # Bagging
```

### Lineup Generation

```bash
# Generate from predictions
python scripts/generate_lineups.py --predictions predictions.csv --num-lineups 20

# Different strategies
python scripts/generate_lineups.py --predictions predictions.csv --strategy conservative
python scripts/generate_lineups.py --predictions predictions.csv --strategy aggressive
```

### Daily Workflow (Production)

**Step 1: Generate predictions for slate**
```bash
python scripts/predict_slate.py --date 20250210 --analyze
python scripts/predict_slate.py --date 20250210 --features default_features --model stacked_xgb_rf
```

**Step 2: Generate optimal lineups**
```bash
python scripts/generate_lineups.py --predictions predictions.csv --num-lineups 20
python scripts/generate_lineups.py --predictions predictions.csv --strategy aggressive
```

### GPP Tournament Workflow (Advanced)

**Step 1: Generate predictions with variance estimates (quantile regression)**
```bash
python scripts/predict_slate.py --date 20250210 --use-quantiles --features full_features --analyze
```

**Step 2: Generate tournament lineups with GPP genetic optimizer**
```bash
# Basic GPP lineups (uses salary-based ownership estimates)
python scripts/generate_lineups.py --predictions predictions.csv --use-gpp-genetic --num-lineups 20

# Advanced: Custom ownership projections
python scripts/generate_lineups.py --predictions predictions.csv \
  --use-gpp-genetic \
  --ownership-file ownership.csv \
  --num-lineups 20 \
  --ownership-weight 0.4 \
  --population-size 150 \
  --generations 100
```

**Parameters:**
- `--use-quantiles`: Use quantile regression for ceiling/floor predictions
- `--use-gpp-genetic`: Use genetic algorithm optimizer for tournaments
- `--ownership-file`: CSV with playerID and ownership columns (optional)
- `--ownership-weight`: Penalty for high ownership (0.0-1.0, default 0.3)
- `--population-size`: GA population size (default 100)
- `--generations`: Number of GA iterations (default 50)

### Walk-Forward Backtest Simulation

**Run simulation across historical dates:**
```bash
python scripts/run_walk_forward_backtest.py --start-date 20250201 --end-date 20250210
python scripts/run_walk_forward_backtest.py --start-date 20250201 --end-date 20250210 --num-lineups 20
```

**Customize configuration:**
```bash
python scripts/run_walk_forward_backtest.py --start-date 20250201 --end-date 20250210 \
  --features default_features \
  --model stacked_xgb_rf \
  --num-lineups 20 \
  --strategy aggressive \
  --save-predictions \
  --save-lineups
```

**Generate report from results:**
```bash
python -m src.evaluation.backtest_report --results data/backtest_results/simulation_results_TIMESTAMP.json
```

**Legacy backtesting (deprecated):**
```bash
# Old scripts moved to scripts/deprecated/
# Use notebooks 01_single_day_foundation.ipynb and 02_full_slate_prediction.ipynb for validation
```

Requires TANK01_API_KEY in .env file. RapidAPI key from Tank01 Fantasy Stats API.

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
- Uses DuckDB for efficient in-memory querying of parquet files
- Reads directly from date-partitioned parquet storage with hive partitioning
- load_slate_data(date): Load all data for a specific slate
- load_historical_data(start_date, end_date): Load data across date range
- load_historical_player_logs(start_date, end_date, num_seasons, player_ids): Load player logs with temporal validation
  - start_date: Optional training start date (overrides num_seasons)
  - end_date: Required exclusive end date
  - num_seasons: Number of seasons to load (default 2: current + previous)
  - player_ids: Optional list of player IDs to filter (optimizes memory usage)
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

EfficiencyMetricsTransformer (src/features/transformers/efficiency_metrics.py):
- eFG%: Effective Field Goal % (accounts for 3PT value)
- TS%: True Shooting % (accounts for FTs and 3PT)
- FTR: Free Throw Rate (FT attempts relative to FGA)
- TotalReb: Total rebounds (OffReb + DefReb)

PlaymakingMetricsTransformer (src/features/transformers/playmaking_metrics.py):
- AST_TO_ratio: Assists per turnover

ImpactMetricsTransformer (src/features/transformers/impact_metrics.py):
- GameScore: Hollinger's composite performance metric

ContextualFeaturesTransformer (src/features/transformers/contextual.py):
- home_game: 1 if playing at home, 0 if away (parsed from gameID)
- rest_days: Days since last game (capped at 7)
- back_to_back: 1 if playing on consecutive days
- days_off_3plus: 1 if 3+ days rest
- Config: config/features/contextual_features.yaml

OpponentStatsTransformer (src/features/transformers/opponent_stats.py):
- opp_pace: Opponent pace (possessions per game)
- opp_def_rating_last_10: Opponent defensive rating (recent 10 games)
- Position-specific points allowed: pg/sg/sf/pf/c_fpts_allowed
- is_home_team: Home court advantage indicator
- opp_3pt_defense_rank, opp_rest_days, opp_foul_rate, opp_turnover_rate
- Matchup-based features for opponent defensive strength

FeatureConfig (src/utils/feature_config.py):
- Load feature configurations from YAML files
- Available configs:
  - base_features.yaml: 6 core stats + efficiency metrics
  - default_features.yaml: 21 statistics, 147 features with rolling windows
  - contextual_features.yaml: Includes contextual transformers
  - full_features.yaml: Complete feature set with all transformers
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

BaggingModel (src/models/bagging_model.py):
- Bootstrap aggregating for variance reduction
- Trains multiple models on bootstrap samples
- Averages predictions for improved stability
- Config: config/models/bagged_xgboost.yaml

StackingModel (src/models/stacking_model.py):
- Multi-model ensemble with meta-learner
- Trains base models (XGBoost + Random Forest)
- Meta-learner combines base predictions
- 5-fold CV for out-of-fold training
- Config: config/models/stacked_xgb_rf.yaml

MinutesProjectionModel (src/models/minutes_model.py):
- Separate model for predicting player minutes
- Filters low-minute players before fantasy point prediction
- XGBoost regressor targeting minutes played
- predict_with_threshold(): Returns predictions and mask for players above threshold
- Config: config/models/minutes_projection.yaml
- Min minutes threshold: 15 (configurable)

QuantileRegressionModel (src/models/quantile_model.py):
- Predicts floor (10th), median (50th), ceiling (90th) percentiles
- Enables variance estimation and confidence intervals
- Used for GPP tournament optimization (high-ceiling strategy)
- predict_with_variance(): Returns floor/median/ceiling/variance/iqr/cv
- Three separate XGBoost models (one per quantile)
- Config: config/models/quantile_regression.yaml
- Usage: `python scripts/predict_slate.py --date 20250210 --use-quantiles`

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

GPPGeneticOptimizer (src/optimization/optimizers/gpp_genetic.py):
- Genetic algorithm optimizer for GPP tournaments
- Optimizes for ceiling projections (90th percentile) instead of expected value
- Ownership penalty to favor contrarian, low-ownership plays
- Population-based evolution (selection, crossover, mutation)
- Generates diverse, uncorrelated lineups for multi-entry tournaments
- Parameters:
  - population_size: Number of lineups per generation (default 100)
  - generations: Number of evolutionary iterations (default 50)
  - ownership_weight: Penalty for high ownership (default 0.3)
  - diversity_weight: Bonus for lineup variance (default 0.2)
- Usage: `python scripts/generate_lineups.py --predictions predictions.csv --use-gpp-genetic --num-lineups 20`
- Requires ceiling projections from quantile regression model

DraftKings constraints (src/optimization/constraints/draftkings.py):
- Salary cap $50,000
- Exactly 8 players
- Position requirements: PG/SG/SF/PF/C/G/F/UTIL
- Min 2 teams, min 2 games

OptimizerRegistry (src/optimization/registry.py):
- register(name, optimizer_class): Register optimizer
- create(name, **kwargs): Instantiate optimizer

### Evaluation: src/evaluation/

WalkForwardSimulation (src/evaluation/walk_forward_simulation.py):
- run(): Execute walk-forward simulation across date range
- Simulates production workflow: predict → optimize → score
- Per-player model training for each slate
- Lineup generation and scoring against actual results
- Aggregates metrics across entire backtest period
- Configurable features, models, and strategies
- Returns aggregated results with daily breakdown
- CLI: scripts/run_walk_forward_backtest.py

BacktestReport (src/evaluation/backtest_report.py):
- generate_summary(): Text summary of simulation results
- plot_daily_performance(): Daily actual vs projected points
- plot_score_distribution(): Histogram of lineup scores and errors
- plot_error_vs_projected(): Bias analysis (error vs projection magnitude)
- generate_full_report(): Complete report with all visualizations

WalkForwardBacktest (DEPRECATED - src/evaluation/deprecated/backtest/walk_forward.py):
- Legacy backtest framework moved to deprecated/
- Use WalkForwardSimulation for new projects

Trainers (src/evaluation/backtest/trainers/):
- PerPlayerTrainer: Train individual models per player
- PerSlateTrainer: Train single model for entire slate

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
- CappedMAPEMetric: MAPE with outlier capping
- SMAPEMetric: Symmetric MAPE
- WMAPEMetric: Weighted MAPE

Segmented Analysis (src/evaluation/metrics/segmentation.py):
- analyze_by_salary(results_df): Performance metrics by salary tier ($3-5k, $5-7k, $7-9k, $9k+)
- analyze_by_position(results_df): Performance metrics by position (PG, SG, SF, PF, C)
- cross_analysis(results_df): MAPE cross-tabulation by salary × position
- get_best_segments(results_df): Top/bottom performing salary-position combinations
- summary_report(results_df): Text summary of segmented performance
- Integrated into predict_slate.py via --analyze flag

MetricRegistry (src/evaluation/metrics/registry.py):
- register(name, metric_class): Register metric
- create(name): Instantiate metric

Deprecated (src/deprecated/ and src/evaluation/deprecated/):
- walk_forward_backtest.py: Legacy backtest framework (backward compatibility shim)
- src/evaluation/deprecated/backtest/: Old WalkForwardBacktest, PerPlayerTrainer, PerSlateTrainer
- scripts/deprecated/: Legacy run_backtest.py and run_backtest_with_lineups.py
- notebooks/deprecated/: Old backtest notebooks
- src/interface/deprecated/: Legacy Panel UI
- Use scripts/predict_slate.py + scripts/generate_lineups.py for production workflow

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

### Interfaces: src/interface/

Interfaces (DEPRECATED - moved to src/interface/deprecated/):
- Legacy Panel-based UI in src/interface/deprecated/panel_backtest_app.py
- Old Streamlit UI removed in previous cleanup
- Current workflow: Command-line scripts (predict_slate.py, generate_lineups.py)
- Notebook-based analysis: 01_single_day_foundation.ipynb, 02_full_slate_prediction.ipynb

## Implementation Status

All five layers implemented with working end-to-end pipeline. Walk-forward backtesting framework operational with benchmark comparison.

Current notebooks:
- 01_single_day_foundation.ipynb: Phase 1 - Single player prediction validation
- 02_full_slate_prediction.ipynb: Phase 2 - Full slate per-player model training with position/salary analysis
- notebooks/deprecated/: Legacy backtesting notebooks (run_backtest.ipynb, evaluate_backtest.ipynb, colab_walk_forward_backtest.ipynb)

Walk-forward backtesting:
- DEPRECATED: Legacy WalkForwardBacktest framework in src/evaluation/deprecated/
- IMPLEMENTED: New multi-day simulation framework (src/evaluation/walk_forward_simulation.py)
  - Wraps predict_slate.py + generate_lineups.py in date loop
  - Validates production workflow on historical data
  - Tracks lineup performance vs actual results
  - Generates comprehensive performance reports with visualizations
  - Command: `python scripts/run_walk_forward_backtest.py --start-date YYYYMMDD --end-date YYYYMMDD`

Performance benchmarks (2025-02-05):
- Elite players ($8k+): 32.9% MAPE (near 30% target)
- Overall: 81.18% MAPE, 0.728 correlation
- Coverage: 96.4% of players with models
- Issues: Low-output player MAPE inflation, missing contextual features

Recent improvements (2025-10-26):
- New efficiency metrics transformers: eFG%, TS%, FTR, GameScore, AST/TO ratio
- Modular slate prediction workflow: predict_slate.py + generate_lineups.py
- Position-based performance analysis with cross-tabulation
- Deprecated legacy backtesting infrastructure (moved to src/deprecated/)
- Cleaned codebase: Removed GPU optimizations, consolidated deprecated code
- Production-ready scripts for daily DFS workflow
- Walk-forward backtest simulation: End-to-end pipeline validation on historical data
  - Iterates through historical slates sequentially
  - Generates predictions → optimizes lineups → scores vs actuals per day
  - Aggregates performance metrics across entire backtest period
  - Comprehensive reports with visualizations (daily performance, error distribution, bias analysis)

Active development:
- Contextual features (home/away, rest days)
- SG position performance improvement (currently 161.5% MAPE)

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
- GPU configurations (xgboost_a100.yaml for GPU training)

Note: XGBoost 2.0+ uses `device: "cuda:0"` instead of deprecated `gpu_hist`/`gpu_id` parameters

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
  tree_method: hist
```

GPU config (config/models/xgboost_a100.yaml):
```yaml
model:
  type: xgboost
  params:
    max_depth: 10
    learning_rate: 0.05
    n_estimators: 500
    min_child_weight: 3
    subsample: 0.85
    colsample_bytree: 0.85
    objective: reg:squarederror
    random_state: 42
    tree_method: hist
    device: cuda:0
    max_bin: 512
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
4. Verify data in ./data/inputs/ subdirectories (stored as parquet files)

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

### Daily DFS Workflow

Production workflow (command-line):
1. Generate slate predictions: python scripts/predict_slate.py --date 20250210 --analyze
2. Optimize lineups: python scripts/generate_lineups.py --predictions predictions.csv --num-lineups 20
3. Upload lineups to DraftKings

Development workflow (notebooks):
1. Phase 1 validation: notebooks/01_single_day_foundation.ipynb (single player)
2. Phase 2 validation: notebooks/02_full_slate_prediction.ipynb (full slate with position/salary analysis)
3. Feature engineering and model tuning

Legacy notebooks (deprecated):
- notebooks/deprecated/run_backtest.ipynb: Old backtest execution
- notebooks/deprecated/evaluate_backtest.ipynb: Old results analysis
- notebooks/deprecated/colab_walk_forward_backtest.ipynb: Colab version

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

### Slate Prediction (Production Workflow)

```python
# Command-line usage (recommended)
# python scripts/predict_slate.py --date 20250205 --analyze --output predictions.csv

# Programmatic usage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.utils.feature_config import load_feature_config
from src.features.pipeline import FeaturePipeline
from src.models.registry import registry as model_registry

loader = HistoricalDataLoader('data')
feature_config = load_feature_config('base_features')
pipeline = feature_config.build_pipeline(FeaturePipeline)

# Load slate and historical data
test_slate = loader.load_slate_data('20250205')
historical_logs = loader.load_historical_player_logs(
    end_date='20250205',
    num_seasons=2,
    player_ids=eligible_player_ids
)

# Train per-player models and generate predictions
# See notebooks/02_full_slate_prediction.ipynb for complete example
```

Note: Legacy WalkForwardBacktest framework moved to src/deprecated/ and src/evaluation/deprecated/

### Data Loading

```python
from src.data.loaders.historical_loader import HistoricalDataLoader

# Initialize loader with data directory containing parquet files
loader = HistoricalDataLoader(data_dir='data')

slate_data = loader.load_slate_data('20241215')
historical_data = loader.load_historical_data('20241201', '20241231')

# Load current season data
player_logs = loader.load_historical_player_logs(
    end_date='20241215',
    num_seasons=1
)

# Load with player filtering (memory optimization)
filtered_logs = loader.load_historical_player_logs(
    end_date='20241215',
    num_seasons=2,
    player_ids=['2544', '201935', '203507']  # LeBron, Durant, Giannis
)
```

