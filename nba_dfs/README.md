# NBA DFS ML Pipeline

Modular machine learning system for NBA DFS optimization on DraftKings with per-player XGBoost models.

## Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[Tank01 API] --> B[Cache]
        B --> C[Parquet Storage]
        C --> D[Historical Loader]
    end

    subgraph "Feature Layer"
        D --> E[YAML Config]
        E --> F[Feature Pipeline]
        F --> G[Rolling Stats]
        F --> H[EWMA]
        G --> I[147 Features]
        H --> I
    end

    subgraph "Model Layer"
        I --> J[Per-Player XGBoost]
        J --> K[Model Registry]
        K --> L[Saved Models]
    end

    subgraph "Optimization Layer"
        J --> M[Projections]
        M --> N[Linear Programming]
        N --> O[DraftKings Constraints]
        O --> P[Optimal Lineups]
    end

    subgraph "Evaluation Layer"
        M --> Q[Walk-Forward Backtest]
        Q --> R[Metrics: MAPE/RMSE/MAE]
        Q --> S[Benchmark Comparison]
        R --> T[Results by Salary Tier]
    end
```

### Design Philosophy

- **Modular**: Swap models, features, optimizers via configuration
- **Simple**: Explicit over implicit, readable over clever
- **Pluggable**: Registry pattern for components
- **Testable**: Clean interfaces, walk-forward validation
- **Reproducible**: YAML configuration tracking

### Project Structure

```
delapan-fantasy/
├── src/
│   ├── data/                 # Data layer
│   │   ├── collectors/       # API integrations
│   │   │   ├── tank01_client.py      # Tank01 RapidAPI client
│   │   │   ├── endpoints.py          # API endpoint definions
│   │   │   ├── local_data_client.py  # Local data access
│   │   │   └── cache.py              # API response caching
│   │   ├── storage/          # Storage backends
│   │   │   ├── base.py               # Abstract storage interface
│   │   │   ├── parquet_storage.py    # Parquet implementation
│   │   │   └── versioning.py         # Dataset versioning
│   │   └── loaders/          # Data loaders
│   │       ├── base.py               # Loader interface
│   │       └── historical_loader.py  # Historical data loader
│   ├── features/             # Feature engineering
│   │   ├── base.py           # FeatureTransformer interface
│   │   ├── registry.py       # Feature plugin system
│   │   ├── pipeline.py       # Sequential transformation pipeline
│   │   └── transformers/     # Feature implementations
│   │       ├── rolling_stats.py      # Rolling averages
│   │       └── ewma.py               # Exponential weighted MA
│   ├── models/               # ML models
│   │   ├── base.py           # BaseModel interface
│   │   ├── registry.py       # Model registry
│   │   ├── xgboost_model.py  # XGBoost implementation
│   │   └── random_forest_model.py    # Random Forest
│   ├── optimization/         # Lineup generation
│   │   ├── base.py           # Optimizer & Constraint initterfaces
│   │   ├── registry.py       # Optimizer registry
│   │   ├── constraints/      # Constraint implementations
│   │   │   └── draftkings.py         # DK rules
│   │   └── optimizers/       # Optimizer implementations
│   │       └── linear_program.py     # PuLP LP solver
│   ├── evaluation/           # Testing and validation
│   │   ├── backtest/         # Backtesting framework
│   │   │   └── validator.py          # Walk-forward validator
│   │   └── metrics/          # Performance metrics
│   │       ├── base.py               # Metric interface
│   │       ├── registry.py           # Metric registry
│   │       └── accuracy.py           # MAPE, RMSE, MAE, Correlation
│   ├── config/               # Configuration
│   │   └── paths.py          # Path management
│   └── utils/                # Utilities
│       ├── logging.py        # Logging configuration
│       ├── config_loader.py  # YAML config loader
│       ├── feature_config.py # Feature config loader
│       └── io.py             # I/O utilities
├── config/                   # Configuration files
│   ├── features/             # Feature configurations
│   │   ├── default_features.yaml    # Full feature set (21 stats)
│   │   └── base_features.yaml       # Minimal set (6 stats)
│   ├── models/               # Model configurations
│   └── experiments/          # Experiment configurations
├── scripts/                  # Data collection & processing scripts
│   ├── collect_games.py              # Collect schedules and box scores
│   ├── collect_dfs_salaries.py       # Collect DFS salaries
│   ├── load_games_to_db.py           # Load Parquet to SQLite
│   └── optimize_xgboost_hyperparameters.py  # Bayesian hyperparameter tuning
├── notebooks/                # Jupyter notebooks
│   ├── backtest_1d_by_player.ipynb   # Single-day per-player backtest
│   ├── backtest_1d_by_slate.ipynb    # Single-day slate-level backtest
│   ├── backtest_season.ipynb         # Season-long backtest
│   └── api_endpoint_exploration.ipynb
├── tests/                    # Unit tests
│   ├── data/                 # Data layer tests
│   │   ├── collectors/       # Collector tests
│   │   └── storage/          # Storage tests
│   └── features/             # Feature tests
└── requirements.txt
```

## Current Status

All five layers implemented with working end-to-end pipeline. Walk-forward backtesting framework operational with benchmark comparison.

### Data Layer
- Tank01 RapidAPI client with caching
- Parquet storage (date-partitioned)
- Historical data loader with temporal validation
- 3+ seasons of NBA data collected

### Feature Layer
- YAML-configured feature pipelines
- Rolling stats (3, 5, 10 game windows)
- EWMA transformers
- 147 features from 21 box score statistics

### Model Layer
- Per-player XGBoost models
- Bayesian hyperparameter optimization
- Model serialization with metadata
- Model recalibration logic (7-day default)
- 500+ player models trained per backtest
- Training inputs saved for reproducibility

### Optimization Layer
- Linear programming via PuLP
- DraftKings constraints (8 players, $50k salary cap)
- Multi-lineup generation

### Evaluation Layer
- Walk-forward backtesting framework (WalkForwardBacktest)
- Benchmark comparison (SeasonAverageBenchmark)
- Statistical significance testing (paired t-test, Cohen's d)
- MAPE, RMSE, MAE, Correlation metrics
- Error analysis by salary tier
- Feature importance tracking
- Predictions saved per slate with actuals

## Performance Benchmarks

### Walk-Forward Backtest (Multi-Slate)
- Framework: WalkForwardBacktest with recalibration every 7 days
- Benchmark: SeasonAverageBenchmark baseline comparison
- Statistical validation: Paired t-test, Cohen's d effect size
- Salary tier analysis: Performance breakdown by salary bins
- Model vs Benchmark: MAPE improvement tracking
- Coverage: 96%+ players per slate

### Single-Day Backtest (2025-02-05)
- Elite players ($8k+): 32.9% MAPE (target: 30%)
- High salary ($6-8k): 51.8% MAPE
- Mid salary ($4-6k): 76.8% MAPE
- Low salary ($0-4k): 103.6% MAPE
- Overall: 81.18% MAPE
- Correlation: 0.728
- Coverage: 96.4% (239/248 players)

Per-player models show strong correlation but struggle with low-output players. Elite tier meets target threshold. Walk-forward framework provides statistical validation against season average baseline.

## Key Design Patterns

### Configuration-Driven Features

YAML files define feature engineering pipelines:

```python
from src.utils.feature_config import load_feature_config

feature_config = load_feature_config('default_features')
pipeline = feature_config.build_pipeline(FeaturePipeline)
features = pipeline.fit_transform(training_data)
```

Configuration files in [config/features/](config/features/):
- default_features.yaml: 21 statistics, 147 features
- base_features.yaml: 6 core statistics for rapid experimentation

### Registry Pattern

Hot-swap components via registries:

```python
from src.models.registry import ModelRegistry
from src.features.registry import FeatureRegistry
from src.optimization.registry import OptimizerRegistry

model = ModelRegistry.create('xgboost', config)
feature = FeatureRegistry.create('rolling_stats', windows=[3,5,10])
optimizer = OptimizerRegistry.create('linear_program', constraints)
```

### Per-Player Training

Individual models capture player-specific patterns:

```python
from src.data.loaders.historical_loader import HistoricalDataLoader

loader = HistoricalDataLoader(storage)
for player_id in slate_players:
    player_data = loader.load_player_historical(player_id, lookback_days=365)
    model = XGBoostModel(config)
    model.train(player_data[features], player_data['fpts'])
    model.save(f'models/{date}/{player_name}_{player_id}.pkl')
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create .env file with Tank01 API key:

```bash
TANK01_API_KEY=your_rapidapi_key_here
```

Get API key from RapidAPI Tank01 Fantasy Stats subscription.

### Data Collection

Collect historical game data:

```bash
python scripts/collect_games.py --start-date 20241201 --end-date 20241231
```

Collect DFS salaries:

```bash
python scripts/collect_dfs_salaries.py --start-date 20241201 --end-date 20241231
```

See [scripts/README.md](scripts/README.md) for detailed documentation.

### Testing

```bash
pytest tests/
pytest tests/data/ -v
```

## API Rate Limits

Tank01 RapidAPI limits:
- 1000 requests/month (free tier)
- Client tracks usage via request_count
- Estimate: 1 request per date + 1 per game (11 games/day average = 12 requests/day)

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
pyarrow>=19.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
PuLP>=2.7.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

## Success Criteria

- **Model Performance**: ~30% MAPE on player projections
- **Optimization Speed**: Valid DK lineups in <1 second
- **Modularity**: Clean model/feature/optimizer swapping
- **Validation**: Walk-forward framework functional
- **Code Quality**: Unit tests, type hints, documentation

## Research Foundation

Based on academic research:

- **Papageorgiou et al. (2024)**: Individual player models, 28-30% MAPE
- **Hunter, Vielma & Zaman**: Linear programming optimization
- **Wang et al. (2024)**: XGBoost + SHAP interpretability

Key insights:

- Individual per-player models > aggregate approaches (+1.7-2.1%)
- Ensemble ML (XGBoost + RF) > single algorithms
- Linear programming optimal for single lineups
- Genetic algorithms for multi-lineup portfolios
- Fractional Kelly (1/3) for bankroll management

## Current Usage

### Data Collection Example

```python
from src.data.collectors.tank01_client import Tank01Client
from src.data.storage.csv_storage import CSVStorage

client = Tank01Client()
storage = CSVStorage()

date = '20241215'
salaries = client.get_dfs_salaries(date)
storage.save_dfs_salaries(salaries, date)

schedule = client.get_schedule(date)
storage.save_schedule(schedule, date)

odds = client.get_betting_odds(date)
storage.save_betting_odds(odds, date)
```

### Data Loading

```python
from src.data.storage.csv_storage import CSVStorage

storage = CSVStorage()

df = storage.load_data(
    'dfs_salaries',
    start_date='20241201',
    end_date='20241231'
)
```

### Historical Data Collection

```bash
python scripts/build_historical_game_logs.py
```

Edit script to configure date range before running.

## Notebooks

### backtest_1d_by_player.ipynb
Per-player model training and evaluation for single date. Demonstrates:
- Historical data loading with temporal validation
- YAML-configured feature pipeline
- Per-player XGBoost training
- Bayesian hyperparameter optimization
- Error analysis by salary tier

### backtest_1d_by_slate.ipynb
Slate-level model (single model for all players) for comparison baseline.

### backtest_season.ipynb
Season-long walk-forward backtesting across multiple dates using WalkForwardBacktest framework:
- Automated model recalibration
- Benchmark comparison with season average baseline
- Statistical significance testing
- Salary tier performance analysis
- Model and prediction persistence

### benchmark_comparison.ipynb
Comparative analysis between ML models and season average benchmark:
- Head-to-head performance comparison
- Statistical significance testing (paired t-test)
- Effect size calculation (Cohen's d)
- Salary tier breakdown

## Identified Issues and Roadmap

### Critical Issues
1. Low-output player MAPE inflation (103.6% for $0-4k tier)
2. Missing injury/inactive status filtering
3. No contextual features (home/away, rest days, matchups)
4. Variance prediction failure (hot streaks, cold streaks)

### Immediate Priority
1. Add injury/inactive filtering before prediction
2. Implement starter/bench role indicators
3. Add home/away and rest day features
4. Multi-day backtesting for validation

### Future Enhancements
- Opponent defensive rating features
- Minutes projection model
- Ensemble methods (XGBoost + Random Forest)
- Quantile regression for confidence intervals
- GPP optimizer with genetic algorithms
- Exposure management for multi-lineup generation

## License

MIT
