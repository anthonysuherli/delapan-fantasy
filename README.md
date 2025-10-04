# NBA DFS ML Pipeline

Modular machine learning system for NBA DFS optimization on DraftKings.

## Architecture

### Design Philosophy

- **Modular**: Swap models, features, optimizers without rewriting code
- **Simple**: Core functionality only, no over-engineering
- **Pluggable**: Registry pattern for components
- **Testable**: Clean interfaces, walk-forward validation

### Project Structure

```
delapan-fantasy/
├── src/
│   ├── data/                 # Data layer (Phase 1: COMPLETE)
│   │   ├── collectors/       # API integrations
│   │   │   ├── tank01_client.py      # Tank01 RapidAPI client
│   │   │   └── endpoints.py          # API endpoint definitions
│   │   └── storage/          # Storage backends
│   │       ├── base.py               # Abstract storage interface
│   │       └── csv_storage.py        # CSV/Parquet implementation
│   ├── features/             # Feature engineering (PLANNED)
│   │   ├── base.py           # Feature interface
│   │   ├── box_score.py      # Basic stats features
│   │   ├── advanced.py       # Rolling averages, EWMA
│   │   ├── opponent.py       # DvP, defensive ratings
│   │   ├── vegas.py          # Betting lines, O/U
│   │   └── registry.py       # Feature plugin system
│   ├── models/               # ML models (PLANNED)
│   │   ├── base.py           # Abstract model interface
│   │   ├── xgboost_model.py  # XGBoost implementation
│   │   ├── random_forest.py  # Random Forest
│   │   ├── linear.py         # Ridge/Lasso baseline
│   │   └── ensemble.py       # Stacking meta-learner
│   ├── optimization/         # Lineup generation (PLANNED)
│   │   ├── base.py           # Optimizer interface
│   │   ├── linear_prog.py    # PuLP LP solver (cash)
│   │   ├── genetic.py        # Genetic algorithm (GPP)
│   │   ├── constraints.py    # DK rules, exposure limits
│   │   └── stacking.py       # Correlation/stacking logic
│   ├── projections/          # Projection engine (PLANNED)
│   │   └── generator.py      # Model to projections pipeline
│   ├── evaluation/           # Testing and validation (PLANNED)
│   │   ├── backtest.py       # Historical slate testing
│   │   ├── metrics.py        # MAPE, RMSE, ROI
│   │   └── monte_carlo.py    # Simulation framework
│   └── utils/                # Utilities (PLANNED)
│       ├── bankroll.py       # Kelly criterion
│       └── logger.py         # Logging
├── scripts/                  # Data collection scripts
│   └── build_historical_game_logs.py
├── tests/                    # Unit tests
│   ├── data/                 # Data layer tests
│   │   ├── test_tank01_client.py
│   │   ├── test_validators.py
│   │   └── test_csv_storage.py
│   └── conftest.py           # Pytest fixtures
└── requirements.txt
```

## Implementation Roadmap

### Phase 1: Data Foundation (COMPLETE)

**Status**: Complete

**Implementation**:

- Tank01 RapidAPI client (src/data/collectors/tank01_client.py)
- CSV/Parquet storage backend (src/data/storage/csv_storage.py)
- API endpoint configuration (src/data/collectors/endpoints.py)

**Capabilities**:

- DFS salaries (DraftKings, FanDuel, Yahoo)
- Betting odds and Vegas lines
- Fantasy projections
- Daily schedules and game IDs
- Injury reports
- Team metadata
- Box scores via historical data scripts

**Data Storage**:

- Parquet format for efficiency
- Date-based file organization
- Support for date range queries

### Phase 2: Feature Pipeline

**Goal**: Transform raw data into ML features

**Components**:

- Feature base class with `calculate()` interface
- Feature registry for dynamic composition
- Core feature implementations

**Features**:

- Box score stats (PTS, REB, AST, STL, BLK, TO)
- Rolling averages (3, 5, 10 games)
- EWMA (span=5)
- Minutes per game
- Usage rate

### Phase 3: Model Framework

**Goal**: Pluggable ML models with ~30% MAPE

**Components**:

- Abstract `BaseModel` interface
- Model registry for hot-swapping
- Training/prediction pipeline

**Models**:

- XGBoost (primary, ~30% MAPE target)
- Random Forest (baseline)
- Ridge/Lasso (simple baseline)
- Ensemble (weighted average)

**Interface**:

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry()
registry.register('xgboost', XGBoostModel)
model = registry.create('xgboost')
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

### Phase 4: Optimization

**Goal**: Generate valid DraftKings lineups

**Components**:

- Linear programming optimizer (PuLP)
- DraftKings constraints encoder
- Exposure management

**DraftKings Rules**:

- Total salary ≤ $50,000
- Exactly 8 players
- Positions: PG/SG/SF/PF/C/G/F/UTIL
- Minimum 2 different teams
- Minimum 2 different games

**Interface**:

```python
from src.optimization import get_optimizer

optimizer = get_optimizer(
    strategy='cash',  # or 'gpp'
    projections=projections
)
lineups = optimizer.generate(num_lineups=20)
```

### Phase 5: Evaluation

**Goal**: Validate model performance

**Components**:

- Walk-forward validation framework
- Performance metrics (MAPE, RMSE)
- Backtest runner for historical slates

**Metrics**:

- MAPE (Mean Absolute Percentage Error): Target <30%
- RMSE (Root Mean Squared Error)
- Correlation with actual performance
- ROI tracking by contest type

## Key Design Patterns

### Model Registry

Easy model switching via configuration:

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry()
registry.register('xgboost', XGBoostModel)
registry.register('rf', RandomForestModel)

# Switch models via config
model = registry.create(config['model_type'])
```

### Feature Pipeline

Composable feature engineering:

```python
from src.features import FeaturePipeline

pipeline = FeaturePipeline()
pipeline.add('rolling_avg', window=5)
pipeline.add('dvp_rating')
pipeline.add('vegas_ou')

features = pipeline.transform(raw_data)
```

### Optimizer Strategy

Swap cash game vs GPP strategies:

```python
from src.optimization import get_optimizer

# Cash game: high floor, low variance
cash_optimizer = get_optimizer(strategy='cash', projections=projections)
cash_lineup = cash_optimizer.generate(num_lineups=1)

# GPP: high ceiling, high variance
gpp_optimizer = get_optimizer(strategy='gpp', projections=projections)
gpp_lineups = gpp_optimizer.generate(num_lineups=150)
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

Build historical dataset:

```bash
python scripts/build_historical_game_logs.py
```

Collects schedules and box scores for specified date range. Edit script to configure dates.

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

## Future Development

Phases 2-5 planned:

```bash
# Feature engineering (Phase 2)
python -m src.features.build --config config/features.yaml

# Model training (Phase 3)
python -m src.train --config config/models.yaml

# Lineup optimization (Phase 4)
python -m src.optimize --slate-id 12345 --strategy cash

# Backtesting (Phase 5)
python -m src.backtest --start-date 2024-01-01 --end-date 2024-03-01
```

## License

MIT
