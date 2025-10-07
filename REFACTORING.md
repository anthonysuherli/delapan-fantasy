# Refactoring Documentation

## Overview

This document explains the refactoring of the NBA DFS backtesting codebase from a monolithic structure to a clean, modular architecture.

## Changes Made

### Before (Monolithic Structure)

```
src/
├── backtest/
│   ├── __init__.py
│   └── backtest.py (1147 lines - too large)
└── evaluation/
    ├── __init__.py
    ├── base.py
    ├── backtest_config.py
    ├── data_loader.py
    ├── feature_builder.py
    ├── metrics.py
    └── walk_forward.py
```

Problems:
- src/backtest/backtest.py was too large (1147 lines)
- Mixed responsibilities in single files
- Circular dependencies between backtest and evaluation
- Difficult to test individual components
- Hard to extend with new features

### After (Modular Structure)

```
src/
├── walk_forward_backtest.py    (Clean walk-forward backtest runner)
├── historical_data_loader.py   (Data loading from database)
├── feature_builder_v2.py       (Feature calculation logic)
├── metrics.py                  (Evaluation metrics)
└── features/
    ├── __init__.py
    └── rolling_window_features.py (Pluggable feature calculators)
```

Benefits:
- Single responsibility per module
- Clear separation of concerns
- Easy to test individual components
- Extensible feature system
- Reduced file sizes (200-300 lines each)

## Module Responsibilities

### src/walk_forward_backtest.py
Main walk-forward backtesting orchestrator.

**Responsibilities:**
- Coordinate backtest execution
- Train models on historical data
- Generate projections for slates
- Evaluate predictions against actuals
- Aggregate results

**Key Classes:**
- `WalkForwardBacktest`: Main backtest runner

**Usage:**
```python
from src.walk_forward_backtest import WalkForwardBacktest

# Global model (one model for all players)
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    start_date='20240101',
    end_date='20240131',
    lookback_days=90,
    model_type='xgboost'
)

results = backtest.run()

# Per-player models (individual model for each player)
backtest_per_player = WalkForwardBacktest(
    db_path='nba_dfs.db',
    start_date='20240101',
    end_date='20240131',
    lookback_days=90,
    model_type='xgboost',
    per_player_models=True,
    min_player_games=10
)

results = backtest_per_player.run()
```

### src/historical_data_loader.py
Data loading and retrieval from database.

**Responsibilities:**
- Load slate dates from database
- Load DFS salaries for specific dates
- Load historical player logs
- Load game schedules and betting odds
- Validate data completeness

**Key Classes:**
- `HistoricalDataLoader`: Database data retrieval

**Usage:**
```python
from src.historical_data_loader import HistoricalDataLoader

loader = HistoricalDataLoader('nba_dfs.db')
slate_data = loader.load_slate_data('20240115')
training_data = loader.load_historical_player_logs('20240115', lookback_days=90)
```

### src/feature_builder_v2.py
Feature calculation from raw player data.

**Responsibilities:**
- Calculate DraftKings fantasy points
- Build training features with temporal ordering
- Build slate features for prediction
- Calculate rolling window statistics

**Key Classes:**
- `FeatureBuilder`: Feature calculation engine

**Usage:**
```python
from src.feature_builder_v2 import FeatureBuilder

builder = FeatureBuilder()
X_train, y_train = builder.build_training_features(
    training_data,
    window_sizes=[3, 5, 10]
)
slate_features = builder.build_slate_features(
    slate_data,
    training_data,
    window_sizes=[3, 5, 10]
)
```

### src/metrics.py
Evaluation metrics for prediction quality.

**Responsibilities:**
- Calculate MAPE (Mean Absolute Percentage Error)
- Calculate RMSE (Root Mean Squared Error)
- Calculate MAE (Mean Absolute Error)
- Calculate correlation coefficient
- Calculate R-squared

**Functions:**
```python
from src.metrics import (
    calculate_mape,
    calculate_rmse,
    calculate_correlation,
    evaluate_predictions
)

mape = calculate_mape(actuals, predictions)
results = evaluate_predictions(actuals, predictions)
```

### src/features/
Pluggable feature calculator modules.

**Responsibilities:**
- Modular feature calculation
- Extensible feature system
- Isolated feature logic

**Modules:**
- `rolling_window_features.py`: Rolling window statistics

**Usage:**
```python
from src.features import RollingWindowFeatureCalculator

calculator = RollingWindowFeatureCalculator(window_sizes=[3, 5, 10])
features = calculator.calculate_features(
    player_name='LeBron James',
    game_date='20240115',
    historical_data=df
)
```

## Migration Guide

### Old Code
```python
from src.backtest.backtest import DailyBacktest, WalkForwardValidator
from src.evaluation.feature_builder import FeatureBuilder
from src.evaluation.metrics import calculate_mape
```

### New Code
```python
from src.walk_forward_backtest import WalkForwardBacktest
from src.feature_builder_v2 import FeatureBuilder
from src.metrics import calculate_mape
```

## Design Principles

### 1. Single Responsibility
Each module has one clear purpose:
- `walk_forward_backtest.py`: Orchestrate backtesting
- `historical_data_loader.py`: Load data
- `feature_builder_v2.py`: Calculate features
- `metrics.py`: Evaluate predictions

### 2. Dependency Direction
Clean dependency flow:
```
walk_forward_backtest.py
    -> historical_data_loader.py
    -> feature_builder_v2.py
    -> metrics.py
    -> features/
```

### 3. Testability
Each module can be tested independently:
```python
def test_feature_builder():
    builder = FeatureBuilder()
    features = builder.calculate_dk_fantasy_points(stats)
    assert features > 0
```

### 4. Extensibility
Easy to add new features:
```python
src/features/
├── rolling_window_features.py
├── advanced_metrics.py      (NEW)
├── opponent_features.py     (NEW)
└── vegas_features.py        (NEW)
```

## Performance Improvements

1. Removed redundant database connections
2. Simplified feature calculation loops
3. Reduced memory overhead
4. Cleaner error handling

## Testing Strategy

Each module has corresponding tests:
```
tests/
├── test_backtest.py
├── test_data_loader.py
├── test_feature_engineering.py
├── test_metrics.py
└── features/
    └── test_rolling_window_features.py
```

## Per-Player Models

The backtesting framework supports two modeling approaches:

### Global Model (Default)
Trains a single model on all players' combined data.

**Pros:**
- More training samples
- Faster training
- Better for players with limited history
- Generalizes across player types

**Cons:**
- Cannot capture player-specific patterns
- May underperform for unique player profiles

**Usage:**
```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    start_date='20240101',
    end_date='20240131',
    per_player_models=False  # Default
)
```

### Per-Player Models
Trains individual models for each player using only their historical data.

**Pros:**
- Captures player-specific patterns
- Better for consistent players with ample history
- No cross-player contamination

**Cons:**
- Requires minimum game threshold per player
- Slower training (N models vs 1 model)
- May overfit for players with limited data
- Players without sufficient history are excluded

**Usage:**
```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    start_date='20240101',
    end_date='20240131',
    per_player_models=True,
    min_player_games=10  # Minimum games required per player
)
```

**Output:**
Per-player models save training data to disk for inspection:
```
data/backtest_results/
└── player_models/
    └── 20240115/
        ├── LeBron_James.parquet
        ├── Stephen_Curry.parquet
        └── ...
```

Each parquet file contains:
- Feature columns
- Target column (fpts)
- playerID
- playerName

This allows you to:
1. Inspect what features are being used for each player
2. Analyze per-player model performance
3. Debug prediction issues for specific players

## Future Enhancements

1. Add more feature calculators to `src/features/`
2. Implement feature registry pattern
3. Add caching layer for repeated queries
4. Hybrid models (global + per-player ensemble)
5. Add lineup optimization integration
6. Model persistence and reuse across slates

## Notes

- Old structure kept in `src/backtest/` and `src/evaluation/` for backward compatibility
- Remove old structure once migration is complete
- All new development should use refactored modules
