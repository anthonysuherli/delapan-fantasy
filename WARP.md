# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

NBA DFS machine learning pipeline for DraftKings optimization using per-player XGBoost models. The system achieves ~30% MAPE on elite player projections through a modular, configuration-driven architecture with YAML-based features and registry patterns for hot-swapping components.

## Development Commands

### Installation & Setup
```powershell
pip install -r requirements.txt
```

Set up Tank01 API key in `.env`:
```powershell
TANK01_API_KEY=your_rapidapi_key_here
```

### Testing
```powershell
# Run all tests
pytest tests/

# Verbose testing for specific modules
pytest tests/data/ -v
pytest tests/features/ -v
```

### Data Collection
```powershell
# Collect historical game data
python scripts/collect_games.py --start-date 20241201 --end-date 20241231

# Collect DFS salaries  
python scripts/collect_dfs_salaries.py --start-date 20241201 --end-date 20241231

# Load data to SQLite database
python scripts/load_games_to_db.py
```

### Backtesting Commands
```powershell
# Single-day per-player backtest
python scripts/run_backtest.py --test-start 20250205 --test-end 20250206 --per-player

# Multi-day season backtest
python scripts/run_backtest.py --test-start 20250201 --test-end 20250228 --per-player --recalibrate-days 7

# GPU-accelerated backtest (if available)
python scripts/run_backtest_gpu.py --test-start 20250205 --test-end 20250206 --model-config config/models/xgboost_a100.yaml --per-player --gpu-id 0

# Fast iteration with minimal features
python scripts/run_backtest.py --test-start 20250205 --test-end 20250206 --feature-config base_features
```

### Single Test Execution
```powershell
# Test specific model training
pytest tests/models/test_xgboost_model.py::test_training -v

# Test specific data collector
pytest tests/data/collectors/test_tank01_client.py::test_get_dfs_salaries -v
```

## Architecture Overview

### Five-Layer Modular Architecture

The system follows a registry-based pattern enabling hot-swapping of components through configuration:

**1. Data Layer (`src/data/`)**
- `Tank01Client`: RapidAPI wrapper for NBA data collection with rate limiting (1000 requests/month)
- `ParquetStorage`: Date-partitioned columnar storage in `data/inputs/` subdirectories
- `HistoricalDataLoader`: Temporal validation to prevent lookahead bias in training data
- `Cache`: API response caching to minimize rate limit usage

**2. Feature Layer (`src/features/`)**
- `FeaturePipeline`: Sequential transformer pipeline with YAML configuration
- `RollingStatsTransformer`: 3/5/10-game rolling averages with standard deviation
- `EWMATransformer`: Exponentially weighted moving averages (span=5)
- Configuration-driven: 147 features from 21 box score statistics

**3. Model Layer (`src/models/`)**
- `XGBoostModel`: Gradient boosting with Bayesian hyperparameter optimization
- `RandomForestModel`: Ensemble baseline for comparison
- Per-player models: 500+ individual models per slate vs single slate-level model
- Model serialization with metadata and training inputs for reproducibility

**4. Optimization Layer (`src/optimization/`)**
- `LinearProgramOptimizer`: PuLP-based integer linear programming
- `DraftKingsConstraints`: 8 players, $50K salary cap, position requirements
- Multi-lineup generation capability

**5. Evaluation Layer (`src/evaluation/`)**
- `WalkForwardBacktest`: Automated model recalibration every N days with benchmark comparison
- `SeasonAverageBenchmark`: Statistical baseline using player season averages
- Metrics: MAPE, RMSE, MAE, Correlation with salary tier analysis
- Statistical significance testing (paired t-test, Cohen's d)

### Registry Pattern Implementation

Components are registered for runtime hot-swapping:

```python
# Model registry usage
from src.models.registry import ModelRegistry
model = ModelRegistry.create('xgboost', config)

# Feature registry usage  
from src.features.registry import FeatureRegistry
transformer = FeatureRegistry.create('rolling_stats', windows=[3,5,10])
```

### Configuration-Driven Design

YAML files control feature engineering and model behavior:

- `config/features/default_features.yaml`: Full 21-stat, 147-feature pipeline
- `config/features/base_features.yaml`: Minimal 6-stat pipeline for fast iteration
- `config/models/xgboost_default.yaml`: CPU training parameters
- `config/models/xgboost_a100.yaml`: GPU training with `device: cuda:0`
- `config/experiments/`: Backtest configurations for reproducible experiments

## Key Implementation Details

### Per-Player vs Slate-Level Models
The system supports both training approaches:
- **Per-player**: Individual XGBoost model per player (500+ models/slate, +1.7-2.1% accuracy)
- **Slate-level**: Single model for all players (baseline comparison)

Per-player models are preferred and show statistically significant improvements over baselines.

### Walk-Forward Validation Framework
`WalkForwardBacktest` class implements:
- Automated model recalibration every N days (default: 7)
- Temporal data splitting to prevent lookahead bias
- Benchmark comparison with statistical significance testing
- Model and prediction persistence for analysis
- Error analysis by salary tier ($0-4K, $4-6K, $6-8K, $8K+)

### Data Storage Architecture
Two deployment patterns supported:
1. **Integrated**: Code and data in same directory (default)
2. **Separated**: Code repo separate from data directory for better organization

### Performance Benchmarks
Current performance on elite players ($8K+ salary):
- Target: 30% MAPE 
- Achieved: 32.9% MAPE (close to research target)
- Overall correlation: 0.728
- Coverage: 96%+ players with trained models

## Development Patterns

### Adding New Features
1. Create transformer class extending `FeatureTransformer` in `src/features/transformers/`
2. Implement `fit()` and `transform()` methods
3. Register in `FeatureRegistry`
4. Add to YAML configuration in `config/features/`

### Adding New Models  
1. Create model class extending `BaseModel` in `src/models/`
2. Implement `train()`, `predict()`, `save()`, `load()` methods
3. Register in `ModelRegistry`
4. Add configuration in `config/models/`

### Configuration Loading Pattern
```python
from src.utils.feature_config import load_feature_config

feature_config = load_feature_config('default_features')
pipeline = feature_config.build_pipeline(FeaturePipeline)
```

### Temporal Data Validation
All historical data loading includes temporal validation:
```python
from src.data.loaders.historical_loader import HistoricalDataLoader

loader = HistoricalDataLoader(storage)
# Prevents using future data in training
player_logs = loader.load_historical_player_logs(end_date, lookback_days=365)
```

## Important Context from CLAUDE.md

### API Rate Limits
- Tank01 RapidAPI: 1000 requests/month (free tier)
- Estimate: ~12 requests per game day (1 schedule + 11 games average)
- Cache responses to minimize usage via `src/data/collectors/cache.py`

### Training Performance
- Per-player models: 500+ models per slate requires significant compute
- GPU acceleration available for XGBoost 2.0+ with `device: cuda:0`
- Parallel training with `n_jobs` parameter for CPU systems

### Known Issues & Roadmap
1. Low-output player MAPE inflation (103.6% for $0-4K salary tier)
2. Missing contextual features (home/away, rest days, matchups)  
3. No injury/inactive status filtering during prediction
4. Variance prediction challenges (hot/cold streaks)

Priority improvements: Injury filtering, starter/bench indicators, contextual features.

