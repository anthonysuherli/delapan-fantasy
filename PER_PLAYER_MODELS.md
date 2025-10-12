# Per-Player Model Implementation

## Overview

Per-player XGBoost models train individual models for each player to capture player-specific patterns. Implemented in notebooks/backtest_1d_by_player.ipynb with configuration-driven features and Bayesian hyperparameter optimization.

## Implementation Details

### Directory Structure

Models are saved in the following structure:
```
data/models/
└── {year}/
    └── {month}/
        └── {day}/
            ├── {player_name}_{player_id}.pkl
            ├── {player_name}_{player_id}.json
            └── ...
```

**Example:**
```
data/models/
└── 2024/
    └── 01/
        └── 15/
            ├── LeBron_James_2544.pkl
            ├── LeBron_James_2544.json
            ├── Stephen_Curry_201939.pkl
            └── Stephen_Curry_201939.json
```

### Model Files

Each player gets two files:

1. **`.pkl` file** (Pickle file containing):
   - Trained XGBoost model
   - Player name
   - Player ID
   - Number of training samples
   - Model type
   - Window sizes used
   - Lookback days

2. **`.json` file** (Metadata):
   - Player name
   - Player ID
   - Number of training samples
   - Model type
   - Window sizes
   - Lookback days
   - Model filename

## Usage

### Using Notebook (Recommended)

Open notebooks/backtest_1d_by_player.ipynb and configure:

```python
# Configuration
SLATE_DATE = '20250205'
LOOKBACK_DAYS = 365
MIN_GAMES_THRESHOLD = 10

# Feature configuration
feature_config = load_feature_config('default_features')

# XGBoost hyperparameters (use Bayesian optimized values)
xgb_params = {
    'max_depth': 5,
    'learning_rate': 0.08,
    'n_estimators': 289,
    'subsample': 0.82,
    'colsample_bytree': 0.75,
    'objective': 'reg:squarederror',
    'random_state': 42
}
```

Run notebook to:
1. Load slate and historical data
2. Build features from YAML config
3. Train per-player XGBoost models
4. Generate projections
5. Calculate MAPE, RMSE, MAE, Correlation
6. Analyze errors by salary tier

### Key Parameters

- `SLATE_DATE`: Date to generate projections for
- `LOOKBACK_DAYS`: Historical data window (typically 365 for 1 season)
- `MIN_GAMES_THRESHOLD`: Minimum games required to train model (default: 10)
- `feature_config`: Load from config/features/*.yaml
- `xgb_params`: Hyperparameters (use optimize_xgboost_hyperparameters.py for tuning)

### Loading Saved Models

```python
import pickle
import json
from pathlib import Path

model_file = Path('data/models/2024/01/15/LeBron_James_2544.pkl')
metadata_file = Path('data/models/2024/01/15/LeBron_James_2544.json')

with open(metadata_file, 'r') as f:
    metadata = json.load(f)
    print(f"Player: {metadata['player_name']}")
    print(f"Training samples: {metadata['num_training_samples']}")

with open(model_file, 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    predictions = model.predict(X_test)
```

## How It Works

### Backtest Process

For each slate date:

1. **Load slate data** (salaries, schedule, etc.)
2. **Load historical data** (lookback period before slate date)
3. **For each player on the slate**:
   - Filter historical data for that player
   - Check if player has minimum required games
   - Build features from player's historical games
   - Train individual XGBoost model on player's data
   - Save model to `data/models/{year}/{month}/{day}/{player_name}_{player_id}.pkl`
   - Save metadata to corresponding `.json` file
   - Generate projection using player's model
4. **Evaluate** projections against actual results
5. **Aggregate** results across all slates

### Feature Engineering

Each player model uses:
- Rolling window statistics (3, 5, 10 game windows by default)
- Stats: pts, reb, ast, stl, blk, TOV, mins, fpts, usage
- Features: avg, std, min, max, EWMA for each stat
- Only historical data before the prediction date (no lookahead bias)

## Benefits of Per-Player Models

1. **Personalized predictions**: Each model learns player-specific patterns and consistency
2. **Elite player accuracy**: 32.9% MAPE for $8k+ players (near 30% target)
3. **Interpretability**: Analyze feature importance per player
4. **Flexibility**: Different hyperparameters per player in future
5. **Reproducibility**: Models saved with metadata for exact reproduction

## Performance Benchmarks

Single-day backtest (2025-02-05):
- Elite players ($8k+): 32.9% MAPE, 25 players
- High salary ($6-8k): 51.8% MAPE, 28 players
- Mid salary ($4-6k): 76.8% MAPE, 75 players
- Low salary ($0-4k): 103.6% MAPE, 111 players
- Overall: 81.18% MAPE, 0.728 correlation
- Coverage: 96.4% (239/248 matched players)

Elite tier performance near production target. Low-output players require different approach (classification vs regression, contextual features).

## Current Notebooks

1. backtest_1d_by_player.ipynb: Single-day per-player backtest with full analysis
2. backtest_1d_by_slate.ipynb: Slate-level baseline for comparison
3. backtest_season.ipynb: Season-long walk-forward validation

See notebooks/performance_report_20250205.md for detailed analysis.

## Model File Size

- Each `.pkl` file: ~50-500 KB (depends on model complexity)
- Each `.json` file: ~1 KB
- For 100 players per day: ~5-50 MB per day
- For 30 days: ~150 MB - 1.5 GB

## Performance Considerations

- **Training time**: ~5-10 minutes for 500+ players (sequential training)
- **Memory**: Moderate (models trained one at a time)
- **Storage**: 5-50 MB per day (500 players x 50-500 KB per model)
- **Prediction time**: Fast (one forward pass per player)

## Known Issues and Roadmap

### Critical Issues
1. Low-output player MAPE inflation (103.6% for $0-4k tier)
   - Division by near-zero inflates percentage errors
   - Need classification approach or alternative metric
2. Missing injury/inactive status filtering
   - Models assume historical minutes continue
   - Need DNP detection before prediction
3. No contextual features
   - Missing: home/away, rest days, opponent strength
   - Missing: starter/bench role, recent form
4. Variance prediction failure
   - Captures mean well (0.728 correlation)
   - Cannot predict hot streaks or cold streaks

### Immediate Priority
1. Add injury/inactive filtering before prediction
2. Implement starter/bench role indicators
3. Add home/away and rest day features
4. Multi-day backtesting for validation

### Future Enhancements
1. Opponent defensive rating features
2. Minutes projection model (separate from points projection)
3. Ensemble methods (XGBoost + Random Forest)
4. Quantile regression for confidence intervals
5. Per-player hyperparameter tuning
6. Parallel training across players
7. Model versioning and performance tracking over time
