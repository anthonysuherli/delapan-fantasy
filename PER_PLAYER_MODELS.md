# Per-Player Model Implementation

## Overview

The walk-forward backtest now supports training individual XGBoost models for each player, with models saved in a structured directory format.

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

### Configuration Parameters

```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    start_date='20240101',
    end_date='20240131',
    lookback_days=90,
    model_type='xgboost',
    model_params={
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    rolling_window_sizes=[3, 5, 10],
    output_dir='data/backtest_results',
    per_player_models=True,        # ENABLE PER-PLAYER MODELS
    min_player_games=10             # MINIMUM GAMES REQUIRED
)
```

**Key Parameters:**

- `per_player_models=True`: Enables per-player model training
- `min_player_games=10`: Minimum games required in lookback period to train a model for that player

### Running Backtest with Per-Player Models

```python
from src.walk_forward_backtest import WalkForwardBacktest

backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    start_date='20240115',
    end_date='20240115',
    per_player_models=True,
    min_player_games=10
)

results = backtest.run()
```

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

1. **Personalized predictions**: Each model learns player-specific patterns
2. **Better accuracy**: Captures individual player tendencies and consistency
3. **Interpretability**: Can analyze model importance features per player
4. **Flexibility**: Can use different model parameters per player in future
5. **Reproducibility**: Models are saved with all metadata for exact reproduction

## Notebook Usage

Use the [walk_forward_backtest_runner.ipynb](notebooks/walk_forward_backtest_runner.ipynb) notebook:

1. Set `PER_PLAYER_MODELS = True` in the configuration cell
2. Run backtest
3. Inspect saved models in the "Inspect Per-Player Models" section
4. View model metadata and training sample distributions

## Testing

Test script available at `scripts/test_per_player_models.py`:

```bash
python scripts/test_per_player_models.py
```

This runs a single-date backtest to verify model saving functionality.

## Model File Size

- Each `.pkl` file: ~50-500 KB (depends on model complexity)
- Each `.json` file: ~1 KB
- For 100 players per day: ~5-50 MB per day
- For 30 days: ~150 MB - 1.5 GB

## Performance Considerations

- **Training time**: Longer than single global model (trains N models instead of 1)
- **Memory**: Moderate (models trained sequentially, not in parallel)
- **Storage**: More disk space required for model files
- **Prediction time**: Similar to global model (one prediction per player)

## Future Enhancements

1. Parallel model training across players
2. Model versioning and comparison
3. Automatic model retraining based on performance
4. Ensemble predictions combining multiple player models
5. Feature importance analysis per player
6. Model performance tracking over time
