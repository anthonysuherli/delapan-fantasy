# Bayesian Hyperparameter Optimization

## Overview

Bayesian optimization is implemented in `src/utils/bayesian_optimizer.py` using scikit-optimize (skopt) with Gaussian Process optimization.

## Current Implementation

### Core Function

```python
from src.utils.bayesian_optimizer import optimize_xgboost_cv

best_params, best_score, history = optimize_xgboost_cv(
    X=X_train,
    y=y_train,
    cv_folds=5,
    n_iterations=50,
    n_initial_points=10,
    scoring='neg_mean_absolute_error',
    random_state=42,
    verbose=True
)
```

### Optimization Script

`scripts/optimize_xgboost_hyperparameters.py`

**Global Optimization** (single set of hyperparameters):
```bash
python scripts/optimize_xgboost_hyperparameters.py \
  --target-date 20250205 \
  --iterations 50 \
  --cv-folds 5
```

**Per-Player Optimization** (separate hyperparameters per player):
```bash
python scripts/optimize_xgboost_hyperparameters.py \
  --per-player \
  --target-date 20250205 \
  --iterations 50 \
  --min-games 30 \
  --n-jobs -1
```

## Optimized Parameters

The optimization tunes 9 XGBoost hyperparameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `max_depth` | [3, 10] | Maximum tree depth |
| `learning_rate` | [0.01, 0.3] | Step size shrinkage (log-uniform) |
| `n_estimators` | [50, 500] | Number of boosting rounds |
| `min_child_weight` | [1, 10] | Minimum sum of instance weight in child |
| `subsample` | [0.5, 1.0] | Subsample ratio of training instances |
| `colsample_bytree` | [0.5, 1.0] | Subsample ratio of columns |
| `gamma` | [0.0, 5.0] | Minimum loss reduction for split |
| `reg_alpha` | [0.0, 1.0] | L1 regularization |
| `reg_lambda` | [0.0, 1.0] | L2 regularization |

## Output Files

### Global Optimization
- **Config**: `config/models/xgboost_optimized.yaml`
- **History**: `config/models/optimization_history.csv`

Example config:
```yaml
model_type: xgboost
hyperparameters:
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 200
  min_child_weight: 5
  subsample: 0.8
  colsample_bytree: 0.8
  gamma: 0.5
  reg_alpha: 0.1
  reg_lambda: 0.3
  objective: reg:squarederror
  random_state: 42
optimization_metadata:
  mode: global
  cv_score_mae: 8.45
  cv_folds: 5
  n_iterations: 50
  optimized_date: 2025-02-05 14:30:00
```

### Per-Player Optimization
- **Config**: `config/models/xgboost_per_player_configs.yaml`
- **History**: `config/models/per_player_optimization_history.csv`

## Usage in Backtest

### Method 1: Run Optimization First, Then Use Results

```bash
# Step 1: Run optimization
python scripts/optimize_xgboost_hyperparameters.py \
  --target-date 20250205 \
  --iterations 50

# Step 2: Load optimized config in Python
import yaml

with open('config/models/xgboost_optimized.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_params = config['hyperparameters']

# Step 3: Run backtest with optimized params
python scripts/run_backtest.py \
  --test-start 20250205 \
  --test-end 20250430 \
  --per-player
```

### Method 2: Notebook Integration

Add optimization cell before backtest:

```python
from src.utils.bayesian_optimizer import optimize_xgboost_cv
from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.loaders.historical_loader import HistoricalDataLoader

# Load training data
storage = SQLiteStorage('nba_dfs.db')
loader = HistoricalDataLoader(storage)
training_data = loader.load_historical_player_logs(
    start_date='20241001',
    end_date='20250204'
)

# Build features (use existing pipeline)
# ... feature engineering code ...

# Run optimization
best_params, best_score, history = optimize_xgboost_cv(
    X=X_train,
    y=y_train,
    cv_folds=5,
    n_iterations=50,
    verbose=True
)

print(f"Best CV Score (MAE): {-best_score:.4f}")
print("Best Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Use optimized params in backtest
MODEL_PARAMS = best_params
```

### Method 3: Per-Player Optimization Integration

```python
# Run per-player optimization
!python scripts/optimize_xgboost_hyperparameters.py \
  --per-player \
  --target-date {TEST_START} \
  --iterations 30 \
  --min-games 30 \
  --n-jobs -1

# Load per-player configs
import yaml

with open('config/models/xgboost_per_player_configs.yaml', 'r') as f:
    player_configs = yaml.safe_load(f)

# Use in per-player training loop
for player_id, player_data in player_training_data.items():
    # Use player-specific params if available
    if str(player_id) in player_configs['per_player_configs']:
        model_params = player_configs['per_player_configs'][str(player_id)]
    else:
        model_params = DEFAULT_PARAMS

    model = XGBoostModel(model_params)
    model.train(X_train, y_train)
```

## Features

### Early Stopping

Stop optimization if no improvement after N iterations:

```python
best_params, best_score, history = optimize_xgboost_cv(
    X=X_train,
    y=y_train,
    early_stopping_rounds=10,
    early_stopping_threshold=1e-4
)
```

### Custom Parameter Bounds

```python
custom_bounds = {
    'max_depth': (4, 8),           # Narrower range
    'learning_rate': (0.01, 0.1),  # Lower learning rates
    'n_estimators': (100, 300)     # Fewer trees
}

best_params, best_score, history = optimize_xgboost_cv(
    X=X_train,
    y=y_train,
    param_bounds=custom_bounds
)
```

### Parallel Per-Player Optimization

```python
from joblib import Parallel, delayed

def optimize_player(player_id, player_data):
    X_player = player_data[feature_cols].fillna(0)
    y_player = player_data['target']

    best_params, best_score, _ = optimize_xgboost_cv(
        X=X_player,
        y=y_player,
        n_iterations=30,
        verbose=False
    )

    return player_id, best_params, -best_score

# Run in parallel
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(optimize_player)(player_id, player_data)
    for player_id, player_data in player_dict.items()
)

# Collect results
player_configs = {pid: params for pid, params, score in results}
```

## Visualization

Analyze optimization history:

```python
import matplotlib.pyplot as plt
import pandas as pd

history = pd.read_csv('config/models/optimization_history.csv')

# Plot convergence
plt.figure(figsize=(12, 6))
plt.plot(history.index, history['score'], marker='o')
plt.axhline(history['score'].max(), color='r', linestyle='--', label=f'Best: {history["score"].max():.4f}')
plt.xlabel('Iteration')
plt.ylabel('CV Score (MAE)')
plt.title('Bayesian Optimization Convergence')
plt.legend()
plt.grid(True)
plt.show()

# Parameter evolution
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
params = ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight',
          'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']

for ax, param in zip(axes.flat, params):
    ax.plot(history.index, history[param], marker='o', alpha=0.6)
    ax.set_title(param)
    ax.set_xlabel('Iteration')
    ax.grid(True)

plt.tight_layout()
plt.show()
```

## Best Practices

### 1. Training Data Selection
- Use representative time period (1-2 seasons)
- Ensure sufficient sample size (>1000 observations)
- Include diverse player types

### 2. Optimization Settings
- **Quick testing**: 20-30 iterations
- **Production**: 50-100 iterations
- **Per-player**: 30-50 iterations (limited by data per player)

### 3. Cross-Validation
- **Standard**: 5-fold CV
- **Time series**: Use TimeSeriesSplit with 5 splits
- **Small datasets**: 3-fold CV

### 4. Computational Resources
- Global optimization: ~5-15 minutes (50 iterations)
- Per-player (500 players): ~2-4 hours with 8 cores
- Use `n_jobs=-1` for parallel per-player optimization

### 5. When to Re-optimize
- At start of each season
- After significant rule changes
- When adding new features
- If model performance degrades (>5% MAPE increase)

## Integration Roadmap

To fully integrate into WalkForwardBacktest:

1. **Add initialization parameters**:
   - `optimize_hyperparameters: bool = False`
   - `optimization_iterations: int = 50`
   - `use_optimized_config: Optional[str] = None`

2. **Add optimization phase** before benchmark initialization

3. **Store optimized params** in `self.optimized_params`

4. **Use optimized params** in model training:
   ```python
   # Global: use self.model_params
   # Per-player: check self.optimized_params[player_id]
   ```

5. **Save optimization results** to run output directory

## Example: Full Workflow

```python
# 1. Optimize hyperparameters
!python scripts/optimize_xgboost_hyperparameters.py \
  --per-player \
  --target-date 20250205 \
  --iterations 50 \
  --min-games 30 \
  --n-jobs -1

# 2. Load optimized config
import yaml

with open('config/models/xgboost_per_player_configs.yaml', 'r') as f:
    opt_config = yaml.safe_load(f)

print(f"Optimized {opt_config['optimization_metadata']['num_players_optimized']} players")

# 3. Run backtest with optimized params
from src.walk_forward_backtest import WalkForwardBacktest

# Load per-player configs
player_configs = opt_config['per_player_configs']

# Create custom backtest with optimized params
# (Requires manual integration or use default params)

backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20241001',
    train_end='20250204',
    test_start='20250205',
    test_end='20250430',
    model_type='xgboost',
    model_params=DEFAULT_PARAMS,  # Fallback for players without optimized params
    per_player_models=True,
    n_jobs=-1
)

results = backtest.run()
```

## Performance Expectations

### Optimization Impact

Expected improvements from Bayesian optimization:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| CV MAE | 10.5 | 9.2 | -12% |
| MAPE (Elite $8k+) | 35% | 30% | -14% |
| Overall MAPE | 85% | 78% | -8% |
| Correlation | 0.68 | 0.73 | +7% |

**Note**: Actual improvements depend on:
- Data quality and quantity
- Feature engineering quality
- Player consistency
- Optimization iterations

### Computational Cost

| Task | Time (8 cores) | API/Compute Cost |
|------|----------------|------------------|
| Global optimization (50 iter) | 10-15 min | Low |
| Per-player (500 players, 30 iter) | 2-4 hours | Medium |
| Per-player (500 players, 50 iter) | 4-6 hours | High |

## Troubleshooting

### Issue: Optimization takes too long

**Solutions**:
- Reduce `n_iterations` (try 20-30)
- Increase `n_initial_points` for faster convergence
- Use smaller parameter ranges
- Enable early stopping

### Issue: Per-player optimization fails for some players

**Causes**:
- Insufficient data (<30 games)
- NaN values in features
- Constant target variable

**Solutions**:
- Increase `--min-games` threshold
- Add data validation
- Use fallback to global params

### Issue: Optimized params perform worse than defaults

**Causes**:
- Overfitting to CV folds
- Non-representative training period
- Different data distribution in test set

**Solutions**:
- Increase CV folds (5-7)
- Use larger, more diverse training set
- Add regularization (increase reg_alpha/reg_lambda ranges)
- Re-optimize with updated data

## References

- **scikit-optimize**: https://scikit-optimize.github.io/
- **XGBoost parameters**: https://xgboost.readthedocs.io/en/latest/parameter.html
- **Bayesian optimization theory**: https://distill.pub/2020/bayesian-optimization/
