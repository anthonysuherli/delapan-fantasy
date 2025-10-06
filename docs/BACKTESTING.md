# NBA DFS Walk-Forward Backtesting System

Comprehensive backtesting framework for validating NBA DFS projection models with strict temporal ordering to prevent lookahead bias.

## Overview

The backtesting system implements walk-forward validation to:
- Train models on historical data
- Generate projections for upcoming slates
- Evaluate accuracy against actual results
- Prevent lookahead bias through strict temporal ordering
- Calculate projection metrics (MAPE, RMSE, correlation)
- Simulate contest performance (cash games and GPP)

## Architecture

### Core Modules

#### 1. BacktestConfig (`src/evaluation/backtest_config.py`)
Configuration management for backtest runs.

```python
from src.evaluation.backtest_config import BacktestConfig

config = BacktestConfig(
    start_date='20231101',
    end_date='20240331',
    lookback_days=90,
    model_type='xgboost'
)
```

#### 2. HistoricalDataLoader (`src/evaluation/data_loader.py`)
Loads historical data with strict temporal ordering.

Key features:
- Prevents lookahead bias with `< end_date` filtering
- Validates data completeness
- Loads salaries, schedule, betting odds, injuries

#### 3. FeatureBuilder (`src/evaluation/feature_builder.py`)
Generates ML features from historical data.

Features:
- Rolling averages (3, 5, 10 game windows)
- Exponentially weighted moving averages (EWMA)
- Standard deviation for consistency metrics
- Min/max values for floor/ceiling
- DraftKings fantasy points calculation

#### 4. WalkForwardBacktest (`src/evaluation/walk_forward.py`)
Main backtest engine orchestrating the entire process.

Process:
1. Load slate dates in backtest period
2. For each date:
   - Load training data (strictly before test date)
   - Build features
   - Train model
   - Generate projections
   - Load actual results
   - Evaluate performance
3. Aggregate results

#### 5. Metrics (`src/evaluation/metrics.py`)
Evaluation metrics and contest simulation.

Metrics:
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- Correlation

Simulations:
- Cash game (50/50) ROI
- GPP tournament results

#### 6. Analysis (`src/evaluation/analysis.py`)
Results analysis and visualization.

Outputs:
- Summary statistics
- Performance plots (MAPE, RMSE, correlation over time)
- Error analysis
- Daily results CSV

## Usage

### Basic Usage

```bash
python scripts/run_walk_forward_backtest.py \
  --start 20231101 \
  --end 20240331 \
  --lookback 90 \
  --model xgboost
```

### Advanced Usage

```bash
python scripts/run_walk_forward_backtest.py \
  --start 20231101 \
  --end 20240331 \
  --lookback 90 \
  --model random_forest \
  --min-training-games 1000 \
  --output-dir data/backtest_results_rf \
  --debug
```

### Using Config File

```bash
python scripts/run_walk_forward_backtest.py \
  --config config/backtest_production.json
```

Example config file:
```json
{
  "start_date": "20231101",
  "end_date": "20240331",
  "lookback_days": 90,
  "model_type": "xgboost",
  "model_params": {
    "max_depth": 8,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 0.8
  },
  "features": [
    "pts_avg_3", "pts_avg_5", "pts_avg_10",
    "fpts_avg_3", "fpts_avg_5", "fpts_ewma"
  ],
  "min_training_games": 500,
  "output_dir": "data/backtest_results"
}
```

## Output

### Directory Structure

```
data/backtest_results/
├── backtest_config.json          # Config used for run
├── backtest_results.json         # Complete results
├── backtest_plots.png            # Visualization
├── daily_results.csv             # Per-slate metrics
├── summary.txt                   # Text summary
└── daily/                        # Per-slate details
    ├── 20231101/
    │   ├── metrics.json
    │   └── projections_vs_actuals.csv
    ├── 20231102/
    │   ├── metrics.json
    │   └── projections_vs_actuals.csv
    └── ...
```

### Results Format

```json
{
  "num_slates": 120,
  "date_range": "20231101 to 20240331",
  "mean_mape": 28.7,
  "median_mape": 27.9,
  "std_mape": 4.2,
  "mean_rmse": 8.9,
  "std_rmse": 1.2,
  "mean_correlation": 0.67,
  "std_correlation": 0.08,
  "total_players_evaluated": 14523,
  "avg_players_per_slate": 121.0
}
```

## Temporal Consistency Guarantees

### Critical Design Principles

1. **Strict Date Filtering**
   - Training data: `gameDate < test_date` (exclusive)
   - Never includes games on test date
   - Enforced in `HistoricalDataLoader.load_historical_player_logs()`

2. **Feature Calculation**
   - Rolling features computed from prior games only
   - For game at index `i`, uses games `0:i` (exclusive)
   - EWMA uses chronological ordering

3. **Validation**
   - Tests verify no lookahead bias
   - `test_no_lookahead_bias_in_data_loader()`
   - `test_feature_calculation_uses_prior_games_only()`

## DraftKings Scoring

Exact DraftKings fantasy point calculation:

```
Points:        +1.0 per point
Rebounds:      +1.25 per rebound
Assists:       +1.5 per assist
Steals:        +2.0 per steal
Blocks:        +2.0 per block
Turnovers:     -0.5 per turnover
Double-Double: +1.5 bonus
Triple-Double: +3.0 bonus (includes double-double)
```

Implementation in `FeatureBuilder.calculate_dk_fantasy_points()`

## Testing

Run temporal consistency tests:

```bash
pytest tests/evaluation/test_backtest_temporal.py -v
```

Tests verify:
- No lookahead bias in data loading
- Feature calculation uses prior games only
- DK scoring accuracy
- Config validation
- Feature consistency between training and slate
- Rolling window calculations

## Performance Targets

- **MAPE < 30%**: Primary accuracy target
- **RMSE < 10 points**: Secondary target
- **Correlation > 0.6**: Projection quality indicator

## Model Support

Supported models:
- `xgboost`: XGBoost regressor (recommended)
- `random_forest`: Random Forest regressor
- `linear`: Ridge regression baseline

## Best Practices

1. **Sufficient Training Data**
   - Use `--min-training-games 500` minimum
   - Larger lookback for more stable models
   - Balance recency vs. sample size

2. **Feature Engineering**
   - Include multiple window sizes (3, 5, 10 games)
   - EWMA captures recent trends
   - Standard deviation for consistency

3. **Model Selection**
   - Start with XGBoost for best performance
   - Use Random Forest for faster iteration
   - Linear for baseline comparison

4. **Validation**
   - Always review temporal consistency tests
   - Check daily results for anomalies
   - Analyze error patterns by date

## Troubleshooting

### Issue: MAPE > 50%

**Causes:**
- Insufficient training data
- Poor feature selection
- Model overfitting

**Solutions:**
- Increase lookback_days
- Review feature importance
- Tune model hyperparameters

### Issue: No slate dates found

**Causes:**
- Database empty or corrupted
- Date range outside available data

**Solutions:**
- Check database: `sqlite3 nba_dfs.db "SELECT COUNT(*) FROM games;"`
- Verify date range: `sqlite3 nba_dfs.db "SELECT MIN(gameDate), MAX(gameDate) FROM games;"`

### Issue: Insufficient training data

**Causes:**
- Lookback window too short
- Sparse historical data

**Solutions:**
- Increase lookback_days
- Lower min_training_games threshold
- Collect more historical data

## Future Enhancements

Planned features:
- Lineup optimization (cash and GPP)
- Defense vs Position (DvP) ratings
- Vegas line integration
- Pace and usage metrics
- Injury impact modeling
- Home/away splits
- Back-to-back game adjustments

## References

- Specification: Implementation Instructions in repo root
- DK Scoring: https://www.draftkings.com/help/rules/nba
- Walk-Forward Validation: https://en.wikipedia.org/wiki/Walk_forward_optimization

## Support

For issues or questions:
1. Check troubleshooting section
2. Review test failures
3. Enable debug logging: `--debug`
4. Examine daily results in output directory