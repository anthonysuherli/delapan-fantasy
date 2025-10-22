# Backtesting Documentation

## Overview

This directory contains comprehensive documentation for the NBA DFS backtesting pipeline. The pipeline evaluates fantasy point prediction models using walk-forward validation across historical slates.

## Documentation Structure

### Core Pipeline Documentation

1. **[BACKTESTING_PIPELINE_FLOW.md](BACKTESTING_PIPELINE_FLOW.md)** - Main Reference
   - Complete 13-section technical specification
   - 5-layer architecture (data → features → models → evaluation → output)
   - Detailed processing flow for each of 11 pipeline stages
   - Configuration parameters and data structures
   - Logic decisions and validation strategies
   - Resource consumption estimates
   - Common issues & solutions

2. **[BACKTESTING_FLOWCHART.md](BACKTESTING_FLOWCHART.md)** - Visual Reference
   - Simplified flow diagrams
   - Decision trees for model recalibration
   - Directory structure visualization
   - Error handling flows
   - Performance benchmarks
   - Temporal validation timeline

3. **[BACKTESTING_DATA_EXAMPLES.md](BACKTESTING_DATA_EXAMPLES.md)** - Concrete Examples
   - Actual data transformations at each stage
   - Raw box scores → Feature matrices → Predictions → Results
   - Metric calculations with real numbers
   - File outputs with sizes and formats
   - Daily and season aggregation examples

### Supplementary Documentation

4. **[VALID_BACKTEST_CONFIGURATIONS.md](VALID_BACKTEST_CONFIGURATIONS.md)**
   - Configuration examples for different scenarios
   - Parameter combinations tested and validated
   - Use cases and recommendations

5. **[CHECKPOINT_RESUME.md](CHECKPOINT_RESUME.md)**
   - Checkpoint system for resuming interrupted backtests
   - File formats and recovery strategies

6. **[LINEUP_GENERATION.md](LINEUP_GENERATION.md)**
   - Lineup optimization using backtest predictions
   - Integration with DraftKings constraints

7. **[BACKTEST_TROUBLESHOOTING.md](BACKTEST_TROUBLESHOOTING.md)**
   - Common errors and solutions
   - Performance optimization tips

8. **[BACKTEST_FIXES_SUMMARY.md](BACKTEST_FIXES_SUMMARY.md)**
   - Recent fixes and improvements
   - Version history

## Quick Start

### Running a Basic Backtest

```python
from src.walk_forward_backtest import WalkForwardBacktest

backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start=20241001,
    train_end=20250201,
    test_start=20250205,
    test_end=20250215,
    model_type='xgboost',
    feature_config='default_features',
    per_player_models=False,
    n_jobs=32
)

results = backtest.run()
```

### Key Pipeline Stages

1. **Initialization** → Load configs, prepare storage
2. **Benchmark Setup** → Calculate baseline season averages
3. **Slate Loop** → Process each test day
   - Load data & apply filters
   - Retrain model (if needed)
   - Generate predictions
   - Evaluate against actuals
4. **Aggregation** → Combine results, generate reports

### Data Flow Summary

```
SQLite DB → Load → Feature Engineer → Train Model → Predict →
Merge Actuals → Calculate Metrics → Aggregate → Report HTML/PDF
```

### Output Structure

```
data/outputs/{timestamp}/
├── predictions/        # Model projections + actuals
├── inputs/            # Training feature matrices
├── checkpoints/       # Resumption data
└── reports/          # HTML/PDF analysis
```

## Key Concepts

### Model Recalibration

Models are retrained based on:
- `rewrite_models=True` → Always retrain
- First slate → Train
- Days since last training ≥ `recalibrate_days` (default 7) → Retrain
- Otherwise → Reuse previous model

### Feature Engineering

147 features generated from 21 base statistics:
- Rolling windows (3, 5, 10 games) with mean/std
- Exponentially weighted moving averages (span=5)
- Applied to: pts, reb, ast, stl, blk, mins, TOV, and more

### Evaluation Metrics

- **MAPE** (Mean Absolute Percentage Error) - Target < 30% for elite players
- **RMSE** (Root Mean Squared Error) - Standard deviation of errors
- **MAE** (Mean Absolute Error) - Average error magnitude
- **Correlation** - Pearson correlation between actual and predicted

### Temporal Validation

Strict prevention of lookahead bias:
- Training data must be BEFORE test dates
- Features derived from historical games only
- Validation checks: `max(gameDate) < train_end`

## Performance Expectations

### Model Performance
- Elite players ($8k+): MAPE ~27-30%
- Mid-tier ($6-8k): MAPE ~31-35%
- Value players ($5-6k): MAPE ~40-45%
- Overall correlation: 0.70-0.75

### Computation Time
- Per-slate model: ~30 seconds/slate
- Per-player models: ~90 seconds/slate (with 32 cores)
- Full backtest (100 slates): 1-2 hours

### Resource Usage
- Memory: 1-15GB depending on configuration
- Storage: 1-10GB per 100 slates
- CPU: Benefits from parallelization (n_jobs=-1)

## Common Configuration Patterns

### Fast Development
```python
config = {
    'feature_config': 'base_features',  # Fewer features
    'per_player_models': False,         # Single model
    'save_models': False,               # Don't persist
    'n_jobs': 1                         # Sequential
}
```

### Production Quality
```python
config = {
    'feature_config': 'opponent_features',  # Full features
    'per_player_models': True,              # Individual models
    'save_models': True,                    # Persist everything
    'n_jobs': -1                            # All CPU cores
}
```

### GPU Acceleration
```python
config = {
    'model_params': {
        'device': 'cuda:0',
        'tree_method': 'hist'
    }
}
```

## Troubleshooting

### No slate dates found
→ Run data collection scripts for date range

### High MAPE values
→ Check training data quantity, increase date range
→ Verify feature configuration
→ Review player filtering settings

### GPU out of memory
→ Reduce batch size or fallback to CPU
→ Set `use_gpu=False`

### Slow performance
→ Use `base_features` for faster iteration
→ Increase `n_jobs` for parallelization
→ Enable feature caching

## See Also

- [scripts/](../../scripts/) - Command-line scripts for running backtests
- [notebooks/](../../notebooks/) - Interactive notebooks for analysis
- [config/features/](../../config/features/) - Feature configuration YAML files
- [src/walk_forward_backtest.py](../../src/walk_forward_backtest.py) - Core implementation