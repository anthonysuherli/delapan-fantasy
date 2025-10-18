# Valid Backtest Configurations

Reference configurations that work without errors.

## Current NBA Season 2024-25

**Season dates:** October 22, 2024 - April 13, 2025

### Configuration 1: Full Season Backtest

```python
from src.walk_forward_backtest import WalkForwardBacktest

backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20241022',      # Oct 22, 2024 (season start)
    train_end='20250131',        # Jan 31, 2025
    test_start='20250201',       # Feb 1, 2025
    test_end='20250413',         # Apr 13, 2025 (season end)
    model_type='xgboost',
    feature_config='default_features',
    output_dir='data/backtest_results',
    per_player_models=False,     # Faster, less memory
    recalibrate_days=7,          # Retrain weekly
    minutes_threshold=10,        # Include most rotation players
    n_jobs=1,                    # Single worker
    save_models=False,
    save_predictions=True
)

results = backtest.run()
```

**Time estimate:** 30-60 minutes (single player models: 2-4 hours)

### Configuration 2: Quick Validation (5 Days)

```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20250201',      # Feb 1, 2025
    train_end='20250209',        # Feb 9, 2025 (9 days training)
    test_start='20250210',       # Feb 10, 2025
    test_end='20250215',         # Feb 15, 2025 (5 days testing)
    model_type='xgboost',
    feature_config='base_features',  # Faster: 6 features vs 147
    output_dir='data/backtest_results',
    per_player_models=False,
    recalibrate_days=7,
    minutes_threshold=10,
    n_jobs=1,
    save_models=False,
    save_predictions=True
)

results = backtest.run()
```

**Time estimate:** 2-5 minutes

### Configuration 3: Single Day Backtest

```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20250205',      # Feb 5, 2025
    train_end='20250206',        # Feb 6, 2025 (1 day training - minimal!)
    test_start='20250207',       # Feb 7, 2025
    test_end='20250208',         # Feb 8, 2025 (1 day testing)
    model_type='xgboost',
    feature_config='base_features',
    output_dir='data/backtest_results',
    per_player_models=False,
    recalibrate_days=1,
    minutes_threshold=0,         # Include all players
    n_jobs=1,
    save_models=False,
    save_predictions=False
)

results = backtest.run()
```

**Time estimate:** 30-60 seconds (debugging only)

---

## Previous Season: 2023-24

**Season dates:** October 24, 2023 - April 14, 2024

### Configuration: Historical Analysis

```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',  # Requires 2023-24 data collected
    train_start='20231024',
    train_end='20240131',
    test_start='20240201',
    test_end='20240414',
    model_type='xgboost',
    feature_config='default_features',
    output_dir='data/backtest_results_2024',
    per_player_models=False,
    recalibrate_days=7,
    minutes_threshold=10,
    n_jobs=1,
    save_models=False,
    save_predictions=True
)

results = backtest.run()
```

**Requirements:** Must collect 2023-24 data first:
```bash
python scripts/collect_games.py --start-date 20231024 --end-date 20240414
python scripts/collect_dfs_salaries.py --start-date 20231024 --end-date 20240414
```

---

## Configuration: Invalid Examples (Will Fail)

### ❌ Backwards Dates

```python
# WRONG: train_start > train_end
backtest = WalkForwardBacktest(
    train_start='20250228',    # Later date
    train_end='20250101',      # Earlier date
    test_start='20250301',
    test_end='20250430'
)
# Error: train_start (20250228) must be < train_end (20250101)
```

### ❌ Offseason Dates

```python
# WRONG: No games May-September
backtest = WalkForwardBacktest(
    train_start='20250515',    # May 15 (offseason)
    train_end='20250830',      # August 30 (offseason)
    test_start='20250901',
    test_end='20250930'
)
# Error: No training data found for range 20250515 to 20250830
```

### ❌ Future Dates

```python
# WRONG: Year not collected yet
backtest = WalkForwardBacktest(
    train_start='20260101',    # January 2026 (no data yet)
    train_end='20260228',
    test_start='20260301',
    test_end='20260430'
)
# Error: No training data found for range 20260101 to 20260228
```

### ❌ Overlapping Train/Test

```python
# WARNING: Training data leaks into test period
backtest = WalkForwardBacktest(
    train_start='20250101',
    train_end='20250315',      # Training ends March 15
    test_start='20250301',     # Testing starts March 1 (overlap!)
    test_end='20250430'
)
# Proceeds with warning: train_end > test_start
# Results will be biased (lookahead contamination possible)
```

---

## Configuration Tuning Guide

### For Speed (< 5 minutes)

```python
feature_config='base_features'          # 6 features (vs 147)
per_player_models=False                 # Single model
recalibrate_days=14                     # Longer training window
n_jobs=1                                # Single worker
save_models=False                       # Skip I/O
save_predictions=False
minutes_threshold=0                     # All players
```

### For Accuracy (better results, slower)

```python
feature_config='default_features'       # Full 147 features
per_player_models=True                  # 500+ models (SLOW!)
recalibrate_days=3                      # Frequent retraining
n_jobs=2 or 4                          # Parallelization (if RAM available)
save_models=True
save_predictions=True
minutes_threshold=10                    # Filter bench warmers
```

### Balanced (default)

```python
feature_config='default_features'       # Good coverage
per_player_models=False                 # Fast baseline
recalibrate_days=7                      # Weekly
n_jobs=1                                # Single worker (safe)
save_models=False
save_predictions=True
minutes_threshold=10
```

---

## Data Prerequisites

Before running any backtest, verify data collection:

```bash
# Collect current season data (run once)
python scripts/collect_games.py --start-date 20241022 --end-date 20250413
python scripts/collect_dfs_salaries.py --start-date 20241022 --end-date 20250413

# Verify collection succeeded
ls data/inputs/box_scores/ | wc -l
# Should see ~150+ files (one per game)

ls data/inputs/dfs_salaries/ | wc -l
# Should see many files
```

---

## Recommended Workflow

### Step 1: Verify Setup (1 minute)

```python
from src.walk_forward_backtest import WalkForwardBacktest

# Single day test
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20250205',
    train_end='20250206',
    test_start='20250207',
    test_end='20250208',
    feature_config='base_features',
    per_player_models=False,
    minutes_threshold=0,
    n_jobs=1,
    save_models=False,
    save_predictions=False
)

results = backtest.run()
if 'error' in results:
    print(f"Setup failed: {results['error']}")
else:
    print("✓ Setup successful")
```

### Step 2: Quick Validation (5 minutes)

```python
# Use Configuration 2 (5 days)
# Verify MAPE ~70-80%, Correlation ~0.7

results = backtest.run()
print(f"MAPE: {results['mean_mape']:.1f}%")
print(f"Correlation: {results['overall_correlation']:.3f}")
```

### Step 3: Full Season (30-60 minutes)

```python
# Use Configuration 1 (full season)
# Run overnight or when computer is idle
# Monitor logs for issues
```

---

## Troubleshooting by Error Message

| Error | Cause | Solution |
|-------|-------|----------|
| `train_start must be < train_end` | Date order backwards | Swap dates |
| `No training data found for range X to Y` | No data collected | Run collect scripts |
| `gameDate column missing` | Database schema issue | Verify database integrity |
| `No slate dates found` | No games in test range | Check test date range |
| `MemoryError` | Too many models in memory | Use base_features, per_player_models=False |
| `KeyError: 'playerID'` | Missing column in storage | Verify data collection |

---

## Expected Results

### Typical Metrics (Should See)

```
MAPE: 70-85%           (lower is better)
RMSE: 11-14            (lower is better)
MAE: 7-9               (lower is better)
Correlation: 0.70-0.75 (higher is better)
Coverage: 95%+         (% of players with models)
```

### Bad Metrics (Investigate)

```
MAPE: > 100%           (model not learning)
Correlation: < 0.5     (no signal)
Coverage: < 50%        (most players filtered)
All predictions same   (model broke)
```

---

## Reference

- [Backtest Troubleshooting](./BACKTEST_TROUBLESHOOTING.md)
- [Backtest Fixes Summary](./BACKTEST_FIXES_SUMMARY.md)
- [Walk Forward Backtest](../src/walk_forward_backtest.py)
