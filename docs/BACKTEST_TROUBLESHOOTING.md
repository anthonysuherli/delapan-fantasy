# Backtest Troubleshooting Guide

Common issues and solutions when running walk-forward backtests.

## Issue 1: KeyError: 'gameDate'

### Error Message
```
KeyError: 'gameDate'
File "src/walk_forward_backtest.py", line 490, in run
  training_data_sorted['gameDate'] = pd.to_datetime(training_data_sorted['gameDate'], ...)
```

### Root Causes

1. **Empty training data**: Date range has no available historical data
2. **Data not collected**: Box score data not downloaded for date range
3. **Column name mismatch**: Storage returns different column names than expected

### Solutions

**Step 1: Verify data exists**

```bash
# Check if box scores are stored
ls data/inputs/box_scores/

# Should see files like: box_scores_20250205.parquet

# If empty, collect data first:
python scripts/collect_games.py --start-date 20250101 --end-date 20250228
```

**Step 2: Validate date ranges**

```python
# Dates must be in YYYYMMDD format
train_start = "20250101"  # Correct: January 1, 2025
train_end = "20250228"    # Correct: February 28, 2025

# NOT this:
train_start = "20240201"  # Wrong: February 1, 2024
train_end = "20241001"    # Wrong: backwards (earlier than start)
```

**Step 3: Check data schema**

```python
import pandas as pd
from src.data.storage.sqlite_storage import SQLiteStorage

storage = SQLiteStorage('nba_dfs.db')
df = storage.load('box_scores', {'start_date': '20250101', 'end_date': '20250110'})

print(df.columns)  # Should include 'gameDate'
print(df.head())
```

### Fix Applied (v1.1+)

The backtest now includes defensive checks:

```python
if training_data_full.empty:
    logger.error(f"No training data available for date range {self.train_start} to {self.train_end}")
    return {'error': f'No training data found for range {self.train_start} to {self.train_end}'}

if 'gameDate' not in training_data_sorted.columns:
    logger.error("gameDate column missing from training data")
    return {'error': 'gameDate column missing from training data'}
```

This provides clear error messages instead of cryptic KeyError.

---

## Issue 2: Inverted Date Ranges

### Error Message
```
WARNING: No historical data found for date range 20241001 to 20240201
```

### Root Cause

Date range is backwards: start_date > end_date (October 2024 → February 2024)

### Solutions

**Correct date order:**
```python
train_start = "20240101"   # January 1, 2024
train_end = "20240630"     # June 30, 2024 (later than start)
test_start = "20240701"    # July 1, 2024 (after training ends)
test_end = "20240731"      # July 31, 2024
```

**Do NOT use:**
```python
train_start = "20241001"   # October 2024
train_end = "20240201"     # February 2024 (BEFORE October!)
```

### Fix Applied (v1.1+)

Date range validation now catches this:

```python
if train_start >= train_end:
    raise ValueError(f"train_start ({train_start}) must be < train_end ({train_end})")
if test_start >= test_end:
    raise ValueError(f"test_start ({test_start}) must be < test_end ({test_end})")
```

The backtest will fail immediately with clear message instead of silently producing no results.

---

## Issue 3: No Data for Date Range

### Error Message
```
WARNING: No historical data found for date range 20250101 to 20250228
```

### Root Causes

1. **Data not collected** for that date range
2. **Date range outside NBA season** (e.g., July-September)
3. **Database path incorrect** (connecting to wrong database)

### Solutions

**Step 1: Check NBA season dates**

```python
# NBA regular season typically:
# October 2024 - April 2025 (2024-25 season)
# October 2025 - April 2026 (2025-26 season)

# Preseason starts late September
# Offseason: May-September (no games)
```

**Valid date ranges:**
```python
# Current season (2024-25)
train_start = "20241025"   # October 25, 2024
train_end = "20250131"     # January 31, 2025

# Offseason (no games) - INVALID
train_start = "20250515"   # May 15, 2025 - NO GAMES
train_end = "20250831"     # August 31, 2025
```

**Step 2: Collect missing data**

```bash
# Collect all games for season
python scripts/collect_games.py --start-date 20241025 --end-date 20250430

# Collect DFS salaries
python scripts/collect_dfs_salaries.py --start-date 20241025 --end-date 20250430

# Verify collection
ls data/inputs/box_scores/ | wc -l  # Should have many files
```

**Step 3: Verify database connection**

```python
from src.data.storage.sqlite_storage import SQLiteStorage

# Check database exists and is readable
storage = SQLiteStorage('nba_dfs.db')

# Try loading data
df = storage.load('box_scores', {'start_date': '20250101', 'end_date': '20250110'})
print(f"Loaded {len(df)} records")
print(df.columns)
```

---

## Issue 4: Panel Interface Shows Empty Results

### Error Message
```
Results Panel: No results available
Training Input Sample: No training data found
```

### Root Causes

1. Backtest crashed (check logs at bottom)
2. Date ranges produce no slate games
3. Feature pipeline failed silently
4. All players filtered out by minutes_threshold

### Solutions

**Step 1: Check execution logs**

Scroll to bottom of Panel interface. Look for:
- `ERROR: ...` lines indicate failures
- `WARNING: No historical data found ...` indicates no data
- `INFO: ...` shows progress

**Step 2: Verify minutes threshold**

```python
# Too high: filters all players
minutes_threshold = 48  # Only plays entire game - very restrictive

# Reasonable: includes most rotation players
minutes_threshold = 10  # Default - most players qualify

# Lenient: includes garbage time
minutes_threshold = 0   # Anyone with game log entry
```

**Step 3: Test with smaller date range**

```python
# If season-long backtest fails, test with single day
test_start = "20250205"   # Single day
test_end = "20250206"     # Same day + 1
```

**Step 4: Load training sample manually**

```bash
# From Python REPL
from src.walk_forward_backtest import WalkForwardBacktest

backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20250101',
    train_end='20250228',
    test_start='20250301',
    test_end='20250305'
)

# Load training data
training_data = backtest.loader.load_historical_player_logs(
    start_date='20250101',
    end_date='20250228'
)

print(f"Training data shape: {training_data.shape}")
print(f"Columns: {training_data.columns.tolist()}")
print(training_data.head())
```

---

## Issue 5: Out of Memory (OOM)

### Error Message
```
MemoryError: Unable to allocate X.XX GiB for array...
```

### Root Causes

1. Too many per-player models (500+ players × large feature set)
2. Full season backtest without recalibration
3. Parallel jobs (`n_jobs`) too high
4. Feature pipeline creates duplicate columns

### Solutions

**Step 1: Use per-slate models instead**

```python
# Instead of:
per_player_models = True   # 500+ models in memory

# Use:
per_player_models = False  # Single model, much faster
```

**Step 2: Reduce parallel jobs**

```python
# Instead of:
n_jobs = 8    # 8 parallel workers × large data = OOM

# Use:
n_jobs = 1    # Single worker, slower but no OOM
# Or:
n_jobs = 2    # Balance between speed and memory
```

**Step 3: Use smaller feature set**

```python
# Instead of:
feature_config = 'default_features'  # 147 features

# Use:
feature_config = 'base_features'     # 6 features (faster, less memory)
```

**Step 4: Recalibrate frequently**

```python
# Instead of:
recalibrate_days = 30  # Train once per month

# Use:
recalibrate_days = 7   # Train weekly (clears memory between retrains)
```

---

## Issue 6: Backtest Runs But Shows Wrong Results

### Symptoms

- MAPE > 100% (should be 70-85%)
- Correlation ≈ 0 (should be 0.7+)
- All predictions are same value
- Coverage = 0%

### Root Causes

1. Feature pipeline not fitted correctly
2. Target column missing or wrong
3. Data leakage (future data in features)
4. Model not training on sufficient data

### Diagnostics

```python
from src.walk_forward_backtest import WalkForwardBacktest
from src.features.pipeline import FeaturePipeline
from src.utils.feature_config import load_feature_config

# Load config
config = load_feature_config('default_features')
pipeline = config.build_pipeline(FeaturePipeline)

# Check pipeline
print("Transformers:", [t.__class__.__name__ for t in pipeline.transformers])

# Fit and transform sample data
sample_df = backtest.loader.load_historical_player_logs('20250228', lookback_days=30)
features = pipeline.fit_transform(sample_df)

# Check results
print(f"Input shape: {sample_df.shape}")
print(f"Output shape: {features.shape}")
print(f"Missing target: {features['target'].isna().sum()}")
print(f"Feature columns: {[c for c in features.columns if c.startswith(('rolling_', 'ewma_'))]}")
```

---

## Quick Validation Checklist

Before running a backtest:

- [ ] Date ranges in correct order (start < end)
- [ ] Dates within NBA season (Oct-Apr)
- [ ] Data collected for full date range
- [ ] Database file exists and is readable
- [ ] Train/test windows don't overlap (optional warning)
- [ ] Minutes threshold reasonable (0-12 typical)
- [ ] Per-player models off (faster, less memory)
- [ ] Recalibrate cadence set (7-14 days)
- [ ] Feature config file exists
- [ ] Output directory writable

Example validated config:

```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20250101',      # ✓ Jan 1
    train_end='20250228',        # ✓ Feb 28 (after Jan)
    test_start='20250301',       # ✓ Mar 1 (after training)
    test_end='20250430',         # ✓ Apr 30 (end of season)
    model_type='xgboost',
    feature_config='default_features',
    output_dir='data/backtest_results',
    per_player_models=False,     # ✓ Faster
    recalibrate_days=7,          # ✓ Weekly
    minutes_threshold=10,        # ✓ Reasonable
    n_jobs=1,                    # ✓ Single worker
    save_models=False,           # Reduce I/O
    save_predictions=True        # Keep results
)

results = backtest.run()
```

---

## Getting Help

1. Check execution logs for ERROR or WARNING messages
2. Verify date ranges with NBA calendar
3. Confirm data collected with `ls data/inputs/`
4. Test with single-day backtest (`test_start` = `test_end - 1 day`)
5. Reduce scope: smaller date range, fewer features, no parallelization
