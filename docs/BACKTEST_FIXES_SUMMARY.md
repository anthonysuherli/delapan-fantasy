# Backtest Fixes Summary

## Issues Fixed (Commit a345a4f)

### Problem 1: KeyError 'gameDate' on Empty Training Data

**Symptom:**
```
KeyError: 'gameDate'
File "src/walk_forward_backtest.py", line 490
training_data_sorted['gameDate'] = pd.to_datetime(...)
```

**Root Cause:** When `load_historical_player_logs()` returns empty DataFrame (no data for date range), the code still tries to access 'gameDate' column.

**Fix:** Added defensive check before column access:
```python
if training_data_full.empty:
    logger.error(f"No training data available for date range {self.train_start} to {self.train_end}")
    return {'error': f'No training data found for range {self.train_start} to {self.train_end}'}

if 'gameDate' not in training_data_sorted.columns:
    logger.error("gameDate column missing from training data")
    return {'error': 'gameDate column missing from training data'}
```

### Problem 2: Inverted Date Ranges Silently Fail

**Symptom:**
```
WARNING: No historical data found for date range 20241001 to 20240201
```

Root issue: User provided `train_start='20241001'` (October 2024) and `train_end='20240201'` (February 2024), which is backwards.

**Fix:** Added date range validation in `__init__`:
```python
if train_start >= train_end:
    raise ValueError(f"train_start ({train_start}) must be < train_end ({train_end})")
if test_start >= test_end:
    raise ValueError(f"test_start ({test_start}) must be < test_end ({test_end})")
```

Now fails immediately with clear message instead of silently producing no data.

### Problem 3: Column Access Without Null Checks

**Locations fixed:**
1. Benchmark initialization (`line 488-498`)
2. Per-player model training (`line 59-67`)

Both now check for 'gameDate' column existence before accessing.

---

## Changes Made

### File: src/walk_forward_backtest.py

**Location 1 - __init__ (line 169-175):**
- Added date range validation
- Warns if train_end > test_start (training/test overlap)

**Location 2 - Benchmark initialization (line 488-501):**
- Added empty DataFrame check
- Added gameDate column existence check
- Returns clear error message instead of KeyError

**Location 3 - _train_single_player_model (line 62-64):**
- Added gameDate column check
- Returns None gracefully if missing

### File: docs/BACKTEST_TROUBLESHOOTING.md (NEW)

Created comprehensive troubleshooting guide covering:
1. KeyError: 'gameDate' - root causes and solutions
2. Inverted date ranges - detection and fixing
3. No data for date range - validation and collection steps
4. Panel interface empty results - diagnosis steps
5. Out of memory errors - reduction strategies
6. Wrong results - feature pipeline validation
7. Quick validation checklist
8. Example correct configuration

---

## How to Use the Fixes

### For Users:

1. **Check date range is correct:**
   ```python
   # Correct
   train_start = "20250101"  # Earlier date
   train_end = "20250228"    # Later date
   test_start = "20250301"
   test_end = "20250430"

   # NOT this (backwards)
   train_start = "20250228"
   train_end = "20250101"    # ← Will now fail with clear error
   ```

2. **Verify data exists for date range:**
   ```bash
   # Check if data collected
   ls data/inputs/box_scores/ | head
   # Should see: box_scores_20250101.parquet, etc.
   ```

3. **If backtest fails, check logs:**
   - Look for ERROR messages (not just WARNING)
   - Error message now clearly states problem:
     - "No training data found for range X to Y"
     - "gameDate column missing from training data"
     - "train_start must be < train_end"

### For Developers:

The defensive checks follow defensive programming principles:

1. **Fail fast** - Errors at initialization instead of deep in execution
2. **Fail clearly** - Error messages state the problem and date ranges involved
3. **Prevent cascading failures** - Early return prevents downstream KeyErrors
4. **Log context** - All failures logged at ERROR level for debugging

---

## Testing the Fixes

### Test 1: Inverted Dates

```python
from src.walk_forward_backtest import WalkForwardBacktest

try:
    backtest = WalkForwardBacktest(
        db_path='nba_dfs.db',
        train_start='20250228',    # Later date
        train_end='20250101',      # Earlier date ← Should fail
        test_start='20250301',
        test_end='20250430'
    )
except ValueError as e:
    print(e)  # "train_start (20250228) must be < train_end (20250101)"
```

**Expected:** Fails immediately with clear message.

### Test 2: No Data

```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20250515',    # May 15 (offseason, no games)
    train_end='20250830',      # August 30
    test_start='20250901',
    test_end='20250930'
)

results = backtest.run()
# Returns: {'error': 'No training data found for range 20250515 to 20250830'}
```

**Expected:** Clear error about missing data, not KeyError.

### Test 3: Valid Configuration

```python
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20250101',
    train_end='20250228',
    test_start='20250301',
    test_end='20250430'
)

results = backtest.run()
# Runs successfully if data exists
```

**Expected:** Runs normally.

---

## Performance Impact

**Minimal** - Added checks only:
- 2 string comparisons on initialization
- 1 empty check before DataFrame operations
- 1 column name check (O(1) hash table lookup)

All checks happen before expensive data loading/model training, so early failures are fast.

---

## Migration Notes

### Breaking Changes
**None** - This is backward compatible.

### New Error Messages
Users may see these new error messages (which are good - they clarify problems):
- `ValueError: train_start (X) must be < train_end (Y)` - Date order error
- `Error: No training data found for range X to Y` - Missing data error
- `Error: gameDate column missing from training data` - Schema error

### Deprecation Notes
**None** - No APIs changed.

---

## Next Steps

1. Run backtest with valid dates to verify fixes don't break normal operation
2. Test with Panel interface to verify error messages surface clearly
3. Add these date range validations to any other date-based operations
4. Consider adding similar checks to data collection scripts

---

## References

- [Backtest Troubleshooting Guide](./BACKTEST_TROUBLESHOOTING.md)
- [Walk Forward Backtest Implementation](../src/walk_forward_backtest.py)
- Git commit: a345a4f
