# Checkpoint and Resume Functionality

Incremental checkpoint saving and resume capability for walk-forward backtests.

## Overview

The backtest framework now saves progress after each slate completion, allowing you to resume interrupted runs without losing work. Critical for long-running Colab backtests that may disconnect.

## Features

### Checkpoint Saving
- After each slate completes, checkpoint saved to `outputs/{timestamp}/checkpoints/{date}.json`
- Includes daily results (MAPE, RMSE, MAE, correlation, player counts)
- Predictions and actuals saved as parquet files
- Progress metadata tracked in `progress.json`

### Automatic Resume
- Set `resume_from_run` parameter to existing run timestamp
- Framework detects completed slates from checkpoint directory
- Skips completed slates automatically
- Loads existing results and predictions from parquet files
- Continues from where it left off

### Progress Tracking
- `progress.json` file maintains:
  - Run timestamp
  - Test period (start/end dates)
  - List of completed slate dates
  - Total completed count
  - Last update timestamp

## File Structure

```
data/
└── outputs/
    └── {timestamp}/
        ├── checkpoints/
        │   ├── progress.json              # Overall progress tracking
        │   ├── 20250205.json              # Per-slate checkpoint
        │   ├── 20250206.json
        │   └── ...
        ├── predictions/
        │   ├── 20250205.parquet           # Predictions only
        │   ├── 20250205_with_actuals.parquet  # Predictions + actuals
        │   └── ...
        ├── inputs/
        │   └── player_{name}_{id}_inputs.parquet
        └── features/
```

## Usage

### Colab Notebook

1. **Check existing runs** (cell 8a):
```python
# Lists all existing runs with progress
# Shows completed slates and timestamps
```

2. **Set resume parameter** (cell 7):
```python
RESUME_FROM_RUN = '20250205_143022'  # Or None for fresh start
```

3. **Initialize backtest** (cell 9):
```python
backtest = WalkForwardBacktest(
    ...,
    resume_from_run=RESUME_FROM_RUN
)
```

4. **Run backtest** (cell 10):
- Automatically skips completed slates
- Loads existing results from checkpoints
- Continues from first incomplete slate

### Python Script

```python
from src.walk_forward_backtest import WalkForwardBacktest

# Fresh run
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20241001',
    train_end='20250204',
    test_start='20250205',
    test_end='20250430',
    resume_from_run=None  # Fresh start
)

# Resume run
backtest = WalkForwardBacktest(
    db_path='nba_dfs.db',
    train_start='20241001',
    train_end='20250204',
    test_start='20250205',
    test_end='20250430',
    resume_from_run='20250205_143022'  # Resume from existing run
)

results = backtest.run()
```

## Implementation Details

### Checkpoint Data Structure

**Per-slate checkpoint** (`{date}.json`):
```json
{
  "test_date": "20250205",
  "completed_at": "2025-10-14T15:23:45.123456",
  "daily_result": {
    "date": "20250205",
    "num_players": 145,
    "model_mape": 82.5,
    "model_rmse": 12.3,
    "model_mae": 8.9,
    "model_corr": 0.728,
    "benchmark_mape": 95.2,
    "benchmark_rmse": 14.1,
    "mean_projected": 25.4,
    "mean_actual": 26.1,
    "mean_benchmark": 24.8
  },
  "num_players": 145,
  "model_mape": 82.5,
  "benchmark_mape": 95.2
}
```

**Progress tracking** (`progress.json`):
```json
{
  "run_timestamp": "20250205_143022",
  "test_start": "20250205",
  "test_end": "20250430",
  "completed_slates": ["20250205", "20250206", "20250207"],
  "total_completed": 3,
  "last_updated": "2025-10-14T15:23:45.123456"
}
```

### Code Changes

**WalkForwardBacktest class** (`src/walk_forward_backtest.py`):

1. Added `resume_from_run` parameter to `__init__`
2. Modified `run()` to:
   - Use existing timestamp if resuming
   - Create output directories in data_dir if specified
   - Load completed slates from checkpoints
   - Skip completed slates in main loop
   - Load existing results from parquet files
3. Added checkpoint methods:
   - `_save_slate_checkpoint()`: Save after each slate
   - `_load_checkpoint()`: Load completed slate dates
   - `_load_slate_checkpoint()`: Load specific slate results

**Key code sections**:

```python
# Detect resume mode
if self.resume_from_run:
    self.run_timestamp = self.resume_from_run
    logger.info(f"RESUMING existing run: {self.run_timestamp}")
else:
    self.run_timestamp = dt.now().strftime('%Y%m%d_%H%M%S')

# Load completed slates
completed_slates = self._load_checkpoint()
if completed_slates:
    logger.info(f"RESUMING from checkpoint: {len(completed_slates)} slates completed")

# Skip completed slates
for i, test_date in enumerate(slate_dates):
    if test_date in completed_slates:
        logger.info(f"Skipping slate {i+1}: {test_date} (already completed)")
        # Load existing results
        slate_result = self._load_slate_checkpoint(test_date)
        merged_df = pd.read_parquet(predictions_path)
        self.results.append(slate_result['daily_result'])
        self.all_predictions.append(merged_df)
        continue

    # Process slate normally...

    # Save checkpoint after completion
    self._save_slate_checkpoint(test_date, daily_results, merged_df)
```

## Benefits

1. **Fault Tolerance**: Colab disconnections no longer lose hours of work
2. **Incremental Progress**: See results as each slate completes
3. **Flexible Scheduling**: Run partial backtests, stop, resume later
4. **Data Persistence**: All predictions and models saved incrementally
5. **Debugging**: Easy to inspect intermediate results without waiting for full run

## Limitations

- Resume must use same configuration (model params, feature config, etc.)
- Changing test date range requires fresh run
- Models are not checkpointed (re-trained on resume if recalibration needed)
- Benchmark must be re-initialized on resume

## Future Enhancements

- Save benchmark state to checkpoints
- Checkpoint model cache for faster resume
- Validate configuration matches on resume
- Support changing test end date on resume
- Add checkpoint cleanup utilities
