# Notebook Output Routing Configuration

## Overview

Both backtest notebooks have been updated to route all outputs to a unified `data/outputs` directory structure, ensuring consistency across execution environments (local, cloud, and Colab).

## Changes Made

### 1. backtest_season.ipynb (Local Execution)

**File**: `notebooks/backtest_season.ipynb`

#### Configuration Change (Cell 7)
```python
# OLD
OUTPUT_DIR = str(repo_root / 'data' / 'backtest_results')

# NEW
OUTPUT_DIR = str(repo_root / 'data' / 'outputs')
```

#### Save Results Logic (Cell 28)
```python
# OLD (OUTPUT_DIR was a Path object)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUTPUT_DIR / f'backtest_results_{TEST_START}_to_{TEST_END}.csv'
summary_path = OUTPUT_DIR / f'summary_{TEST_START}_to_{TEST_END}.txt'

# NEW (OUTPUT_DIR is a string, wrapped with Path())
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
csv_path = Path(OUTPUT_DIR) / f'backtest_results_{TEST_START}_to_{TEST_END}.csv'
summary_path = Path(OUTPUT_DIR) / f'summary_{TEST_START}_to_{TEST_END}.txt'
```

#### Output Directory Structure
```
data/
└── outputs/
    ├── backtest_results_{YYYYMMDD}_to_{YYYYMMDD}.csv
    ├── summary_{YYYYMMDD}_to_{YYYYMMDD}.txt
    └── {TIMESTAMP}/                          # Via WalkForwardBacktest
        ├── predictions/
        │   ├── {date}.parquet
        │   └── {date}_with_actuals.parquet
        ├── inputs/
        ├── checkpoints/
        ├── features/
        ├── performance_report.txt
        └── performance_metrics.json
```

### 2. colab_backtest.ipynb (Google Colab)

**File**: `notebooks/colab_backtest.ipynb`

#### Status: Already Correctly Configured

No changes needed. The notebook already routes outputs to:
```python
DATA_DIR = '/content/delapan-fantasy/MyDrive/dfs/data'
OUTPUT_DIR = 'outputs'
# Resolves to: {DATA_DIR}/outputs/
```

#### Output Directory Structure
```
{DATA_DIR}/outputs/
├── {TIMESTAMP}/                              # Via WalkForwardBacktest
│   ├── predictions/
│   ├── inputs/
│   ├── checkpoints/
│   ├── features/
│   ├── performance_report.txt
│   └── performance_metrics.json
├── summary_{YYYYMMDD}_to_{YYYYMMDD}.csv     # Cell 42
└── tier_comparison_{YYYYMMDD}_to_{YYYYMMDD}.csv  # Cell 42
```

**Google Drive Path** (when mounted in Colab):
- Base: `/content/delapan-fantasy/MyDrive/dfs/data/outputs/`
- On Drive: `delapan-fantasy/dfs/data/outputs/`

## File Output Summary

### CSV Exports (both notebooks)
| File | Purpose | Format |
|------|---------|--------|
| `backtest_results_{date_range}.csv` | Daily results with metrics | CSV - one row per slate |
| `summary_{date_range}.csv` | Aggregated daily results | CSV - summary statistics |
| `tier_comparison_{date_range}.csv` | Performance by salary tier | CSV - salary bin analysis |
| `summary_{date_range}.txt` | Text report (backtest_season only) | TXT - human-readable summary |

### JSON/Parquet Exports (via WalkForwardBacktest)
| File | Purpose | Type |
|------|---------|------|
| `{date}.parquet` | Per-slate predictions | Parquet |
| `{date}_with_actuals.parquet` | Predictions + actual values | Parquet |
| `performance_report.txt` | Comprehensive backtest report | TXT |
| `performance_metrics.json` | Metrics for programmatic access | JSON |
| `progress.json` | Run progress tracking | JSON |
| `{date}.json` | Per-slate checkpoint | JSON |

## Environment-Specific Paths

### Local Execution (backtest_season.ipynb)
```
Repository Root
└── data/
    └── outputs/                    # All backtest outputs here
        ├── backtest_results_*.csv
        ├── summary_*.csv/txt
        └── {TIMESTAMP}/
            ├── predictions/
            ├── inputs/
            ├── checkpoints/
            ├── features/
            ├── performance_report.txt
            └── performance_metrics.json
```

### Google Colab (colab_backtest.ipynb)
```
Google Drive (mounted at /content/delapan-fantasy)
└── MyDrive/
    └── dfs/
        └── data/
            └── outputs/            # All backtest outputs here
                ├── summary_*.csv
                ├── tier_comparison_*.csv
                └── {TIMESTAMP}/
                    ├── predictions/
                    ├── inputs/
                    ├── checkpoints/
                    ├── features/
                    ├── performance_report.txt
                    └── performance_metrics.json
```

## WalkForwardBacktest Integration

Both notebooks use `WalkForwardBacktest` which automatically creates timestamped output directories:

### Automatic Directory Creation
```python
# From src/walk_forward_backtest.py (lines 443-466)
if self.data_dir:
    # Separated architecture (Colab)
    base_output = Path(self.data_dir) / self.output_dir
    self.run_output_dir = base_output / self.run_timestamp
else:
    # Default architecture (Local)
    self.run_output_dir = Path('data') / 'outputs' / self.run_timestamp
```

### Subdirectory Auto-Creation
```python
self.run_inputs_dir = f"{self.run_output_dir}/inputs/"
self.run_features_dir = f"{self.run_output_dir}/features/"
self.run_predictions_dir = f"{self.run_output_dir}/predictions/"
self.run_checkpoint_dir = f"{self.run_output_dir}/checkpoints/"
```

## Resume Capability

Both notebooks support resuming interrupted runs:

### backtest_season.ipynb
- Set `RESUME_FROM_RUN` parameter (not currently implemented)
- Manual directory management if interrupted

### colab_backtest.ipynb
- Check existing runs with cell 8a
- Set `RESUME_FROM_RUN = '{YYYYMMDD_HHMMSS}'` in section 7
- Re-run from section 9 onwards
- Completed slates skipped automatically via `progress.json`

## Consistency Benefits

1. **Unified Output Location**: All outputs go to `data/outputs` regardless of execution environment
2. **Timestamped Organization**: Each run creates its own timestamped directory for easy identification
3. **Reproducibility**: Training inputs saved for every run
4. **Checkpointing**: Progress tracked for resume capability
5. **Scalability**: Nested directory structure prevents file conflicts across runs

## Migration Guide (if needed)

For existing code referencing old paths:

### backtest_season.ipynb
```python
# OLD
~/data/backtest_results/

# NEW
~/data/outputs/

# Update imports or path references to use:
data/outputs/{timestamp}/
```

### colab_backtest.ipynb
No changes needed - already using correct paths:
```python
{DATA_DIR}/outputs/  # Already correct
```

## File Size Expectations

Per backtest run (30 days, ~1000 predictions/day):

| Component | Size |
|-----------|------|
| Predictions (Parquet) | ~50-100 MB |
| Training inputs | ~100-200 MB |
| Checkpoints (JSON) | <1 MB |
| Models (if saved) | Varies (typically 50+ MB per player) |
| **Total** | **200-500+ MB per run** |

## Troubleshooting

### Missing data/outputs directory
```python
# Automatically created by notebooks, but can manually create:
from pathlib import Path
Path('data/outputs').mkdir(parents=True, exist_ok=True)
```

### Permission errors on Colab
Ensure Google Drive is properly mounted:
```python
from google.colab import drive
drive.mount('/content/delapan-fantasy')
```

### Outputs not saving
Check that `OUTPUT_DIR` variable is correctly set in configuration cells and that parent directories exist.
