# Google Colab Setup Guide

Complete guide for running NBA DFS per-player model training on Google Colab.

## Overview

Google Colab provides free cloud compute with:
- 2-8 CPU cores (depending on tier)
- 12-50 GB RAM
- 12-24 hour session limits
- Persistent storage via Google Drive

## Architecture

**Best Practice Setup:**
- **Code**: Cloned from GitHub to `/content/` (fast operations, always up-to-date)
- **Data**: Synced to Google Drive (persistent between sessions)
- **Outputs**: Saved to Google Drive (persistent between sessions)

This avoids git operations on Drive's FUSE filesystem and ensures fast execution.

## Pricing Comparison

| Tier | Cost | Cores | RAM | Session | Speed/Slate | Season Cost |
|------|------|-------|-----|---------|-------------|-------------|
| Free | $0 | 2 | 12GB | 12hr | 21 min | $0 |
| Pro | $10/mo | 4 | 25GB | 24hr | 10.4 min | $10/season |
| Pro+ | $50/mo | 8 | 50GB | 24hr | 5.2 min | $50/season |

**Recommended: Colab Pro ($10/month)** - Best value for regular use.

## Prerequisites

1. Google account
2. GitHub repository with your code (https://github.com/anthonysuherli/delapan-fantasy)
3. Local data collected via `scripts/collect_games.py`
4. Google Drive Desktop app (for data sync) or web interface

## Setup Instructions

### Step 1: Install Google Drive Desktop

**Windows/Mac:**
1. Download: https://www.google.com/drive/download/
2. Install and sign in
3. Drive mounts at `G:\MyDrive` (Windows) or `/Users/you/Google Drive/My Drive` (Mac)

**Verify mount:**
```bash
ls G:\MyDrive  # Windows
ls ~/Google\ Drive/My\ Drive  # Mac
```

### Step 2: Sync Data to Google Drive

**Only data files are synced. Code comes from GitHub.**

```bash
python scripts/sync_to_gdrive.py --gdrive-dir G:\MyDrive
```

This syncs:
- All parquet files from `data/inputs/box_scores/`, `dfs_salaries/`, etc.
- SQLite database `nba_dfs.db`

**What NOT to sync:**
- Source code (`src/`, `config/`, `scripts/`) - cloned from GitHub
- Notebooks - uploaded directly to Colab
- Requirements - in repository

### Step 3: Create SQLite Database (First Time Only)

Run locally:

```bash
python scripts/load_games_to_db.py
```

Database automatically synced by Step 2.

### Step 4: Upload Notebook to Colab

1. Go to https://colab.research.google.com/
2. Click **File > Upload notebook**
3. Select `notebooks/colab_backtest.ipynb`
4. Notebook opens in Colab

### Step 5: Run Backtest

1. **Mount Drive**: Run cell 1 (authorizes Drive access)
2. **Install dependencies**: Run cell 2 (1-2 minutes)
3. **Clone from GitHub**: Run cell 3 (auto-clones latest code)
4. **Verify data**: Run cell 4 (checks data files on Drive)
5. **Check resources**: Run cell 5 (shows CPU/RAM)
6. **Configure parameters**: Edit cell 6 (date ranges, model params)
7. **Run backtest**: Run cell 9 (main execution)
8. **View results**: Run cells 10-12 (summaries and visualizations)

## Configuration

### Backtest Parameters

Edit in notebook cell 6:

```python
TRAIN_START = '20241001'  # Training start date
TRAIN_END = '20241130'    # Training end date
TEST_START = '20241201'   # Test start date
TEST_END = '20241215'     # Test end date

MODEL_TYPE = 'xgboost'
FEATURE_CONFIG = 'default_features'
MIN_PLAYER_GAMES = 10
RECALIBRATE_DAYS = 7

N_JOBS = -1  # Use all available cores
```

### Model Hyperparameters

Edit in notebook cell 8:

```python
model_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

## Performance Optimization

### For Free Tier (2 cores)

1. Reduce date range:
   ```python
   TEST_START = '20241201'
   TEST_END = '20241207'  # Just one week
   ```

2. Filter high-salary players:
   Add to cell 8:
   ```python
   MIN_SALARY = 5000  # Only train models for $5k+ players
   ```

3. Increase recalibration frequency:
   ```python
   RECALIBRATE_DAYS = 14  # Train less often
   ```

### For Pro Tier (4 cores)

Keep default settings. Full season backtest completes in ~2.4 hours.

### For Pro+ Tier (8 cores)

Reduce recalibration for faster iteration:
```python
RECALIBRATE_DAYS = 3  # More frequent model updates
```

## Data Management

### Sync Updates Incrementally

After collecting new games locally, sync new data:

```bash
python scripts/sync_to_gdrive.py --gdrive-dir G:\MyDrive
```

Script automatically skips unchanged files (checks modification time).

**Skip database sync (data only):**
```bash
python scripts/sync_to_gdrive.py --gdrive-dir G:\MyDrive --no-db
```

### Update Code

Code comes from GitHub, so updates are automatic:

```python
# In Colab cell 3, code pulls latest from GitHub on each run
# No manual sync needed
```

To use a different branch:
```python
# Edit cell 3
!git clone -b your-branch https://github.com/anthonysuherli/delapan-fantasy.git
```

### Check Data Size

```python
import os
from pathlib import Path

data_dir = Path('/content/drive/MyDrive/nba_dfs/data/inputs')
total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
print(f"Total data size: {total_size / (1024**3):.2f} GB")
```

Google Drive free tier: 15 GB storage.

### Clean Up Old Results

Delete old backtest outputs to save space:

```python
import shutil
from pathlib import Path

outputs_dir = Path('/content/drive/MyDrive/nba_dfs/data/outputs')
for old_dir in outputs_dir.glob('2024*'):
    if old_dir.is_dir():
        shutil.rmtree(old_dir)
        print(f"Deleted {old_dir}")
```

## Troubleshooting

### Issue: "Drive not mounted"

**Solution:**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Issue: "Module not found"

Code should be cloned from GitHub automatically. If still failing:

```python
# In cell 3, force re-clone
!rm -rf /content/delapan-fantasy
!git clone https://github.com/anthonysuherli/delapan-fantasy.git
```

### Issue: "Git clone failed"

**Solution:** Use zip download fallback (already in cell 3):
```python
!wget https://github.com/anthonysuherli/delapan-fantasy/archive/refs/heads/main.zip
!unzip -q main.zip
!mv delapan-fantasy-main /content/delapan-fantasy
```

### Issue: "Session timeout"

Free tier disconnects after inactivity. Enable Colab Pro for background execution.

**Workaround (Free tier):**
```javascript
// Run in browser console to prevent timeout
function KeepAlive() {
  document.querySelector("colab-connect-button").click()
}
setInterval(KeepAlive, 60000)
```

### Issue: "Out of memory"

**Solutions:**
1. Reduce parallel jobs:
   ```python
   N_JOBS = 2  # Instead of -1
   ```

2. Filter players by salary:
   ```python
   MIN_SALARY = 6000
   ```

3. Upgrade to Colab Pro (25GB RAM)

### Issue: "Training too slow"

Check allocated resources:
```python
import psutil
print(f"CPU cores: {psutil.cpu_count()}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
```

If showing 2 cores on free tier, consider upgrading to Colab Pro.

### Issue: "Database locked"

SQLite has concurrency issues on Drive's FUSE filesystem. Copy to local Colab storage:

```python
# Add to cell 6 before configuration
!cp /content/drive/MyDrive/nba_dfs/nba_dfs.db /content/nba_dfs.db
DB_PATH = '/content/nba_dfs.db'  # Use local copy
```

This gives 10-20x faster database operations.

## Advanced Usage

### Run Multiple Experiments

Create parameter sweep:

```python
configs = [
    {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100},
    {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 200},
    {'max_depth': 8, 'learning_rate': 0.03, 'n_estimators': 300},
]

for i, params in enumerate(configs):
    print(f"\nExperiment {i+1}/{len(configs)}")
    backtest = WalkForwardBacktest(
        model_params=params,
        output_dir=f'/content/drive/MyDrive/nba_dfs/outputs/exp_{i+1}',
        ...
    )
    results = backtest.run()
```

### Automated Scheduling

Use Google Apps Script to run weekly:

```javascript
function runBacktest() {
  var notebook = DriveApp.getFileById('YOUR_NOTEBOOK_ID');
  // Trigger via Colab API or manual execution
}

function createTrigger() {
  ScriptApp.newTrigger('runBacktest')
    .timeBased()
    .everyWeeks(1)
    .onWeekDay(ScriptApp.WeekDay.MONDAY)
    .create();
}
```

### Export to Kaggle Notebooks

Similar workflow works on Kaggle:
1. Upload data to Kaggle Dataset
2. Create Kaggle Notebook
3. Attach dataset
4. Run backtest

Kaggle provides 30 hours/week GPU time (free).

## Best Practices

### 1. Use Checkpoints

Save intermediate results:

```python
# After each slate
results_so_far = {
    'daily_results': backtest.results,
    'all_predictions': backtest.all_predictions
}
import pickle
with open(f'/content/drive/MyDrive/nba_dfs/checkpoint_{test_date}.pkl', 'wb') as f:
    pickle.dump(results_so_far, f)
```

### 2. Monitor Progress

Add progress tracking:

```python
from tqdm.notebook import tqdm

for test_date in tqdm(slate_dates, desc="Processing slates"):
    # Backtest code here
    pass
```

### 3. Verify Outputs

Check results exist:

```python
predictions_dir = Path('/content/drive/MyDrive/nba_dfs/data/outputs/.../predictions')
assert len(list(predictions_dir.glob('*.parquet'))) > 0, "No predictions saved!"
```

### 4. Clean Runtime

Restart runtime between long runs:
```python
# At end of notebook
import os
os.kill(os.getpid(), 9)
```

## Comparison with Other Options

| Feature | Local | Colab Free | Colab Pro | SageMaker |
|---------|-------|------------|-----------|-----------|
| Cost | $0 | $0 | $10/mo | $44/season |
| Speed/slate | 5.2 min | 21 min | 10.4 min | 70 min |
| Machine free? | No | Yes | Yes | Yes |
| Setup time | 0 min | 10 min | 10 min | 60 min |
| Max runtime | Unlimited | 12 hrs | 24 hrs | Unlimited |
| Reliability | High | Medium | High | High |

## Support Resources

- Colab FAQ: https://research.google.com/colaboratory/faq.html
- Colab Pro features: https://colab.research.google.com/signup
- Community: https://stackoverflow.com/questions/tagged/google-colaboratory

## Quick Reference

### Essential Commands

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone from GitHub
!git clone https://github.com/anthonysuherli/delapan-fantasy.git

# Update code from GitHub
%cd /content/delapan-fantasy
!git pull
%cd /content

# Check resources
import psutil
print(f"Cores: {psutil.cpu_count()}, RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")

# Check data location
!ls /content/drive/MyDrive/nba_dfs/data/inputs

# Download results
from google.colab import files
files.download('/content/drive/MyDrive/nba_dfs/results.csv')

# Clear output
from IPython.display import clear_output
clear_output(wait=True)

# Install package
!pip install -q package_name
```

### Keyboard Shortcuts

- `Ctrl+Enter`: Run cell
- `Shift+Enter`: Run cell and move to next
- `Ctrl+M B`: Insert cell below
- `Ctrl+M D`: Delete cell
- `Ctrl+S`: Save notebook

## Next Steps

1. Complete setup (Steps 1-4)
2. Run notebook cell by cell
3. Verify results saved to Drive
4. Download summary CSV for analysis
5. Iterate with different hyperparameters

For production deployment, see `docs/SAGEMAKER_DEPLOYMENT.md`.
