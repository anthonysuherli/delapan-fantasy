# Local Separated Architecture Setup

Guide for running NBA DFS training with code and data in separate locations.

## Overview

Separate code from data for cleaner organization and flexibility:

**Structure:**
```
C:\Code\
  └── delapan-fantasy\              # Code (from git)
      ├── src\
      ├── config\
      ├── scripts\
      └── ...

D:\NBA_Data\                         # Data (large files)
  ├── nba_dfs.db                     # Database
  ├── data\
  │   ├── inputs\
  │   │   ├── box_scores\           # Parquet files
  │   │   ├── dfs_salaries\
  │   │   └── ...
  │   └── outputs\                   # Results
  └── models\                        # Trained models
```

## Benefits

1. **Clean git repository** - no large data files
2. **Flexible storage** - data on different drive (SSD/HDD)
3. **Easy backup** - separate code and data backup strategies
4. **Multiple projects** - share data across code branches
5. **Portability** - clone code anywhere, point to data

## Setup Instructions

### 1. Create Data Directory

```bash
# Create data directory structure
mkdir D:\NBA_Data
mkdir D:\NBA_Data\data
mkdir D:\NBA_Data\data\inputs
mkdir D:\NBA_Data\data\outputs
mkdir D:\NBA_Data\models
```

### 2. Collect Data

Run data collection scripts (code stays in project directory):

```bash
cd C:\Code\delapan-fantasy

# Collect games
python scripts\collect_games.py --start-date 20241001 --end-date 20241231

# Collect salaries
python scripts\collect_dfs_salaries.py --start-date 20241001 --end-date 20241231
```

### 3. Move Data to Separate Location

**Option A: Move after collection**

```bash
# Move parquet files
move data\inputs\* D:\NBA_Data\data\inputs\

# Create database
python scripts\load_games_to_db.py
move nba_dfs.db D:\NBA_Data\
```

**Option B: Symlink (Windows requires admin)**

```cmd
mklink /D C:\Code\delapan-fantasy\data D:\NBA_Data\data
mklink C:\Code\delapan-fantasy\nba_dfs.db D:\NBA_Data\nba_dfs.db
```

**Option C: Collect directly to data directory**

Modify storage initialization in collection scripts:

```python
from src.data.storage.parquet_storage import ParquetStorage

storage = ParquetStorage(base_dir='D:/NBA_Data/data/inputs')
```

### 4. Run Backtest with Data Directory

```bash
cd C:\Code\delapan-fantasy

python scripts\run_backtest.py \
  --data-dir D:\NBA_Data \
  --test-start 20241201 \
  --test-end 20241215 \
  --per-player \
  --n-jobs -1
```

**What happens:**
- Code runs from `C:\Code\delapan-fantasy\`
- Database loaded from `D:\NBA_Data\nba_dfs.db`
- Data read from `D:\NBA_Data\data\inputs\`
- Results saved to `D:\NBA_Data\data\outputs\`
- Models saved to `D:\NBA_Data\models\`

## Usage Examples

### Basic Backtest

```bash
python scripts\run_backtest.py \
  --data-dir D:\NBA_Data \
  --test-start 20241201 \
  --test-end 20241215
```

### Per-Player Models with Parallel Training

```bash
python scripts\run_backtest.py \
  --data-dir D:\NBA_Data \
  --test-start 20241201 \
  --test-end 20241215 \
  --per-player \
  --n-jobs -1
```

### Custom Database Location

```bash
python scripts\run_backtest.py \
  --data-dir D:\NBA_Data \
  --db-path custom_db.db \
  --test-start 20241201 \
  --test-end 20241215
```

Database loaded from: `D:\NBA_Data\custom_db.db`

### Absolute Paths (Override data-dir)

```bash
python scripts\run_backtest.py \
  --db-path E:\Databases\nba_dfs.db \
  --output-dir E:\Results \
  --test-start 20241201 \
  --test-end 20241215
```

Absolute paths ignore `--data-dir`.

## Configuration Options

### Path Resolution

When `--data-dir` is set:

- **Relative paths** resolved relative to data-dir:
  - `--db-path nba_dfs.db` → `D:\NBA_Data\nba_dfs.db`
  - `--output-dir data/outputs` → `D:\NBA_Data\data\outputs`

- **Absolute paths** used as-is:
  - `--db-path E:\nba_dfs.db` → `E:\nba_dfs.db`
  - `--output-dir E:\Results` → `E:\Results`

### Environment Variables

Create `.env` file:

```bash
# In project root
NBA_DATA_DIR=D:\NBA_Data
```

Update scripts to use:

```python
from dotenv import load_dotenv
import os

load_dotenv()
data_dir = os.getenv('NBA_DATA_DIR')
```

## Development Workflow

### Daily Workflow

1. **Collect new data:**
   ```bash
   python scripts\collect_games.py --start-date 20250101 --end-date 20250101
   python scripts\load_games_to_db.py
   ```

2. **Run backtest:**
   ```bash
   python scripts\run_backtest.py --data-dir D:\NBA_Data --test-start 20250101 --test-end 20250101
   ```

3. **Check results:**
   ```bash
   ls D:\NBA_Data\data\outputs\
   ```

### Multi-Branch Development

Use same data across code branches:

```bash
# Branch 1: Feature development
cd C:\Code\delapan-fantasy
git checkout feature-branch
python scripts\run_backtest.py --data-dir D:\NBA_Data --test-start 20250101 --test-end 20250105

# Branch 2: Baseline
cd C:\Code\delapan-fantasy
git checkout main
python scripts\run_backtest.py --data-dir D:\NBA_Data --test-start 20250101 --test-end 20250105

# Compare results
python scripts\compare_runs.py D:\NBA_Data\data\outputs\run1 D:\NBA_Data\data\outputs\run2
```

### Git Integration

`.gitignore` updated to exclude data:

```gitignore
# Data (stored separately)
data/inputs/
data/outputs/
nba_dfs.db
*.parquet

# Models (stored separately)
models/
```

Code stays clean, data excluded from version control.

## Storage Recommendations

### SSD vs HDD

**Code (SSD recommended):**
- Fast git operations
- Quick IDE indexing
- Instant script execution

**Data:**
- **Inputs (HDD acceptable):** Parquet files read sequentially
- **Database (SSD preferred):** Random access, benefits from SSD
- **Outputs (HDD acceptable):** Write-once, read rarely
- **Models (SSD preferred):** Frequent loading during backtest

### Optimal Setup

```
C:\ (SSD)
  └── Code\
      └── delapan-fantasy\

D:\ (SSD)
  └── NBA_Data\
      ├── nba_dfs.db
      └── models\

E:\ (HDD)
  └── NBA_Archive\
      ├── data\
      └── old_outputs\
```

**Workflow:**
- Active work: D:\ (SSD)
- Archive: E:\ (HDD)

## Troubleshooting

### Issue: "Database not found"

**Solution:** Check path resolution:

```bash
python scripts\run_backtest.py --data-dir D:\NBA_Data --db-path nba_dfs.db --verbose
```

Look for log: `Database: D:\NBA_Data\nba_dfs.db`

### Issue: "Permission denied"

**Solution:** Check directory permissions:

```bash
icacls D:\NBA_Data
```

Grant full control:

```cmd
icacls D:\NBA_Data /grant %USERNAME%:F /T
```

### Issue: "Output directory not found"

**Solution:** Directories auto-created. Check parent exists:

```bash
mkdir D:\NBA_Data\data
```

### Issue: Performance slow

**Database on HDD?** Move to SSD:

```bash
move D:\NBA_Data\nba_dfs.db C:\NBA_Data_Fast\nba_dfs.db
python scripts\run_backtest.py --data-dir C:\NBA_Data_Fast --test-start 20250101 --test-end 20250105
```

## Comparison: Architectures

| Aspect | Default (Integrated) | Separated |
|--------|---------------------|-----------|
| Setup | Simple | One-time config |
| Git repo | Large (with data) | Clean (code only) |
| Flexibility | Limited | High |
| Multiple branches | Duplicate data | Shared data |
| Backup | All-or-nothing | Selective |
| Performance | Same | Same (if on SSD) |

## Migration: Integrated → Separated

Existing setup to separated:

```bash
# 1. Create data directory
mkdir D:\NBA_Data

# 2. Move data
move data D:\NBA_Data\data
move nba_dfs.db D:\NBA_Data\nba_dfs.db
move models D:\NBA_Data\models

# 3. Update git
echo "data/" >> .gitignore
echo "nba_dfs.db" >> .gitignore
echo "models/" >> .gitignore
git rm -r --cached data/
git rm --cached nba_dfs.db
git commit -m "Move to separated architecture"

# 4. Run with data-dir
python scripts\run_backtest.py --data-dir D:\NBA_Data --test-start 20250101 --test-end 20250105
```

Done. Code clean, data separated.

## Advanced: Network Storage

**Data on NAS:**

```bash
# Map network drive
net use Z: \\NAS\NBA_Data

# Use as data directory
python scripts\run_backtest.py --data-dir Z:\ --test-start 20250101 --test-end 20250105
```

**Note:** Database on network slower. Copy to local SSD:

```bash
copy Z:\nba_dfs.db C:\Temp\nba_dfs.db
python scripts\run_backtest.py --db-path C:\Temp\nba_dfs.db --data-dir Z:\ --test-start 20250101 --test-end 20250105
```

## Best Practices

1. **Keep data directory stable** - don't move frequently
2. **Use environment variables** - avoid hardcoding paths
3. **Document your structure** - team members need to know
4. **Backup separately** - code (git), data (rsync/robocopy)
5. **Monitor disk space** - parquet files accumulate
6. **Archive old outputs** - move to HDD after analysis

## Summary

**Default architecture:**
```bash
python scripts\run_backtest.py --test-start 20250101 --test-end 20250105
```

**Separated architecture:**
```bash
python scripts\run_backtest.py --data-dir D:\NBA_Data --test-start 20250101 --test-end 20250105
```

One flag enables clean separation. Choose based on needs.
