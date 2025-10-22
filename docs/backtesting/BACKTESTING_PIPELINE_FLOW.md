# NBA DFS Backtesting Pipeline - Complete Flow & Logic

## Overview

The walk-forward backtesting pipeline is a systematic evaluation framework that trains NBA DFS fantasy point prediction models on historical data and validates them across future dates. The pipeline processes multiple daily slates sequentially, generating projections and comparing them against actual results.

---

## 1. Pipeline Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKTESTING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. DATA LAYER - Historical Data Loading                 │   │
│  │    • Load training data (historical box scores)         │   │
│  │    • Load slate data (salaries, schedule, injuries)     │   │
│  │    • Prevent lookahead bias with temporal validation    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. FEATURE LAYER - Feature Engineering                  │   │
│  │    • Build rolling window statistics (3, 5, 10 games)  │   │
│  │    • Calculate EWMA features (span=5)                   │   │
│  │    • Generate 147 features from 21 box score stats      │   │
│  │    • Output: X_train, X_test feature matrices           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. MODEL LAYER - Training & Prediction                  │   │
│  │    • Train XGBoost/RandomForest on features             │   │
│  │    • Generate predictions for test players              │   │
│  │    • Mode: Per-player (500+ models) or Per-slate (1)    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. EVALUATION LAYER - Metrics & Comparison              │   │
│  │    • Load actual fantasy points from game results       │   │
│  │    • Calculate MAPE, RMSE, MAE, Correlation            │   │
│  │    • Benchmark vs season average baseline               │   │
│  │    • Analyze by salary tier                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 5. OUTPUT LAYER - Results & Reporting                   │   │
│  │    • Save predictions to parquet files                  │   │
│  │    • Generate PDF-styled reports                        │   │
│  │    • Interactive Plotly visualizations                  │   │
│  │    • Model & training input persistence                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Diagram

```
INPUT DATA (External Sources)
├── SQLite Database (nba_dfs.db)
│   ├── box_scores → Player game-by-game stats
│   ├── dfs_salaries → DraftKings salary data
│   ├── schedule → Game dates & matchups
│   └── injuries → Player injury status
│
├── YAML Configuration Files
│   ├── config/features/opponent_features.yaml
│   ├── config/models/xgboost_default.yaml
│   └── config/experiments/experiment_config.yaml
│
└── Runtime Filters
    ├── Salary range filtering
    ├── Injury status filtering
    └── Player name/ID filtering

            ↓↓↓

PROCESSING PIPELINE
├── TRAINING PHASE (Historical Data)
│   ├── Load historical box scores
│   │   └── Date range: train_start → train_end
│   │   └── Temporal validation: NO data ≥ train_end
│   │
│   ├── Feature engineering
│   │   ├── Rolling averages (3, 5, 10 game windows)
│   │   ├── EWMA transformers (span=5)
│   │   └── 147 total features generated
│   │
│   └── Model training
│       ├── Per-player: 500+ individual XGBoost models
│       └── Per-slate: 1 slate-level model
│
├── BENCHMARK INITIALIZATION
│   ├── Fit SeasonAverageBenchmark on training data
│   └── Calculate per-player average fantasy points
│
└── TESTING PHASE (Each Daily Slate)
    ├── Load slate data
    │   ├── Salaries for that date
    │   ├── Injuries for that date
    │   └── Schedule/matchups
    │
    ├── Feature building
    │   ├── Use latest game stats from training data
    │   ├── Calculate rolling/EWMA features
    │   └── Prepare feature vectors
    │
    ├── Generate predictions
    │   ├── Per-player: Run individual models
    │   └── Per-slate: Run single model
    │
    ├── Load actuals
    │   ├── Fetch game box scores
    │   └── Calculate fantasy points for that slate
    │
    └── Evaluation & metrics calculation

            ↓↓↓

OUTPUT DATA
├── Predictions Directory (data/outputs/{timestamp}/predictions/)
│   ├── {date}.parquet → Model predictions
│   └── {date}_with_actuals.parquet → Predictions + actuals + metrics
│
├── Model Files (if save_models=True)
│   ├── Per-slate: models/per_slate_models/
│   └── Per-player: models/per_player_models/
│
├── Training Inputs (if save_predictions=True)
│   ├── Slate inputs: data/outputs/{timestamp}/inputs/slate_training_inputs_{date}.parquet
│   └── Player inputs: data/outputs/{timestamp}/inputs/player_*_inputs.parquet
│
├── Features Directory
│   └── Generated feature files
│
└── Reports
    ├── HTML report with Plotly charts
    ├── PDF-styled analysis report
    └── Summary statistics tables
```

---

## 3. Detailed Processing Flow

### Phase 1: Initialization

**Input**: Configuration parameters
- Training period: `train_start`, `train_end`
- Testing period: `test_start`, `test_end`
- Model config: `model_type`, `model_params`
- Feature config: `feature_config` (YAML file)

**Processing**:
1. Initialize SQLiteStorage and HistoricalDataLoader
2. Load YAML feature configuration
3. Build FeaturePipeline from config
4. Create output directories
5. Initialize metrics (MAPE, RMSE, MAE, Correlation)

**Output**: Configured backtest object, ready for execution

---

### Phase 2: Benchmark Initialization

**Input**: Training data date range

**Processing**:
```
1. Load historical data
   └─ Range: train_start → train_end
   └─ Data: Box scores with player stats
   └─ Validation: No lookahead bias (max_date < train_end)

2. Build training features
   ├─ Date parsing: Convert to datetime
   ├─ Sort by: playerID, gameDate
   ├─ Calculate: fantasy points (if missing)
   ├─ Feature pipeline fit_transform()
   └─ Filter: Remove NaN targets

3. Qualify players
   └─ Filter: min_games_for_benchmark ≥ 5 games

4. Fit SeasonAverageBenchmark
   ├─ Calculate: mean fantasy points per player
   └─ Store: player_id → avg_fpts mapping
```

**Output**: Benchmark object with per-player average fantasy points

**Example**:
```python
benchmark.player_averages = {
    28908111729: 45.2,  # Nikola Jokic avg
    28118035349: 38.5,  # Giannis avg
    ...
}
```

---

### Phase 3: Per-Slate Processing Loop

For each test date in `test_start` → `test_end`:

#### Step 3.1: Load Slate Data

**Input**: Test date (YYYYMMDD format)

**Processing**:
```python
# Load all data for this date
slate_data = {
    'dfs_salaries': Load salary data for date
    'schedule': Game schedule
    'betting_odds': Vegas lines
    'injuries': Player injury reports
}

# Apply player filters
salaries_df.filter(salary >= FILTER_SALARY_MIN)
salaries_df.filter(salary <= FILTER_SALARY_MAX)
salaries_df.filter_injuries(exclude_OUT, exclude_DOUBTFUL)
salaries_df.filter_by_player_names(FILTER_PLAYER_NAMES)
salaries_df.filter_by_player_ids(FILTER_PLAYER_IDS)
```

**Output**: Filtered list of players eligible for this slate

---

#### Step 3.2: Load Training Data

**Input**: None (uses cached data)

**Processing**:
```python
# Load once per backtest, cache for all slates
training_data = load_historical_player_logs(
    start_date=train_start,
    end_date=train_end,
    num_seasons=1  # Current season only, or 2 for previous too
)
```

**Output**: DataFrame with all historical box scores for training period

---

#### Step 3.3: Model Recalibration Check

**Input**: Current date, last training date

**Processing**:
```python
should_recalibrate = (current_date - last_training_date) >= recalibrate_days

# OR if rewrite_models flag is True → Always retrain
```

**Decision Tree**:
- First slate? → YES, train
- Days since last training ≥ recalibrate_days? → YES, retrain
- Otherwise? → Reuse previous model

**Output**: Boolean flag to train or reuse model

---

#### Step 3.4: Model Training (if recalibration needed)

**Input**: Training data, feature config

**Processing** for **Per-Slate Model**:
```
1. Build training features
   ├─ Call: feature_pipeline.fit_transform(training_data)
   ├─ Output: X_train [n_samples, 147 features]
   │          y_train [n_samples] (target = fantasy points)
   └─ Rows: Minimum 1,000s of player-game combinations

2. Train model
   ├─ Model type: XGBoost (or RandomForest)
   ├─ Params: {max_depth=6, learning_rate=0.05, n_estimators=200}
   ├─ GPU training: device='cuda:0' (if USE_GPU=True)
   └─ Output: Fitted model object

3. Save model
   ├─ File: models/per_slate_models/xgboost_{date}.pkl
   ├─ Metadata: {n_training_samples, feature_count, train_date}
   └─ Training inputs: data/outputs/{timestamp}/inputs/slate_training_inputs_{date}.parquet

4. Cache model
   └─ In-memory: self.current_model
```

**Processing for Per-Player Models**:
```
1. For each player in salaries_df (parallel, n_jobs=32):

   a) Filter training data
      └─ Player-specific data: training_data[training_data['playerID'] == player_id]

   b) Skip if insufficient games
      └─ If len(player_data) < min_player_games (10) → skip

   c) Build features (fresh pipeline per worker thread)
      ├─ Create new FeaturePipeline() to avoid thread safety
      ├─ fit_transform() on player's games
      └─ Output: X [n_games, 147 features], y [n_games]

   d) Train model
      ├─ Create XGBoostModel(model_params)
      ├─ model.train(X, y)
      └─ Output: Player-specific model

   e) Generate projection
      ├─ latest_features = X.iloc[[-1]]  # Last game only
      ├─ prediction = model.predict(latest_features)[0]
      └─ Store: playerID → projected_fpts

2. Collect results
   └─ projections DataFrame with all players
```

**Output**: Model object stored in `self.current_model`

---

#### Step 3.5: Build Slate Features

**Input**: Training data, slate data

**Processing**:
```
1. Extract salaries
   └─ salaries_df = slate_data['dfs_salaries']

2. For each player in salaries_df:

   a) Find player's historical games
      └─ player_features = training_data[training_data['playerID'] == player_id]

   b) Extract last game features
      └─ last_row = player_features.iloc[-1]

   c) Build feature vector
      └─ features = {
             playerID, playerName, team, pos, salary,
             feature_1, feature_2, ..., feature_147
         }

   d) Append to slate_features

3. Output: DataFrame with shape [n_players_in_slate, 1+147+metadata]
```

**Output**: Feature matrix ready for prediction

**Example row**:
```
playerID: 28908111729
playerName: "Nikola Jokic"
team: "DEN"
pos: "C"
salary: 11500
rolling_mean_pts_3: 27.33
rolling_mean_reb_3: 11.67
ewma_ast_5: 8.2
... (144 more features)
```

---

#### Step 3.6: Generate Predictions

**Input**: Model object, feature matrix

**Processing** for **Per-Slate Model**:
```python
# Use same model for all players
projections_df = model.predict(slate_features[[feature_columns]])
# Output: 1D array of projected fantasy points

# Attach to player data
projections_df['projected_fpts'] = predictions
```

**Processing for Per-Player Models**:
```
# Already done in training step - use cached projections
# Each player has own projection in results dictionary
```

**Output**: DataFrame with columns:
```
playerID, playerName, team, pos, salary, projected_fpts, ...
```

---

#### Step 3.7: Add Benchmark Predictions

**Input**: Projections DataFrame, benchmark object

**Processing**:
```python
projections_df['benchmark_pred'] = projections_df['playerID'].map(
    self.benchmark.player_averages
).fillna(0)

# Result: For each player, add baseline (season average)
```

**Example**:
```
playerID  projected_fpts  benchmark_pred
28908111729    48.5           45.2
28118035349    42.1           38.5
28778646789    35.2           32.1
```

---

#### Step 3.8: Load Actual Results

**Input**: Test date

**Processing**:
```
1. Query database for games on test_date

2. For each game:
   ├─ Get box scores
   ├─ Calculate DK fantasy points per player
   └─ Store: playerID → actual_fpts

3. Return DataFrame:
   └─ playerID, actual_fpts, team, pos, ...
```

**Output**: Actuals DataFrame

---

#### Step 3.9: Evaluation & Metrics

**Input**:
- Predictions DataFrame: `projected_fpts`, `benchmark_pred`
- Actuals DataFrame: `actual_fpts`

**Processing**:
```
1. Merge predictions with actuals
   └─ Outer join on playerID

2. Calculate metrics

   a) Model Metrics:
      ├─ MAPE = mean(|actual - projected| / |actual|) × 100
      ├─ RMSE = sqrt(mean((actual - projected)²))
      ├─ MAE = mean(|actual - projected|)
      └─ Correlation = pearson(actual, projected)

   b) Benchmark Metrics:
      ├─ Same formulas but with benchmark_pred instead
      └─ Compare improvement: benchmark_MAPE - model_MAPE

   c) Salary Tier Analysis:
      └─ Group by salary_bin, calculate MAPE for each tier

3. Build daily_results dict
   └─ {
        'date': test_date,
        'num_players': len(merged),
        'model_mape': 32.1,
        'model_rmse': 12.5,
        'model_mae': 10.2,
        'model_corr': 0.745,
        'benchmark_mape': 35.2,
        'mape_improvement': +3.1,  # positive = better
        ...
      }
```

**Output**: Daily results dictionary + merged predictions/actuals DataFrame

---

#### Step 3.10: Save Results

**Input**: Predictions, results, model

**Processing**:
```
1. Save predictions
   ├─ File: data/outputs/{timestamp}/predictions/{date}.parquet
   ├─ Columns: playerID, playerName, projected_fpts, benchmark_pred
   └─ Rows: All players with projections

2. Save predictions with actuals
   ├─ File: data/outputs/{timestamp}/predictions/{date}_with_actuals.parquet
   ├─ Columns: All above + actual_fpts, error, salary_bin, etc.
   └─ Rows: Players with both projection and actual

3. Save model (if save_models=True)
   ├─ File: models/per_slate_models/xgboost_{date}.pkl
   ├─ Metadata: models/per_slate_models/xgboost_{date}.json
   └─ Includes: feature importance, training stats

4. Save training inputs
   ├─ File: data/outputs/{timestamp}/inputs/slate_training_inputs_{date}.parquet
   ├─ Columns: All X_train features
   └─ Rows: All training samples

5. Save checkpoint
   ├─ File: data/outputs/{timestamp}/checkpoints/{date}.json
   ├─ Content: daily_results for resumption
   └─ Purpose: Allow resuming interrupted backtests

6. Log file locations
   └─ Show file sizes and row counts in logs
```

**Directory structure after processing all slates**:
```
data/outputs/{timestamp}/
├── inputs/
│   ├── slate_training_inputs_20250205.parquet
│   ├── slate_training_inputs_20250206.parquet
│   └── player_*.parquet (if per-player)
├── features/
│   └── (generated features)
├── predictions/
│   ├── 20250205.parquet
│   ├── 20250205_with_actuals.parquet
│   ├── 20250206.parquet
│   └── 20250206_with_actuals.parquet
├── checkpoints/
│   ├── 20250205.json
│   └── 20250206.json
└── reports/
    ├── backtest_report.html
    ├── backtest_analysis.pdf
    └── charts/
        ├── mape_timeline.html
        ├── salary_tier_analysis.html
        └── ...
```

---

## 4. Configuration Parameters Reference

### Training Data Selection
```yaml
train_start: 20241001      # Training data start (YYYYMMDD)
train_end: 20250201        # Training data end (exclusive)
test_start: 20250205       # Test period start
test_end: 20250215         # Test period end
num_seasons: 1             # Number of seasons to load (1 = current, 2 = current + previous)
```

### Model Configuration
```yaml
model_type: "xgboost"      # Model: xgboost or random_forest
model_params:
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 200
  min_child_weight: 5
  subsample: 0.8
  colsample_bytree: 0.8
  objective: "reg:squarederror"
  random_state: 42
  tree_method: "hist"
  device: "cuda:0"         # GPU device (if GPU enabled)
```

### Feature Configuration
```yaml
feature_config: "opponent_features"  # YAML config file in config/features/
# Options:
#   - default_features.yaml (21 stats, 147 features)
#   - opponent_features.yaml (includes opponent context)
#   - base_features.yaml (6 core stats for fast iteration)
```

### Model Recalibration
```yaml
recalibrate_days: 7       # Retrain model every N days
rewrite_models: false     # Force retrain on every slate
per_player_models: false  # True = per-player models, False = slate-level
```

### Execution Settings
```yaml
n_jobs: 32                # Parallel jobs (-1 = all cores)
save_models: true         # Persist trained models
save_predictions: true    # Save projection outputs
use_gpu: true             # GPU acceleration
```

### Player Filtering
```yaml
filter_salary_min: 5000
filter_salary_max: null
filter_exclude_out: false
filter_exclude_doubtful: false
filter_exclude_questionable: false
filter_player_names: ['Lebron James', 'Stephen Curry']
filter_player_ids: null
filter_players_csv: null
```

---

## 5. Data Structures

### Training Data Format
```
Box scores DataFrame:
  Columns: playerID, playerName, team, pos, gameDate, gameID,
           pts, reb, ast, stl, blk, TOV, mins, fpts, ...
  Shape: [n_player_games, n_statistics]
  Example: 50,000 rows (500 players × 100 games each)
```

### Feature Matrix
```
X_train / X_test:
  Columns: rolling_mean_pts_3, rolling_std_pts_3,
           rolling_mean_pts_5, rolling_std_pts_5,
           rolling_mean_pts_10, rolling_std_pts_10,
           ewma_pts_5, ewma_reb_5, ..., (147 total)
  Shape: [n_samples, 147]
  Values: Continuous floats, NaN replaced with 0

Target y_train / y_test:
  Values: DraftKings fantasy points (float)
  Range: 0 to ~150 (depending on player output)
```

### Predictions DataFrame
```
Columns:
  - playerID: Player ID (int)
  - playerName: Player name (str)
  - team: Team abbreviation (str)
  - pos: Position (str: PG, SG, SF, PF, C)
  - salary: DraftKings salary (int)
  - projected_fpts: Model prediction (float)
  - benchmark_pred: Season average baseline (float)

Example:
  playerID  playerName      team pos salary projected_fpts benchmark_pred
  28908111729 Nikola Jokic    DEN  C  11500       48.5          45.2
  28118035349 Giannis Antetokounmpo MIL C  11200       42.1          38.5
```

### Actuals DataFrame (after loading game results)
```
Additional columns added to predictions:
  - actual_fpts: Actual fantasy points scored (float)
  - error: actual - projected (float)
  - error_pct: error / actual × 100 (float)
  - salary_bin: Salary tier for analysis (str)
```

### Daily Results Dictionary
```python
{
    'date': '20250205',
    'num_players': 127,
    'num_players_with_actuals': 125,

    # Model Performance
    'model_mape': 32.1,
    'model_rmse': 12.5,
    'model_mae': 10.2,
    'model_corr': 0.745,
    'model_mean_fpts': 29.3,

    # Benchmark Performance
    'benchmark_mape': 35.2,
    'benchmark_rmse': 13.8,
    'benchmark_mae': 11.1,
    'benchmark_corr': 0.712,
    'benchmark_mean_fpts': 28.9,

    # Comparison
    'mape_improvement': 3.1,      # Positive = model better
    'model_better_count': 78,      # Players where model < benchmark error
    'benchmark_better_count': 47,

    # Statistics
    'mean_actual': 29.3,
    'mean_projected': 29.1,
    'mean_benchmark': 28.8,

    # By salary tier
    'salary_tier_analysis': {
        '$5k-$6k': {'mape': 45.2, 'count': 32},
        '$6k-$8k': {'mape': 32.1, 'count': 45},
        '$8k+': {'mape': 28.5, 'count': 48},
    }
}
```

---

## 6. File Output Summary

### Predictions Files (Parquet format)
```
{date}.parquet
├─ Columns: playerID, playerName, team, pos, salary,
│           projected_fpts, benchmark_pred
├─ Size: ~100KB - 500KB (depends on slate size)
└─ Purpose: Model predictions only

{date}_with_actuals.parquet
├─ Additional: actual_fpts, error, error_pct, salary_bin,
│              model_rank, benchmark_rank
├─ Size: ~150KB - 700KB
└─ Purpose: Full evaluation data for analysis
```

### Model Files (Pickle format)
```
models/per_slate_models/
├─ xgboost_20250205.pkl (~1-50MB)
├─ xgboost_20250205.json (metadata)
└─ Purpose: Reuse model for future predictions

models/per_player_models/
├─ player_Nikola_Jokic_28908111729.pkl
└─ Purpose: Per-player model persistence
```

### Training Input Files (Parquet format)
```
data/outputs/{timestamp}/inputs/
├─ slate_training_inputs_20250205.parquet
│  └─ Columns: [feature_1 ... feature_147]
│  └─ Rows: All training samples used for model
│  └─ Size: 5-50MB
│
└─ player_*_inputs.parquet (per-player mode)
   └─ One file per player trained
```

### Report Files (HTML/PDF)
```
data/outputs/{timestamp}/
├─ backtest_report.html
│  └─ PDF-styled report with embedded charts
│  └─ Full analysis and summary tables
│
└─ charts/
   ├─ mape_timeline.html (interactive Plotly)
   ├─ salary_tier_analysis.html
   ├─ daily_results_table.html
   └─ correlation_scatter.html
```

---

## 7. Key Logic Decisions

### When to Retrain Models
```
Decision Tree:
  if rewrite_models == True:
    → Always retrain
  elif last_training_date is None:
    → Retrain (first slate)
  elif (current_date - last_training_date) >= recalibrate_days:
    → Retrain
  else:
    → Reuse model from previous slate
```

### Feature Generation Strategy
```
Training Phase:
  1. Load ALL historical data for date range
  2. FIT feature pipeline on training data (learn statistics)
  3. TRANSFORM training data with fitted pipeline
  4. Output: X_train with fitted statistics

Testing Phase (Each Slate):
  1. TRANSFORM training data only (don't refit)
  2. Extract latest features per player
  3. Use those features for prediction

  IMPORTANT: Never fit on test data - prevents lookahead bias!
```

### Temporal Validation
```
Lookahead Bias Prevention:

Training Phase:
  - Load data ending BEFORE train_end date
  - Check: max(data.gameDate) < train_end
  - If violated: Raise error "LOOKAHEAD BIAS DETECTED"

Testing Phase:
  - Load historical data UP TO test_date
  - Check: max(data.gameDate) < test_date
  - Features derived from games BEFORE test_date only

Result:
  - No information leakage from future into past
  - Realistic out-of-sample evaluation
```

### Player Filtering Priority
```
1. Salary range filter
   └─ if playerID.salary < FILTER_SALARY_MIN: skip

2. Injury status filter
   └─ if playerID.injury_status in [OUT, DOUBTFUL]: skip

3. Player name filter
   └─ if playerID.name not in FILTER_PLAYER_NAMES: skip

4. Player ID filter
   └─ if playerID not in FILTER_PLAYER_IDS: skip

Applied in order: if any filter returns False, exclude player
```

### Benchmark Expansion Strategy
```
if benchmark_use_all_history == True:
  benchmark_start = train_start  # Expand from beginning
else:
  benchmark_start = train_start  # Use full training range

Rationale:
  - benchmark_use_all_history=True: More data for stable average
  - benchmark_use_all_history=False: More recent/relevant data
```

---

## 8. Performance Metrics Explained

### MAPE (Mean Absolute Percentage Error)
```
MAPE = mean(|actual - projected| / |actual|) × 100

Target: < 30% for elite players (salaries $8k+)
Interpretation: On average, predictions within ±30% of actual

Handling edge cases:
  - Division by zero: Exclude if actual = 0
  - Infinite values: Cap at reasonable maximum
```

### RMSE (Root Mean Squared Error)
```
RMSE = sqrt(mean((actual - projected)²))

Units: Fantasy points
Interpretation: Standard deviation of errors
Sensitive to: Large outliers more than small errors
```

### MAE (Mean Absolute Error)
```
MAE = mean(|actual - projected|)

Units: Fantasy points
Interpretation: Average error in fantasy points
Example: MAE=10 means avg error of ±10 fantasy points
```

### Correlation
```
Correlation = pearson(actual, projected)

Range: -1 to +1
Target: > 0.7 (good), > 0.8 (excellent)
Interpretation: How well relative rankings match
- 0.9: Rankings almost perfect
- 0.7: Rankings generally correct
- 0.5: Rankings moderately useful
```

---

## 9. Checkpoint & Resumption

### Checkpoint Saving
```python
# After each successful slate
checkpoint = {
    'date': test_date,
    'daily_result': {...},  # Full daily results dict
    'timestamp': datetime.now().isoformat(),
    'status': 'completed'
}

# Save to: data/outputs/{timestamp}/checkpoints/{date}.json
```

### Resumption Logic
```python
completed_slates = load_checkpoint()

for slate_date in slate_dates:
    if slate_date in completed_slates:
        logger.info(f"Skipping {slate_date} - already completed")
        # Load cached results
        results.append(load_slate_checkpoint(slate_date))
        continue

    # Process this slate
    ...
```

### Use Cases
- Long-running backtests (100+ slates) interrupted mid-run
- Restarting backtest without re-processing completed dates
- Incremental backtest expansion

---

## 10. Error Handling & Logging

### Critical Error Points

1. **Data Loading**
   - Missing files → Log warning, skip slate
   - Empty DataFrames → Check database, suggest collection scripts
   - Lookahead bias → Halt backtest, show error

2. **Feature Generation**
   - NaN values → Fillna(0) and continue
   - Insufficient data → Skip player (per-player) or slate (slate-level)
   - Type mismatches → Coerce to numeric, log warning

3. **Model Training**
   - Insufficient samples → Skip player/slate, log debug
   - GPU out of memory → Fallback to CPU, log warning
   - Model convergence → Use default params, log warning

4. **Prediction/Evaluation**
   - Missing actuals → Log warning, continue to next slate
   - Metric calculation errors → Log and use default value

### Log Levels
```
DEBUG: Detailed info for troubleshooting (filtered in production)
INFO: Important milestones and progress
WARNING: Non-fatal issues (recovered or skipped)
ERROR: Fatal issues requiring human intervention
```

---

## 11. Resource Consumption

### Memory
```
Training Data: ~500MB (500k player-games)
Feature Cache: ~1GB (if enable_feature_caching=True)
Active Models: ~100MB (per-slate) to ~10GB (per-player)
Total: 1-15GB depending on configuration
```

### Computation
```
Per Slate (Per-Slate Model):
  - Feature building: 5-10 seconds
  - Model training: 10-30 seconds
  - Prediction: 1-2 seconds
  - Evaluation: 1-2 seconds
  - Total: ~30 seconds per slate

Per Slate (Per-Player Models, 32 cores):
  - Training 500+ models in parallel: 60-120 seconds
  - Prediction: 5-10 seconds
  - Total: ~90 seconds per slate
```

### Storage
```
Per Slate Output:
  - Predictions: 100-500KB
  - Predictions with actuals: 150-700KB
  - Training inputs: 5-50MB
  - Model file: 1-50MB
  - Checkpoint: 5-10KB

Per 100 Slates:
  - Predictions: 50-70MB
  - Training inputs: 500MB-5GB
  - Models: 100MB-5GB
  - Total: 1-10GB depending on model type
```

---

## 12. Common Issues & Solutions

### Issue: "No slate dates found"
```
Cause: Database empty for test date range
Solution:
  python scripts/collect_games.py --start-date 20250205 --end-date 20250215
  python scripts/collect_dfs_salaries.py --start-date 20250205 --end-date 20250215
```

### Issue: MAPE much higher than expected
```
Causes:
  1. Too few games for feature stability (< 10 games)
  2. Wrong feature configuration
  3. Model underfitting (low learning_rate, few n_estimators)
  4. Missing key features (injury status, rest days)
Solutions:
  - Increase training data date range
  - Try opponent_features configuration
  - Tune hyperparameters in config/models/xgboost_default.yaml
```

### Issue: GPU out of memory
```
Cause: Batch size too large for GPU
Solutions:
  - Reduce batch size in model config
  - Use CPU-only: set use_gpu=False
  - Reduce n_jobs to free CPU RAM
```

### Issue: Model training much slower than expected
```
Causes:
  1. Too many features
  2. Insufficient parallelization
  3. Old hardware
Solutions:
  - Use base_features config (fewer features)
  - Increase n_jobs to -1 (all cores)
  - Enable GPU training
```

---

## 13. Future Enhancements

1. **Multi-day predictions**: Predict multiple days ahead
2. **Confidence intervals**: Add uncertainty estimates
3. **Feature importance**: SHAP value analysis per player
4. **Ensemble models**: Combine multiple models
5. **Real-time updates**: Integrate live injury/status updates
6. **Lineup generation**: Optimize lineups using predictions
7. **Walk-backward validation**: Test different historical periods
8. **Cross-validation**: K-fold validation within training period

---

## Summary Table

| Component | Input | Processing | Output |
|-----------|-------|-----------|--------|
| **Data Loading** | Dates, DB | SQL queries | DataFrames |
| **Features** | Training data, YAML config | Rolling/EWMA transform | Feature matrices |
| **Models** | Features, Labels | XGBoost/RF training | Trained model |
| **Prediction** | Model, Feature matrix | inference | Projected FPTS |
| **Actuals** | Game date, DB | Box score lookup | Actual FPTS |
| **Evaluation** | Predictions, Actuals | Metric calculation | MAPE/RMSE/MAE/Corr |
| **Output** | Results | Aggregation | Parquet + Reports |
