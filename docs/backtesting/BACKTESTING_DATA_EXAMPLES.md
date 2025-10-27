# Backtesting Pipeline - Data Transformation Examples

This document shows concrete examples of data structures at each stage of the pipeline.

---

## Stage 1: Raw Training Data (from SQLite)

### Input: Box Scores Table
```
playerID      | playerName    | team | pos | gameDate | gameID              | pts | reb | ast | stl | blk | TOV | mins | fpts
28908111729   | Nikola Jokic  | DEN  | C   | 20241020 | DEN@LAL_20241020    | 23  | 11  | 8   | 1   | 2   | 3   | 32   | 48.5
28908111729   | Nikola Jokic  | DEN  | C   | 20241022 | DEN@MEM_20241022    | 26  | 9   | 7   | 0   | 1   | 2   | 30   | 50.2
28908111729   | Nikola Jokic  | DEN  | C   | 20241024 | DEN@HOU_20241024    | 21  | 10  | 6   | 1   | 3   | 4   | 29   | 46.1
28118035349   | Giannis...    | MIL  | C   | 20241020 | MIL@BOS_20241020    | 28  | 10  | 4   | 2   | 1   | 2   | 33   | 51.2
28118035349   | Giannis...    | MIL  | C   | 20241022 | MIL@CLE_20241022    | 30  | 11  | 5   | 1   | 2   | 1   | 35   | 54.1
...
```

---

## Stage 2: After Feature Engineering

### After Rolling Stats Transformer (3-game window)

```
playerID  | pts | reb | ast | rolling_mean_pts_3 | rolling_std_pts_3 | rolling_mean_reb_3 | rolling_mean_ast_3
28908111729| 21 | 10  | 6   | 23.33              | 2.31              | 10.00              | 7.00
28118035349| 30 | 11  | 5   | 29.33              | 1.15              | 10.67              | 4.67
```

**How it's calculated** (for Nikola Jokic game 3):
```
rolling_mean_pts_3 = mean([23, 26, 21]) = 23.33
rolling_std_pts_3 = std([23, 26, 21]) = 2.31
rolling_mean_reb_3 = mean([11, 9, 10]) = 10.00
```

### After EWMA Transformer (span=5)

```
playerID      | ewma_pts_5 | ewma_reb_5 | ewma_ast_5 | ewma_stl_5 | ... (147 total features)
28908111729   | 23.82      | 10.18      | 7.23       | 0.91       | ...
28118035349   | 29.45      | 10.82      | 4.67       | 1.45       | ...
```

**How EWMA is calculated**:
```
ewma_t = (value_t × α) + (ewma_{t-1} × (1-α))
where α = 2/(span+1) = 2/(5+1) = 0.333

For game 1:  ewma = 23 (no history)
For game 2:  ewma = (26 × 0.333) + (23 × 0.667) = 24.33
For game 3:  ewma = (21 × 0.333) + (24.33 × 0.667) = 22.89
For game 4:  ewma = (27 × 0.333) + (22.89 × 0.667) = 24.30
For game 5:  ewma = (30 × 0.333) + (24.30 × 0.667) = 26.43
```

---

## Stage 3: Complete Feature Matrix for Training

### X_train - Feature Matrix

```
[n_samples=15000, n_features=147]

Sample row for Nikola Jokic (2025-01-31 game):

rolling_mean_pts_3=25.2    rolling_mean_pts_5=24.8    rolling_mean_pts_10=24.3
rolling_std_pts_3=2.1      rolling_std_pts_5=1.9      rolling_std_pts_10=1.8
rolling_mean_reb_3=10.1    rolling_mean_reb_5=10.0    rolling_mean_reb_10=9.9
rolling_std_reb_3=0.8      rolling_std_reb_5=0.7      rolling_std_reb_10=0.8
rolling_mean_ast_3=7.2     rolling_mean_ast_5=7.0     rolling_mean_ast_10=6.8
rolling_std_ast_3=0.9      rolling_std_ast_5=0.8      rolling_std_ast_10=0.9
rolling_mean_stl_3=1.1     rolling_mean_stl_5=1.0     rolling_mean_stl_10=1.0
rolling_std_stl_3=0.3      rolling_std_stl_5=0.3      rolling_std_stl_10=0.3
rolling_mean_blk_3=1.5     rolling_mean_blk_5=1.4     rolling_mean_blk_10=1.3
rolling_std_blk_3=0.4      rolling_std_blk_5=0.4      rolling_std_blk_10=0.3
rolling_mean_TOV_3=2.8     rolling_mean_TOV_5=2.7     rolling_mean_TOV_10=2.8
rolling_std_TOV_3=0.8      rolling_std_TOV_5=0.7      rolling_std_TOV_10=0.8
rolling_mean_mins_3=31.3   rolling_mean_mins_5=31.0   rolling_mean_mins_10=31.2
rolling_std_mins_3=1.2     rolling_std_mins_5=1.1     rolling_std_mins_10=1.3

ewma_pts_5=25.1     ewma_reb_5=10.0     ewma_ast_5=7.1     ewma_stl_5=1.0
ewma_blk_5=1.4      ewma_TOV_5=2.8      ewma_mins_5=31.2

opponent_pts_allowed_3=108.2   opponent_reb_allowed_3=45.1   opponent_ast_allowed_3=25.3
opponent_pts_allowed_5=107.8   opponent_reb_allowed_5=44.9   opponent_ast_allowed_5=25.1
...
(147 features total)
```

### y_train - Target Variable

```
[n_samples=15000]

= [48.5, 50.2, 46.1, 51.2, 54.1, ..., 47.3, 49.8, ...]

Represents: DraftKings Fantasy Points for that player-game
Range: 0-150 (typically 20-60 for most players)
```

---

## Stage 4: Slate Data (Test Date)

### DFS Salaries for 2025-02-05

```
playerID      | longName          | pos | team | salary | fppg | status
28908111729   | Nikola Jokic      | C   | DEN  | 11500  | 45.2 | OK
28118035349   | Giannis...        | C   | MIL  | 11200  | 38.5 | OK
28778646789   | Shai Gilgeous-... | PG  | OKC  | 10800  | 32.1 | QUESTIONABLE
28698011729   | Joel Embiid       | C   | PHI  | 10500  | 41.2 | OUT
28898319129   | Devin Booker      | SG  | PHX  | 10200  | 38.1 | OK
... (100+ more players)
```

### After Player Filtering

```
# Applied filters:
# - salary >= 5000
# - status NOT in [OUT, DOUBTFUL] (if injury filters enabled)
# - player name IN ['Lebron James', 'Stephen Curry'] (if names filter enabled)

Result: 85 qualified players for this slate (filtered from 120)
```

---

## Stage 5: Slate Features Built for Prediction

### Feature Matrix for Test Slate (Per-Slate Model)

```
Shape: [85 players, 147 features]

For Nikola Jokic (2025-02-05 slate):

playerID: 28908111729
playerName: "Nikola Jokic"
team: "DEN"
pos: "C"
salary: 11500

Feature columns (147 total):
rolling_mean_pts_3: 25.2
rolling_mean_pts_5: 24.8
rolling_mean_pts_10: 24.3
rolling_std_pts_3: 2.1
rolling_std_pts_5: 1.9
rolling_std_pts_10: 1.8
rolling_mean_reb_3: 10.1
rolling_mean_reb_5: 10.0
rolling_mean_reb_10: 9.9
... (144 more features)

For Giannis Antetokounmpo (2025-02-05 slate):

playerID: 28118035349
playerName: "Giannis Antetokounmpo"
team: "MIL"
pos: "C"
salary: 11200

rolling_mean_pts_3: 28.1
rolling_mean_pts_5: 27.5
rolling_mean_pts_10: 27.1
... (144 more features)

...and so on for 85 players total
```

**How these features are extracted**:
```
1. For each player in slate, find their most recent games
2. Extract last game row from training data features
3. Use those features as input to model

Example - Nikola Jokic's latest game in training data was 2025-01-31
- rolling_mean_pts_3 based on 3 games before that date
- ewma_pts_5 calculated through 2025-01-31
- All features represent player's state BEFORE test date (no lookahead)
```

---

## Stage 6: Model Predictions

### Predictions for 2025-02-05 Slate

```
X_test shape: [85, 147]
↓
model.predict(X_test)
↓
predictions shape: [85]

playerID      | playerName            | team | pos | salary | projected_fpts
28908111729   | Nikola Jokic          | DEN  | C   | 11500  | 47.2
28118035349   | Giannis...            | MIL  | C   | 11200  | 41.3
28778646789   | Shai Gilgeous-...     | OKC  | PG  | 10800  | 33.1
28898319129   | Devin Booker          | PHX  | SG  | 10200  | 37.8
28046691632   | Stephen Curry         | GSW  | PG  | 9800   | 35.4
... (80 more players)
```

**Prediction interpretation**:
```
- Model predicts Nikola Jokic will score 47.2 DK fantasy points
- This is the model's expected value for this slate
- Based on his historical patterns from training data
```

---

## Stage 7: After Adding Benchmark

### Benchmark Predictions Added

```
playerID      | playerName            | projected_fpts | benchmark_pred
28908111729   | Nikola Jokic          | 47.2           | 45.2
28118035349   | Giannis...            | 41.3           | 38.5
28778646789   | Shai Gilgeous-...     | 33.1           | 32.1
28898319129   | Devin Booker          | 37.8           | 38.1
28046691632   | Stephen Curry         | 35.4           | 34.2
... (80 more)

benchmark_pred = Historical average DK points for each player
```

**Benchmark calculation** (from training period):
```
For each player, calculate: avg_fpts = mean(all_fpts_in_training_period)

Nikola Jokic training period (Oct 1 - Jan 31):
  Games: 65
  Total FPTS: 2938
  Average: 2938 / 65 = 45.2 FPTS

Stephen Curry training period:
  Games: 58
  Total FPTS: 1984
  Average: 1984 / 58 = 34.2 FPTS
```

---

## Stage 8: Actual Results (After Game Played)

### Actual Fantasy Points (2025-02-05 Games Completed)

```
playerID      | playerName            | team | pos | salary | actual_fpts | game_result
28908111729   | Nikola Jokic          | DEN  | C   | 11500  | 46.8        | 27 pts, 11 reb, 8 ast
28118035349   | Giannis...            | MIL  | C   | 11200  | 42.1        | 30 pts, 10 reb, 5 ast
28778646789   | Shai Gilgeous-...     | OKC  | PG  | 10800  | 31.5        | 22 pts, 5 reb, 7 ast
28898319129   | Devin Booker          | PHX  | SG  | 10200  | 39.2        | 28 pts, 4 reb, 6 ast
28046691632   | Stephen Curry         | GSW  | PG  | 9800   | 36.1        | 26 pts, 4 reb, 8 ast
... (80 more players)
```

**DK Fantasy Points Calculation**:
```
For Nikola Jokic (27 pts, 11 reb, 8 ast, 0 stl, 1 blk, 2 TOV, 32 mins):

DK Points = (pts×1) + (reb×1.25) + (ast×1.5) + (stl×2) + (blk×2) + (TOV×-0.5)
          = (27×1) + (11×1.25) + (8×1.5) + (0×2) + (1×2) + (2×-0.5)
          = 27 + 13.75 + 12 + 0 + 2 - 1
          = 53.75 FPTS
```

---

## Stage 9: Merged Predictions + Actuals

### Combined Results DataFrame

```
playerID    | playerName       | team| pos|salary|projected|benchmark|actual|error|error_pct|salary_bin|rank_diff
28908111729 | Nikola Jokic     | DEN | C  |11500|47.2    |45.2     |46.8  |-0.4 |-0.85%   |$11k+   |-0.5
28118035349 | Giannis...       | MIL | C  |11200|41.3    |38.5     |42.1  |-0.8 |-1.90%   |$11k+   |+0.2
28778646789 | Shai...          | OKC | PG |10800|33.1    |32.1     |31.5  |+1.6 |+5.08%   |$10k    |0.0
28898319129 | Devin Booker     | PHX | SG |10200|37.8    |38.1     |39.2  |-1.4 |-3.57%   |$10k    |-0.2
28046691632 | Stephen Curry    | GSW | PG |9800 |35.4    |34.2     |36.1  |-0.7 |-1.93%   |$9k     |+0.1
...
```

**Error columns**:
```
error = actual - projected
error_pct = (error / actual) × 100

For Nikola Jokic:
  error = 46.8 - 47.2 = -0.4
  error_pct = (-0.4 / 46.8) × 100 = -0.85%
  (Model overestimated by 0.4 points, or 0.85%)
```

---

## Stage 10: Daily Metrics Calculation

### Metrics Calculated for 2025-02-05

```
# Filter to players with both predictions and actuals
valid_players = 83 (out of 85 with projections, 80 with actuals)

# Model Performance Metrics:
MAPE = mean(|error_pct|) = 31.2%
RMSE = sqrt(mean(error²)) = 12.3
MAE = mean(|error|) = 9.8
Correlation = pearson(actual, projected) = 0.748

# Benchmark Performance Metrics:
Benchmark_MAPE = mean(|benchmark_error_pct|) = 34.1%
Benchmark_RMSE = sqrt(mean(benchmark_error²)) = 13.5
Benchmark_MAE = mean(|benchmark_error|) = 10.2
Benchmark_Correlation = 0.712

# Improvement:
MAPE_Improvement = 34.1% - 31.2% = +2.9%  (positive = model better)
Model_Better_Count = 52 players (model error < benchmark error)
Benchmark_Better_Count = 31 players

# Salary Tier Breakdown:
Tier "$5k-$6k": MAPE=42.1%, count=18, mean_error=+2.1
Tier "$6k-$8k": MAPE=31.5%, count=35, mean_error=+0.3
Tier "$8k+":    MAPE=26.8%, count=30, mean_error=-1.2

# Fantasy Points Summary:
Mean_Actual = 28.4 FPTS
Mean_Projected = 28.1 FPTS (slightly conservative)
Mean_Benchmark = 27.9 FPTS (more conservative)
```

**Metric Calculations** (for all 83 players):

```
1. MAPE:
   errors = |46.8-47.2|, |42.1-41.3|, |31.5-33.1|, ...
   error_pcts = [0.85, 1.90, 5.08, ..., 1.93]
   MAPE = mean(error_pcts) = 31.2%

2. RMSE:
   squared_errors = [0.16, 0.64, 2.56, ..., 0.49]
   mse = mean(squared_errors) = 152.09
   RMSE = sqrt(152.09) = 12.3

3. Correlation:
   actual_values = [46.8, 42.1, 31.5, ...]
   projected_values = [47.2, 41.3, 33.1, ...]
   correlation = pearson(actual, projected) = 0.748

4. Salary Tier (example):
   tier_$8k+ = df[df['salary'] >= 8000]
   tier_errors = |tier_actual - tier_projected|
   tier_MAPE = mean(tier_errors / tier_actual) × 100 = 26.8%
```

---

## Stage 11: Daily Results Dictionary

```python
daily_results = {
    'date': '20250205',
    'num_players': 85,                  # Players with projections
    'num_players_with_actuals': 83,     # Players with actual results

    # Model Performance
    'model_mape': 31.2,
    'model_rmse': 12.3,
    'model_mae': 9.8,
    'model_corr': 0.748,
    'model_median_mape': 28.5,
    'model_std_mape': 15.3,
    'model_mean_fpts': 28.1,

    # Benchmark Performance
    'benchmark_mape': 34.1,
    'benchmark_rmse': 13.5,
    'benchmark_mae': 10.2,
    'benchmark_corr': 0.712,
    'benchmark_median_mape': 31.2,
    'benchmark_std_mape': 18.1,
    'benchmark_mean_fpts': 27.9,

    # Comparison
    'mape_improvement': 2.9,            # positive = better
    'model_better_count': 52,           # Player wins for model
    'benchmark_better_count': 31,       # Player wins for benchmark

    # Fantasy Points
    'mean_actual': 28.4,
    'mean_projected': 28.1,
    'mean_benchmark': 27.9,
    'mean_error': -0.3,
    'mean_benchmark_error': +0.5,

    # Salary Tier Analysis
    'salary_tier_analysis': {
        '$5k-$6k': {
            'mape': 42.1,
            'count': 18,
            'mean_error': 2.1,
            'mean_actual': 18.2,
            'mean_projected': 20.3
        },
        '$6k-$8k': {
            'mape': 31.5,
            'count': 35,
            'mean_error': 0.3,
            'mean_actual': 28.1,
            'mean_projected': 28.4
        },
        '$8k+': {
            'mape': 26.8,
            'count': 30,
            'mean_error': -1.2,
            'mean_actual': 39.2,
            'mean_projected': 38.0
        }
    },

    # Execution Timing
    'date_processed_at': '2025-02-05T14:30:22',
    'processing_time_seconds': 32.4,
}
```

---

## Stage 12: Season Aggregation

### Aggregated Results (All Slates Combined)

```python
aggregated_results = {
    'num_slates': 11,                          # 11 days of backtesting
    'date_range': 'Feb 5-15, 2025',

    # Aggregated Model Metrics
    'model_mean_mape': 31.5,                   # Average MAPE across all slates
    'model_median_mape': 30.8,
    'model_std_mape': 4.2,
    'model_mean_rmse': 12.1,
    'model_mean_mae': 9.6,
    'model_mean_correlation': 0.745,

    # Aggregated Benchmark Metrics
    'benchmark_mean_mape': 34.3,
    'benchmark_median_mape': 33.9,
    'benchmark_std_mape': 5.1,
    'benchmark_mean_rmse': 13.2,
    'benchmark_mean_mae': 10.1,
    'benchmark_mean_correlation': 0.710,

    # Overall Improvement
    'mape_improvement': 2.8,                   # positive = model better
    'num_slates_model_better': 8,              # Days when model beat benchmark
    'num_slates_benchmark_better': 3,

    # Player Statistics
    'total_players_evaluated': 850,            # Across all slates
    'avg_players_per_slate': 77.3,
    'total_predictions': 850,
    'total_actuals': 823,                      # Some missing game data

    # Salary Tier Performance
    'salary_tier_analysis': {
        '$5k-$6k': {
            'model_mape': 41.2,
            'benchmark_mape': 44.5,
            'improvement': 3.3,
            'total_players': 180
        },
        '$6k-$8k': {
            'model_mape': 31.1,
            'benchmark_mape': 34.2,
            'improvement': 3.1,
            'total_players': 350
        },
        '$8k+': {
            'model_mape': 27.3,
            'benchmark_mape': 29.1,
            'improvement': 1.8,
            'total_players': 320
        }
    },

    # Statistical Testing
    'statistical_test': {
        'test_type': 'paired_t_test',
        't_statistic': 2.34,
        'p_value': 0.0156,
        'degrees_of_freedom': 822,
        'cohens_d': 0.15,
        'effect_size': 'small'
    },

    # Daily Breakdown
    'daily_results': [
        {'date': '20250205', 'model_mape': 31.2, 'benchmark_mape': 34.1},
        {'date': '20250206', 'model_mape': 30.8, 'benchmark_mape': 33.5},
        ...
        {'date': '20250215', 'model_mape': 31.9, 'benchmark_mape': 35.2},
    ],

    # All Predictions
    'all_predictions': {
        'total_rows': 823,
        'columns': ['playerID', 'playerName', 'projected_fpts', 'actual_fpts',
                    'benchmark_pred', 'error', 'salary_bin', ...]
    }
}
```

---

## File Outputs Summary

### 1. Predictions File: `20250205.parquet`
```
playerID  | playerName            | team | pos | salary | projected_fpts | benchmark_pred
28908111729| Nikola Jokic          | DEN  | C   | 11500  | 47.2           | 45.2
28118035349| Giannis...            | MIL  | C   | 11200  | 41.3           | 38.5
...
```
**Columns**: ~10 metadata + projection columns
**Size**: ~150KB
**Rows**: 85

### 2. Predictions with Actuals: `20250205_with_actuals.parquet`
```
(All columns from above, plus:)
actual_fpts | error | error_pct | salary_bin | model_rank | benchmark_rank
46.8        | -0.4  | -0.85%    | $11k+      | 1          | 1
42.1        | -0.8  | -1.90%    | $11k+      | 2          | 3
...
```
**Columns**: ~20 (predictions + actuals + derived metrics)
**Size**: ~300KB
**Rows**: 83

### 3. Training Inputs: `slate_training_inputs_20250205.parquet`
```
[15000 rows × 147 features]
All training samples used to train the model for this slate
```
**Columns**: rolling_mean_*, rolling_std_*, ewma_*
**Size**: 25MB
**Rows**: 15,000

### 4. Model Checkpoint: `checkpoints/20250205.json`
```json
{
    "date": "20250205",
    "status": "completed",
    "daily_result": {...full daily_results dict...},
    "timestamp": "2025-02-05T14:30:22.123456"
}
```
**Size**: ~50KB
**Purpose**: Resume interrupted backtests

### 5. Report: `backtest_report.html`
```html
PDF-styled report with:
- Executive summary
- Key metrics table
- Daily performance chart
- Salary tier breakdown
- Model vs benchmark comparison
- Top/bottom performers
- Feature importance (if available)
- Configuration used
```

---

## Data Flow Summary Table

| Stage | Input | Processing | Output | Size | Rows |
|-------|-------|-----------|--------|------|------|
| 1 | Raw box scores | Load from DB | DataFrame | ~500MB | 50k |
| 2 | Box scores | Feature pipeline | Feature matrix X, y | ~100MB | 50k |
| 3 | X, y | Train model | Fitted XGBoost | ~10MB | - |
| 4 | Slate salaries | Load from DB | Eligible players | ~1MB | 85 |
| 5 | Training features | Extract latest | Slate features | ~5MB | 85 |
| 6 | Slate features | Model inference | Predictions | ~500KB | 85 |
| 7 | Predictions | Add benchmark | With benchmark | ~600KB | 85 |
| 8 | Game results | Load from DB | Actuals | ~400KB | 83 |
| 9 | Pred + Actuals | Merge/join | Combined results | ~800KB | 83 |
| 10 | Combined results | Calculate metrics | Daily results dict | ~50KB | - |
| 11 | All daily results | Aggregate | Season summary | ~100KB | - |
| 12 | All data | Generate | HTML/PDF reports | ~5MB | - |
