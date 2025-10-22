# Backtesting Pipeline - Visual Flowchart

## Complete Pipeline Flow (Simplified)

The backtesting pipeline follows these high-level stages:

1. **Initialization** → Load configs, prepare storage
2. **Benchmark Setup** → Calculate baseline season averages
3. **Slate Loop** → Process each day in test period
   - Load data, apply filters
   - Retrain or reuse model (every 7 days)
   - Generate predictions
   - Load actuals, evaluate metrics
   - Save outputs
4. **Aggregation** → Combine results, generate reports

See [BACKTESTING_PIPELINE_FLOW.md](BACKTESTING_PIPELINE_FLOW.md) for detailed step-by-step descriptions.

### Stage 1: Initialization
```
START → Load configs → Prepare storage → Load training data → Validate
```

### Stage 2: Benchmark Initialization
```
Training data → Build features → Fit benchmark → Calculate averages
```

### Stage 3: Per-Slate Loop (for each test date)
```
Load slate data → Filter players → Check recalibrate → Train/reuse model →
Generate predictions → Load actuals → Calculate metrics → Save results
```

### Stage 4: Model Training Decision
```
if rewrite_models=True → Always train
elif first_slate → Train
elif (days_since_last_train ≥ recalibrate_days) → Train
else → Reuse previous model
```

### Stage 5: Aggregation & Reporting
```
Combine all daily results → Calculate season metrics → Generate HTML/PDF reports
```

---

## Detailed Sub-Flow: Feature Generation

**Process:**
1. **Load data** → Historical player box scores with game dates
2. **Sort** → By playerID and gameDate (temporal ordering)
3. **Calculate fantasy points** → DraftKings scoring if missing
4. **Rolling windows** → Calculate mean/std for 3, 5, 10 game windows
   - For each statistic: pts, reb, ast, stl, blk, mins, TOV
5. **EWMA** → Exponential weighted moving average (span=5 games)
   - Captures trend with recent games weighted more
6. **Output** → 147 features per player-game combination
   - 21 base statistics × (3 rolling windows × 2 metrics) + EWMA

**Output**: Feature matrix X [n_samples, 147] and target y [n_samples]

---

## Detailed Sub-Flow: Per-Player Model Training (Parallel)

For each player in the slate (parallel with n_jobs=32):

**Steps:**
1. **Filter** → Get player's training games (must have ≥ min_player_games)
2. **Build Features** → Create fresh FeaturePipeline (thread-safe)
   - Load YAML config
   - Fit/transform player-specific data
3. **Train Model** → XGBoost on player's features
   - Save training inputs to parquet
4. **Predict** → Generate projection using latest game features
5. **Collect** → Merge all player results

**Thread Safety**: Each worker creates its own FeaturePipeline instance to avoid shared state issues.

**Output**: DataFrame with playerID, playerName, team, pos, salary, projected_fpts

---

## Detailed Sub-Flow: Evaluation & Metrics

**Process:**
1. **Merge** → Join predictions DataFrame with actuals on playerID
2. **Calculate Errors** → error = actual - projected, error_pct = (error / actual) × 100
3. **Model Metrics**:
   - MAPE = mean(|error| / |actual|) × 100
   - RMSE = sqrt(mean(error²))
   - MAE = mean(|error|)
   - Correlation = pearson(actual, projected)
4. **Benchmark Metrics** → Same formulas using benchmark_pred instead
5. **Salary Tier Breakdown** → Group by salary bins, recalculate metrics per tier
6. **Calculate Improvement** → improvement = benchmark_MAPE - model_MAPE (positive = better)
7. **Build Results** → Collect all metrics into daily_results dictionary

**Output**: Daily results with model performance, benchmark comparison, and salary tier analysis

---

## Data Flow Diagram (Inputs → Processing → Outputs)

```
INPUTS:
├─ SQLite Database (box_scores, salaries, schedule, injuries)
├─ YAML Configs (feature pipelines, model params, experiment settings)
└─ Runtime Parameters (date ranges, filters, hyperparameters)
        ↓
PROCESSING PIPELINE:
├─ 1. Data Loading → Query SQLite for historical data
├─ 2. Feature Engineering → Rolling stats, EWMA transformations
├─ 3. Model Training → XGBoost/RandomForest on features
├─ 4. Prediction Generation → Infer on test slate
├─ 5. Evaluation → Calculate metrics vs actuals
└─ 6. Aggregation & Reporting → Combine results across slates
        ↓
OUTPUTS:
├─ Predictions (.parquet) → Model projections
├─ Models (.pkl) → Trained model objects
├─ Training Inputs (.parquet) → Feature matrices used
└─ Reports (.html/.pdf) → Summary and visualizations
```

---

## Decision Tree: Model Recalibration

**Logic for each slate:**

```
if rewrite_models == True:
    ├─ ALWAYS RETRAIN (forced)
    └─ Use updated model for this slate
elif first_slate:
    ├─ TRAIN new model
    ├─ Set last_training_date = current_date
    └─ Use trained model
elif (current_date - last_training_date) >= recalibrate_days:
    ├─ RETRAIN model
    ├─ Update last_training_date = current_date
    └─ Use retrained model
else:
    ├─ REUSE model from previous slate
    └─ Use same predictions logic (no retraining)
```

**Default**: rewrite_models=False, recalibrate_days=7 (retrain weekly)

---

## Directory Structure: Outputs

```
data/outputs/
└── {run_timestamp}/               # e.g., 20250205_143022
    ├── inputs/
    │   ├── slate_training_inputs_20250205.parquet
    │   ├── slate_training_inputs_20250206.parquet
    │   └── player_Nikola_Jokic_*.parquet        (if per-player)
    │
    ├── features/
    │   └── (generated feature files)
    │
    ├── predictions/
    │   ├── 20250205.parquet                      # predictions only
    │   ├── 20250205_with_actuals.parquet        # + actual results
    │   ├── 20250206.parquet
    │   └── 20250206_with_actuals.parquet
    │
    ├── checkpoints/
    │   ├── 20250205.json                        # resumption data
    │   └── 20250206.json
    │
    └── reports/
        ├── backtest_report.html                 # Main report
        ├── backtest_analysis.pdf                # PDF export
        ├── backtest_config.json                 # Runtime config
        ├── results_summary.json                 # Aggregated stats
        └── charts/
            ├── mape_timeline.html
            ├── salary_tier_analysis.html
            ├── daily_results_table.html
            ├── model_vs_benchmark.html
            └── correlation_scatter.html

models/
├── per_slate_models/
│   ├── xgboost_20250205.pkl
│   ├── xgboost_20250205.json
│   ├── xgboost_20250206.pkl
│   └── xgboost_20250206.json
│
└── per_player_models/              (if per-player)
    ├── player_Nikola_Jokic_28908111729.pkl
    ├── player_LeBron_James_*.pkl
    └── ...
```

---

## Key Parameters Flow

**Configuration inputs flow:**

1. **Training Period** (train_start, train_end)
   → Load Training Data → Historical player logs

2. **Testing Period** (test_start, test_end)
   → Load Test Dates → Slate-by-slate iteration

3. **Feature Config** (YAML file)
   → Load FeaturePipeline → Build transformers

4. **Model Config** (type, hyperparams, GPU)
   → Initialize Model → XGBoost or RandomForest

5. **Execution Settings** (n_jobs, save_*, recalibrate_days)
   → Set execution params → Parallel/sequential modes

6. **Filters** (salary, injury, player lists)
   → Apply filters → Per-slate eligibility

**All inputs** → Backtest Pipeline → **Outputs**: Predictions, Models, Reports

---

## Temporal Validation (Lookahead Bias Prevention)

**Timeline of data access:**

```
TRAINING PHASE (Oct 1, 2024 → Jan 31, 2025)
├─ Load historical games UP TO Jan 31, 2025
├─ Validation: max(gameDate) < train_end? ✓
└─ Fit feature transformers

VALIDATION CHECK (At Feb 5, 2025 test_start)
├─ Verify: max(training_data.gameDate) < train_end
├─ If violated → LOOKAHEAD BIAS DETECTED ❌ (halt)
└─ If OK → Proceed ✓

TEST PHASE (Feb 5-15, 2025)
├─ Slate 1 (Feb 5): Use training data BEFORE Feb 5 only
├─ Slate 2 (Feb 6): Use training data BEFORE Feb 6 only
├─ Slate N (Feb 15): Use training data BEFORE Feb 15 only
└─ Result: Realistic out-of-sample evaluation

PRINCIPLE: Features for each test date only use data from BEFORE that date
```

---

## Error Handling Flow

**Error types and recovery strategies:**

| Error Type | Condition | Recovery | Log Level |
|-----------|-----------|----------|-----------|
| **Data Loading** | Database error | Halt backtest if critical; skip slate if not | ERROR/WARNING |
| **GPU OOM** | Model training runs out of GPU memory | Fallback to CPU, retry training | WARNING |
| **Model Convergence** | Model fails to converge | Use default hyperparameters, continue | WARNING |
| **Insufficient Data** | Too few samples for player/slate | Skip player or slate | DEBUG |
| **Prediction Failure** | Model inference error | Use benchmark predictions instead | WARNING |
| **Metric Calculation** | Metric computation error | Use default value, continue | WARNING |

**Error Priority**:
- **CRITICAL** (Halt) → Missing training data, lookahead bias detected
- **HIGH** (Skip) → Missing test data, no eligible players
- **MEDIUM** (Fallback) → GPU memory, model convergence, insufficient samples
- **LOW** (Continue) → Prediction/evaluation errors with fallback available

**Logging**: DEBUG → sequential processing details, INFO → major milestones, WARNING → recoverable issues, ERROR → fatal problems

---

## Performance & Resource Usage

### Memory Consumption
- **Training Data**: ~500MB (50k player-games)
- **Feature Cache**: ~1GB (if enabled)
- **Models**: ~100MB (per-slate) to ~10GB (per-player)
- **Total**: 1-15GB depending on configuration

### Computation Time
**Per-Slate Model**:
- Feature building: 5-10 seconds
- Model training: 10-30 seconds
- Prediction: 1-2 seconds
- Total: ~30 seconds per slate

**Per-Player Models (32 cores)**:
- Training 500+ models: 60-120 seconds
- Prediction: 5-10 seconds
- Total: ~90 seconds per slate

### Storage Requirements
**Per Slate**:
- Predictions: 100-500KB
- With actuals: 150-700KB
- Training inputs: 5-50MB
- Models: 1-50MB

**Per 100 Slates**:
- Total: 1-10GB depending on model type and save options
