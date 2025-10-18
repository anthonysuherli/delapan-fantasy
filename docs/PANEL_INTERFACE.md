# Panel Backtest Interface Documentation

Interactive web-based control center for configuring and monitoring walk-forward backtests with real-time streaming results using Panel framework.

## Overview

The Panel interface (`src/interface/panel_backtest_app.py`) provides a dark-themed browser UI for:
- Interactive backtest configuration via left sidebar
- Real-time results streaming in main panel
- Live log monitoring in collapsible bottom panel
- Experiment preset management from YAML files
- Background execution with thread-safe queues

## Quick Start

### Installation

Panel and dependencies required:

```bash
pip install panel>=1.3.0
```

### Launch

From project root:

```bash
panel serve src/interface/panel_backtest_app.py --show
```

Browser opens automatically to http://localhost:5006

Alternative port:
```bash
panel serve src/interface/panel_backtest_app.py --port 8080 --show
```

### Prerequisites

Before starting:

1. **SQLite Database**: Must exist with collected data
   ```bash
   python scripts/collect_games.py --start-date 20241201 --end-date 20250101
   python scripts/collect_dfs_salaries.py --start-date 20241201 --end-date 20250101
   ```

2. **Feature Configuration**: Must exist in config/features/
   - default_features.yaml (21 statistics, 147 features)
   - base_features.yaml (6 core statistics)

3. **Training Data**: Date range must have player game logs

## User Interface

### Layout Architecture

```
┌─────────────────────────────────────────────────────────┐
│         NBA DFS Backtest Control Center                 │
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│  Sidebar     │  Main Panel: Backtest Results            │
│  (350px)     │  ┌────────────────────────────────────┐  │
│              │  │ Status Text                        │  │
│  ● Experiment│  │ Results Table (Tabulator)          │  │
│    Preset    │  │ - Metric | Value                   │  │
│              │  │ - total_slates | 31                │  │
│  ● Data      │  │ - mean_mape | 75.34                │  │
│    Config    │  └────────────────────────────────────┘  │
│              │                                          │
│  ● Date      │  ┌────────────────────────────────────┐  │
│    Ranges    │  │ Final Summary                      │  │
│              │  │ - Formatted metrics display        │  │
│  ● Model     │  └────────────────────────────────────┘  │
│    Config    │                                          │
│              ├──────────────────────────────────────────┤
│  ● Model     │  Bottom Panel: Execution Logs (300px)    │
│    Params    │  ┌────────────────────────────────────┐  │
│  (JSON)      │  │ [08:32:45] INFO: Starting...       │  │
│              │  │ [08:32:47] INFO: Loading data...   │  │
│  ● Training  │  │ [08:33:15] INFO: Training models...│  │
│    Options   │  │ [08:34:32] INFO: Results for date  │  │
│              │  │ (Auto-scroll, last 100 logs)       │  │
│  ● Output    │  └────────────────────────────────────┘  │
│    Options   │                                          │
│              │                                          │
│  [Run        │                                          │
│   Backtest]  │                                          │
└──────────────┴──────────────────────────────────────────┘
```

### Sidebar: Configuration Panel

**Experiment Presets**
- Dropdown: Load pre-configured YAML from `config/experiments/`
- Options: 'Manual configuration' (default) + all experiment YAML files
- Auto-populates all form fields on selection

**Data Configuration**
- Database path (default: "nba_dfs.db")
- Data directory (optional: separate Parquet storage)

**Date Ranges** (YYYYMMDD format)
- Train start (e.g., "20241001")
- Train end (e.g., "20241130")
- Test start (e.g., "20241201")
- Test end (e.g., "20241215")

**Model Configuration**
- Model type: xgboost | random_forest
- Feature config: default_features | base_features
- Output directory: Result storage path

**Model Hyperparameters**
- JSON code editor (Monokai theme, 200px height)
- Example:
  ```json
  {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8
  }
  ```
- Validates JSON on submission

**Training Options**
- Per-player models: Train individual model per player (checkbox)
- Recalibrate cadence: Days between retraining (1-30, default: 7)
- Minutes threshold: Min playing time filter (0-48, default: 12)
- Parallel jobs: CPU cores for parallelization (1-16, default: 1)

**Output Options**
- Save predictions: Store predictions to disk (checkbox)
- Save models: Store trained models for reuse (checkbox)

**Submit**
- Run Backtest: Launch backtest in background thread (primary button)

### Main Panel: Results Streaming

**Status Text**
- Dynamic status: "Waiting for backtest to start..." → "Running..." → "Completed"

**Results Table (Tabulator)**
- Displays final summary metrics as key-value pairs
- Columns: Metric | Value
- Features:
  - Local pagination (20 rows/page)
  - Fit columns to width
  - 400px height

**Final Summary Section**
- Rendered after backtest completes
- Formatted Markdown display:
  ```
  ## Final Summary

  **Total Slates**: 31
  **Total Players**: 12847
  **Coverage**: 96.40%
  **Mean MAPE**: 75.34%
  **Median RMSE**: 12.12
  **Overall Correlation**: 0.728
  ```

### Bottom Panel: Execution Logs

**Log Display**
- Markdown pane with monospace font
- Dark background (#1e1e1e), light text (#e0e0e0)
- 300px height, vertical scroll
- Format: `[HH:MM:SS] **LEVEL**: message`
- Displays last 100 log events
- Auto-updates every 1 second via periodic callback

**Collapsible**
- Can be collapsed to save screen space
- Default: expanded

## State Management

### Application State

BacktestApp class manages:
- `runner`: Active BacktestRunner instance or None
- `logs`: List of log event dictionaries
- `daily_results`: List of per-slate results (currently unused, reserved for future)
- `final_summary`: Final aggregated results dictionary
- `config`: Current configuration dictionary
- `model_params`: Current model hyperparameters dictionary

### Event Loop

`_drain_runner_events()` runs every 1 second:
1. Check if runner exists
2. Consume logs from `runner.log_queue`
3. Append to `self.logs`
4. Update log display with last 100 logs
5. Consume results from `runner.result_queue`
6. Route to `final_summary` based on event type
7. Update results table if final summary received
8. Clean up runner if finished

### Panel Reactive Updates

Panel automatically re-renders components when:
- Widget values change
- Bound functions return new values
- Component properties are updated

## Background Execution: BacktestRunner

### Thread Safety

BacktestRunner uses thread-safe queues:

```python
class BacktestRunner:
    log_queue: queue.Queue        # Log events
    result_queue: queue.Queue     # Result events
    _thread: Optional[Thread]     # Daemon thread
    _error: Optional[str]         # Error message
```

### Execution Flow

1. **Initialization**: Create BacktestRunner with config dict
2. **Start**: Launch daemon thread via `start()`
3. **Execution**:
   - Attach `QueueLogHandler` to root logger
   - Execute `WalkForwardBacktest.run()`
   - Stream logs to `log_queue`
   - Stream results to `result_queue`
4. **Completion**: Queue final status, remove handler
5. **Error Handling**: Catch exceptions, emit error to log_queue

### Queue Communication

**Log Events**
```python
{
    'timestamp': '08:32:45',
    'level': 'INFO',
    'message': 'Starting backtest run...'
}
```

**Result Events**
```python
# Final summary
{
    'type': 'final_summary',
    'payload': {
        'total_slates': 31,
        'total_players': 12847,
        'coverage': 0.964,
        'mean_mape': 75.34,
        ...
    }
}

# Completion marker
{
    'status': 'finished'
}
```

## Experiment Configuration

### YAML Format

File: `config/experiments/baseline_backtest.yaml`

```yaml
name: "Baseline Backtest - XGBoost with Standard Features"
description: "Initial backtest using XGBoost with rolling window features"
version: "1.0.0"

data:
  train_start: "20231101"
  train_end: "20240229"
  test_start: "20240301"
  test_end: "20240331"

model:
  type: "xgboost"
  params:
    max_depth: 6
    learning_rate: 0.05
    n_estimators: 200
    min_child_weight: 5
    subsample: 0.8
    colsample_bytree: 0.8

evaluation:
  output_dir: "data/backtest_results"
```

### Loading Process

1. User selects experiment from dropdown
2. `_on_experiment_change()` triggered
3. Load YAML file from `config/experiments/`
4. Parse `data`, `model`, `evaluation` sections
5. Update all widget values
6. Log confirmation

### Supported Fields

**Data Section**
- `train_start`, `train_end`, `test_start`, `test_end`

**Model Section**
- `type`: Model class name
- `params`: Hyperparameter dictionary

**Evaluation Section**
- `output_dir`: Results storage path

## Panel Interface Features

| Feature | Implementation |
|---------|-----------------|
| **Framework** | Panel/Bokeh |
| **Theme** | Dark mode only |
| **Layout** | Collapsible bottom panel |
| **Training Sample** | Not implemented (future) |
| **Results Display** | Tabulator + Markdown summary |
| **Log Display** | Markdown with custom styling |
| **Daily Results** | Reserved for future |
| **Event Loop** | Periodic callback (1s) |
| **State** | Class instance variables |
| **Reactive** | Explicit (periodic callbacks) |
| **Deployment** | `panel serve` |

## Troubleshooting

### Backtest Never Starts

**Symptoms**: Click "Run Backtest", logs show no activity

**Causes**:
- SQLite database doesn't exist or is locked
- Training date range has no data
- Feature config file not found
- Invalid JSON in model parameters

**Solutions**:
1. Verify database:
   ```bash
   ls -la nba_dfs.db
   ```
2. Validate JSON in model parameters editor
3. Check feature config exists:
   ```bash
   ls config/features/default_features.yaml
   ```

### Logs Not Updating

**Symptoms**: Log panel frozen, no new messages

**Causes**:
- Backtest hung or extremely slow
- Queue communication issue
- Exception in runner thread

**Solutions**:
1. Check system resources (CPU/memory)
2. Refresh browser page
3. Kill and restart Panel server
4. Check terminal output for exceptions

### Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Kill existing Panel process
pkill -f "panel serve"

# Or use different port
panel serve src/interface/panel_backtest_app.py --port 8080
```

### Dark Theme Not Applied

**Symptoms**: UI appears in light mode

**Solutions**:
1. Verify `pn.extension()` includes `theme='dark'`
2. Verify template uses `theme='dark'`
3. Hard refresh browser (Ctrl+Shift+R)

## Performance Considerations

### Memory Usage

- Logs list grows unbounded (last 100 displayed, all stored)
- Consider clearing old logs after threshold:
  ```python
  if len(self.logs) > 1000:
      self.logs = self.logs[-100:]
  ```

### UI Responsiveness

- Periodic callback runs every 1 second
- Heavy backtest phases may delay UI updates
- Normal behavior: logs stream continuously, UI remains responsive

### Optimization Tips

1. **Start small**: Test with 5-day range before full season
2. **Use base_features**: 6 features vs 147 for faster iteration
3. **Disable per_player_models**: Slate-level trains ~50x faster
4. **Limit parallel jobs**: n_jobs=1 recommended for stability

## Development

### Code Structure

```
src/interface/panel_backtest_app.py
├── QueueLogHandler          # Logging → queue bridge
├── BacktestRunner           # Background worker
│   ├── start()              # Launch daemon thread
│   ├── is_running()         # Check execution status
│   ├── consume_logs()       # Get buffered logs
│   ├── consume_results()    # Get buffered results
│   └── _run()               # Execute backtest
├── BacktestApp              # Main application
│   ├── _create_widgets()    # Build UI components
│   ├── _create_layout()     # Assemble layout
│   ├── _list_experiments()  # Scan YAML files
│   ├── _on_experiment_change()      # Load preset
│   ├── _on_run_backtest()           # Start execution
│   ├── _drain_runner_events()       # Poll queues
│   ├── _update_results_table()      # Update Tabulator
│   ├── _render_summary()            # Render Markdown
│   └── view()                       # Return template
└── app.view().servable()    # Entry point
```

### Adding UI Controls

Example: Add new integer input to sidebar

```python
# In _create_widgets()
self.new_param_input = pn.widgets.IntInput(
    name='New Parameter',
    value=10,
    start=1,
    end=100
)

# In _create_layout() sidebar
self.sidebar = pn.Column(
    # ... existing widgets ...
    self.new_param_input,
    # ... rest of sidebar ...
)

# In _on_run_backtest()
config = {
    # ... existing config ...
    'new_parameter': self.new_param_input.value
}
```

### Extending Results Display

To add daily results streaming (currently reserved):

1. Update `WalkForwardBacktest` to yield daily results
2. In `BacktestRunner._run()`, emit daily events:
   ```python
   for daily_result in backtest.run_streaming():
       self.result_queue.put({
           'type': 'daily_result',
           'payload': daily_result
       })
   ```
3. In `BacktestApp._drain_runner_events()`, handle daily events:
   ```python
   if result_event.get('type') == 'daily_result':
       self.daily_results.append(result_event['payload'])
       self._update_daily_table()
   ```
4. Add daily results Tabulator widget to main panel

## Version Compatibility

- **Panel**: 1.3.0+ (uses periodic callbacks)
- **Python**: 3.8+
- **OS**: Linux, macOS, Windows

## See Also

- [Panel Documentation](https://panel.holoviz.org/)
- [WalkForwardBacktest](../CLAUDE.md#evaluation-srcevaluation)
- [Feature Configuration](../CLAUDE.md#features-srcfeatures)
- [Streamlit Interface](STREAMLIT_INTERFACE.md) - Alternative UI
- [scripts/run_backtest.py](../scripts/README.md) - CLI alternative
