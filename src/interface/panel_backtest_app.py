"""
Panel-based Interactive Backtest Interface

Provides browser-based UI for configuring and monitoring walk-forward backtests
with real-time streaming of logs and results.
"""

import sys
import os
from pathlib import Path

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import panel as pn
import pandas as pd
import numpy as np
import logging
import json
import yaml
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.evaluation.backtest import WalkForwardBacktest
from src.filters import InjuryFilter, ColumnFilter
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Panel with dark theme
pn.extension(
    'tabulator',
    sizing_mode='stretch_width',
    design='material',
    theme='dark',
    raw_css=["""
        * {
            font-family: 'Hack', 'Consolas', 'Monaco', monospace !important;
        }
        .bk-root .bk-panel-models-layout-Column,
        .bk-root .bk-panel-models-layout-Row,
        .bk-root .bk {
            border-radius: 8px !important;
        }
        .bk-input,
        .bk-btn,
        .bk-input-group,
        textarea,
        input[type="text"],
        select {
            border-radius: 6px !important;
        }

        /* Terminal-like log display */
        #log-terminal {
            background: #0d1117 !important;
            color: #58a6ff !important;
            font-family: 'Hack', 'Consolas', 'Monaco', monospace !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
            padding: 12px !important;
            border-radius: 8px !important;
            border: 1px solid #30363d !important;
            overflow-y: auto !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
        }

        #log-terminal .log-info {
            color: #58a6ff;
        }

        #log-terminal .log-error {
            color: #f85149;
        }

        #log-terminal .log-warning {
            color: #d29922;
        }

        #log-terminal .log-timestamp {
            color: #8b949e;
        }

        /* Floating terminal panel */
        #floating-terminal-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 50vh;
            min-height: 100px;
            max-height: 90vh;
            background: #0d1117;
            border-top: 2px solid #30363d;
            border-radius: 8px 8px 0 0;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.5);
            resize: vertical;
            overflow: hidden;
        }

        #floating-terminal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
            border-radius: 8px 8px 0 0;
            user-select: none;
            cursor: ns-resize;
        }

        #floating-terminal-header:active {
            cursor: grabbing;
        }

        #floating-terminal-header h3 {
            margin: 0;
            color: #58a6ff;
            font-size: 14px;
            font-weight: 600;
        }

        #floating-terminal-controls {
            display: flex;
            gap: 8px;
        }

        .terminal-btn {
            background: #21262d;
            border: 1px solid #30363d;
            color: #58a6ff;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }

        .terminal-btn:hover {
            background: #30363d;
            border-color: #58a6ff;
        }

        #floating-terminal-content {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 12px 16px;
        }

        #floating-terminal-log {
            color: #58a6ff;
            font-family: 'Hack', 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        #floating-terminal-log .log-info {
            color: #58a6ff;
        }

        #floating-terminal-log .log-error {
            color: #f85149;
        }

        #floating-terminal-log .log-warning {
            color: #d29922;
        }

        #floating-terminal-log .log-timestamp {
            color: #8b949e;
        }

        /* Adjust main content when floating terminal is visible */
        body.terminal-open {
            overflow: hidden;
        }

        #main-content {
            transition: margin-bottom 0.3s;
        }

        #main-content.terminal-open {
            margin-bottom: 50vh;
        }
    """]
)

# Constants
EXPERIMENT_DIR = Path("config/experiments")
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "data/backtest_results"


class QueueLogHandler(logging.Handler):
    """Logging handler that pushes log records to a thread-safe queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Ensure message is properly encoded
            if isinstance(msg, bytes):
                msg = msg.decode('utf-8', errors='replace')
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            self.log_queue.put({
                'timestamp': timestamp,
                'level': record.levelname,
                'message': msg
            })
        except Exception as e:
            # Fallback with ASCII-safe error message
            try:
                self.log_queue.put({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'level': 'ERROR',
                    'message': f'Log encoding error: {repr(e)}'
                })
            except:
                pass


class BacktestRunner:
    """Background worker that executes WalkForwardBacktest and streams logs/results."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    def start(self):
        """Launch backtest in daemon thread"""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        """Check if backtest is still executing"""
        return self._thread is not None and self._thread.is_alive()

    def consume_logs(self) -> List[Dict]:
        """Get buffered logs (non-blocking)"""
        logs = []
        while True:
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    def consume_results(self) -> List[Dict]:
        """Get buffered results (non-blocking)"""
        results = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    @property
    def error(self) -> Optional[str]:
        """Error message if execution failed"""
        return self._error

    def _run(self):
        """Execute backtest with logging/result streaming"""
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        try:
            self.log_queue.put({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'level': 'INFO',
                'message': 'Starting backtest run...'
            })

            backtest = WalkForwardBacktest(**self.config)
            results = backtest.run()

            self.result_queue.put({
                'type': 'final_summary',
                'payload': results
            })

            self.log_queue.put({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'level': 'INFO',
                'message': 'Backtest completed successfully'
            })

        except Exception as e:
            # Handle Unicode in error messages
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            self._error = error_msg
            self.log_queue.put({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'level': 'ERROR',
                'message': f'Backtest failed: {error_msg}'
            })
            logger.exception("Backtest execution error")

        finally:
            root_logger.removeHandler(handler)
            self.result_queue.put({'status': 'finished'})


# Global state
runner = None
logs = []
final_summary = None

# Configuration state
config = {
    'data_dir': DEFAULT_DATA_DIR,
    'train_start': '20241001',
    'train_end': '20241130',
    'test_start': '20241201',
    'test_end': '20241215',
    'model_type': 'xgboost',
    'feature_config': 'default_features',
    'output_dir': DEFAULT_OUTPUT_DIR,
    'per_player_models': True,
    'recalibrate_days': 7,
    'minutes_threshold': 12,
    'n_jobs': 1,
    'save_predictions': True,
    'save_models': True
}

model_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Create widgets
data_dir_input = pn.widgets.TextInput(name='Data Directory', value=config['data_dir'], width=300)
train_start_input = pn.widgets.TextInput(name='Train Start (YYYYMMDD)', value=config['train_start'], width=300)
train_end_input = pn.widgets.TextInput(name='Train End (YYYYMMDD)', value=config['train_end'], width=300)
test_start_input = pn.widgets.TextInput(name='Test Start (YYYYMMDD)', value=config['test_start'], width=300)
test_end_input = pn.widgets.TextInput(name='Test End (YYYYMMDD)', value=config['test_end'], width=300)

model_type_select = pn.widgets.Select(name='Model Type', options=['xgboost', 'random_forest'], value=config['model_type'], width=300)
feature_config_select = pn.widgets.Select(name='Feature Config', options=['default_features', 'base_features'], value=config['feature_config'], width=300)

model_params_input = pn.widgets.TextAreaInput(
    name='Model Hyperparameters (JSON)',
    value=json.dumps(model_params, indent=2),
    height=200,
    width=300
)

per_player_checkbox = pn.widgets.Checkbox(name='Per-player models', value=config['per_player_models'])
recalibrate_input = pn.widgets.IntInput(name='Recalibrate cadence (days)', value=config['recalibrate_days'], start=1, end=30, width=300)
minutes_input = pn.widgets.IntInput(name='Minutes threshold', value=config['minutes_threshold'], start=0, end=48, width=300)

# Filter widgets
injury_exclude_out = pn.widgets.Checkbox(name='Exclude OUT players', value=True, width=300)
injury_exclude_doubtful = pn.widgets.Checkbox(name='Exclude DOUBTFUL players', value=False, width=300)
injury_exclude_questionable = pn.widgets.Checkbox(name='Exclude QUESTIONABLE players', value=False, width=300)

# Column filter
enable_salary_filter = pn.widgets.Checkbox(name='Enable salary filter', value=False, width=300)
salary_min = pn.widgets.IntInput(name='Min salary', value=3000, start=0, end=15000, step=100, width=300)

output_dir_input = pn.widgets.TextInput(name='Output Directory', value=config['output_dir'], width=300)
save_predictions_checkbox = pn.widgets.Checkbox(name='Save predictions', value=config['save_predictions'])
save_models_checkbox = pn.widgets.Checkbox(name='Save models', value=config['save_models'])

# Results display
status_text = pn.pane.Markdown('**Status:** Waiting for backtest to start...', width=600)

# Dataset info display
dataset_info_pane = pn.pane.Markdown(
    '',
    width=950,
    styles={
        'background': '#1a1a1a',
        'padding': '20px',
        'border-radius': '8px',
        'border': '1px solid #444'
    }
)

summary_pane = pn.pane.Markdown('', width=600)

# Log display - terminal-like with auto-scroll
log_display = pn.pane.HTML(
    '<div id="log-terminal" style="height: 300px; width: 100%; overflow-y: auto;"></div>',
    height=300,
    width=1000,
    sizing_mode='stretch_width'
)


def on_run_backtest(event):
    """Start backtest execution"""
    global runner, logs, final_summary

    if runner and runner.is_running():
        logger.warning("Backtest already running")
        return

    # Collect configuration
    try:
        params = json.loads(model_params_input.value)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid model parameters JSON: {e}")
        return

    # Build filters
    filters = []

    # Injury filter
    if injury_exclude_out.value or injury_exclude_doubtful.value or injury_exclude_questionable.value:
        filters.append(InjuryFilter(
            exclude_out=injury_exclude_out.value,
            exclude_doubtful=injury_exclude_doubtful.value,
            exclude_questionable=injury_exclude_questionable.value
        ))

    # Salary filter
    if enable_salary_filter.value:
        filters.append(ColumnFilter('salary', '>=', salary_min.value))

    run_config = {
        'train_start': train_start_input.value,
        'train_end': train_end_input.value,
        'test_start': test_start_input.value,
        'test_end': test_end_input.value,
        'model_type': model_type_select.value,
        'model_params': params,
        'feature_config': feature_config_select.value,
        'output_dir': output_dir_input.value,
        'data_dir': data_dir_input.value,
        'per_player_models': per_player_checkbox.value,
        'recalibrate_days': recalibrate_input.value,
        'minutes_threshold': minutes_input.value,
        'n_jobs': 1,
        'save_predictions': save_predictions_checkbox.value,
        'save_models': save_models_checkbox.value,
        'player_filters': filters if filters else None
    }

    # Reset state
    logs = []
    final_summary = None
    dataset_info_pane.object = ''
    summary_pane.object = ''
    log_display.object = '<div id="log-terminal" style="height: 300px; width: 100%; overflow-y: auto;"></div>'
    status_text.object = '**Status:** Backtest running...'

    # Start runner
    runner = BacktestRunner(run_config)
    runner.start()

    logger.info("Backtest started")


def update_display():
    """Poll runner queues and update UI"""
    global runner, logs, final_summary

    if not runner:
        return

    # Consume logs
    new_logs = runner.consume_logs()
    if new_logs:
        logs.extend(new_logs)

        # Build terminal-style HTML with color coding for main display
        log_lines = []
        for log in logs[-100:]:  # Last 100 logs
            level = log['level']
            level_class = f"log-{level.lower()}"
            # HTML escape and handle Unicode
            message = log["message"].replace('<', '&lt;').replace('>', '&gt;')
            log_lines.append(
                f'<span class="log-timestamp">[{log["timestamp"]}]</span> '
                f'<span class="{level_class}">{level:8s}</span> '
                f'{message}'
            )

        log_html = '<br>'.join(log_lines)

        # Update main log display with auto-scroll JavaScript
        log_display.object = f'''
        <div id="log-terminal" style="height: 300px; width: 100%; overflow-y: auto;">
            {log_html}
        </div>
        <script>
            const logTerminal = document.getElementById('log-terminal');
            if (logTerminal) {{
                logTerminal.scrollTop = logTerminal.scrollHeight;
            }}
        </script>
        '''

        # Build floating terminal HTML with all logs and auto-show when logs arrive
        floating_terminal_logs = []
        for log in logs[-200:]:  # More history in floating terminal
            level = log['level']
            level_class = f"log-{level.lower()}"
            message = log["message"].replace('<', '&lt;').replace('>', '&gt;')
            floating_terminal_logs.append(
                f'<span class="log-timestamp">[{log["timestamp"]}]</span> '
                f'<span class="{level_class}">{level:8s}</span> '
                f'{message}'
            )

        floating_log_html = '<br>'.join(floating_terminal_logs)

        # Update floating terminal with auto-scroll and auto-show
        floating_terminal_html.object = f'''
        <div id="floating-terminal-container" style="display: flex;">
            <div id="floating-terminal-header">
                <h3>⚡ Live Terminal</h3>
                <div id="floating-terminal-controls">
                    <button class="terminal-btn" onclick="clearTerminal()">Clear</button>
                    <button class="terminal-btn" onclick="toggleTerminal()">Minimize</button>
                </div>
            </div>
            <div id="floating-terminal-content">
                <div id="floating-terminal-log">{floating_log_html}</div>
            </div>
        </div>
        <script>
            window.terminalVisible = true;
            document.body.classList.add('terminal-open');
            const container = document.getElementById('floating-terminal-content');
            if (container) {{
                container.scrollTop = container.scrollHeight;
            }}

            function showTerminal() {{
                const container = document.getElementById('floating-terminal-container');
                if (container) {{
                    container.style.display = 'flex';
                    window.terminalVisible = true;
                    document.body.classList.add('terminal-open');
                }}
            }}

            function hideTerminal() {{
                const container = document.getElementById('floating-terminal-container');
                if (container) {{
                    container.style.display = 'none';
                    window.terminalVisible = false;
                    document.body.classList.remove('terminal-open');
                }}
            }}

            function toggleTerminal() {{
                window.terminalVisible ? hideTerminal() : showTerminal();
            }}

            function clearTerminal() {{
                const logDiv = document.getElementById('floating-terminal-log');
                if (logDiv) {{
                    logDiv.innerHTML = '';
                }}
            }}

            // Draggable resize
            let isResizing = false;
            let startY = 0;
            let startHeight = 0;

            const header = document.getElementById('floating-terminal-header');
            const container = document.getElementById('floating-terminal-container');

            if (header && container) {{
                header.addEventListener('mousedown', (e) => {{
                    isResizing = true;
                    startY = e.clientY;
                    startHeight = container.offsetHeight;
                    document.body.style.cursor = 'ns-resize';
                    e.preventDefault();
                }});

                document.addEventListener('mousemove', (e) => {{
                    if (!isResizing) return;
                    const deltaY = startY - e.clientY;
                    const newHeight = Math.min(Math.max(startHeight + deltaY, 100), window.innerHeight * 0.9);
                    container.style.height = newHeight + 'px';
                }});

                document.addEventListener('mouseup', () => {{
                    if (isResizing) {{
                        isResizing = false;
                        document.body.style.cursor = '';
                    }}
                }});
            }}
        </script>
        '''

    # Consume results
    new_results = runner.consume_results()
    for result_event in new_results:
        if result_event.get('type') == 'final_summary':
            final_summary = result_event.get('payload')

            # Update status
            status_text.object = '**Status:** Backtest completed'

            # Update dataset info
            dataset_info = f"""
## Dataset & Model Information

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 15px;">

<div style="background: #2a2a2a; padding: 15px; border-radius: 8px; border-left: 4px solid #58a6ff;">
<h4 style="margin: 0 0 10px 0; color: #58a6ff;">Dataset Coverage</h4>
<p style="font-size: 24px; font-weight: bold; margin: 5px 0; color: #e0e0e0;">{final_summary.get('total_players', 'N/A'):,}</p>
<p style="margin: 0; color: #8b949e; font-size: 13px;">Unique Players</p>
<p style="font-size: 20px; font-weight: bold; margin: 10px 0 5px 0; color: #e0e0e0;">{final_summary.get('total_slates', 'N/A')}</p>
<p style="margin: 0; color: #8b949e; font-size: 13px;">Slates Processed</p>
</div>

<div style="background: #2a2a2a; padding: 15px; border-radius: 8px; border-left: 4px solid #d29922;">
<h4 style="margin: 0 0 10px 0; color: #d29922;">Model Performance</h4>
<p style="font-size: 24px; font-weight: bold; margin: 5px 0; color: #e0e0e0;">{final_summary.get('mean_mape', 0):.1f}%</p>
<p style="margin: 0; color: #8b949e; font-size: 13px;">Mean MAPE</p>
<p style="font-size: 20px; font-weight: bold; margin: 10px 0 5px 0; color: #e0e0e0;">{final_summary.get('overall_correlation', 0):.3f}</p>
<p style="margin: 0; color: #8b949e; font-size: 13px;">Correlation</p>
</div>

<div style="background: #2a2a2a; padding: 15px; border-radius: 8px; border-left: 4px solid #3fb950;">
<h4 style="margin: 0 0 10px 0; color: #3fb950;">Coverage & Quality</h4>
<p style="font-size: 24px; font-weight: bold; margin: 5px 0; color: #e0e0e0;">{final_summary.get('coverage', 0):.1%}</p>
<p style="margin: 0; color: #8b949e; font-size: 13px;">Model Coverage</p>
<p style="font-size: 20px; font-weight: bold; margin: 10px 0 5px 0; color: #e0e0e0;">{final_summary.get('median_rmse', 0):.2f}</p>
<p style="margin: 0; color: #8b949e; font-size: 13px;">Median RMSE</p>
</div>

</div>

<div style="margin-top: 20px; padding: 15px; background: #1e1e1e; border-radius: 8px;">
<h4 style="margin: 0 0 10px 0; color: #58a6ff;">Configuration</h4>
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 13px;">
<div><span style="color: #8b949e;">Model Type:</span> <span style="color: #e0e0e0;">{model_type_select.value}</span></div>
<div><span style="color: #8b949e;">Feature Config:</span> <span style="color: #e0e0e0;">{feature_config_select.value}</span></div>
<div><span style="color: #8b949e;">Per-Player Models:</span> <span style="color: #e0e0e0;">{'Yes' if per_player_checkbox.value else 'No'}</span></div>
<div><span style="color: #8b949e;">Recalibrate Days:</span> <span style="color: #e0e0e0;">{recalibrate_input.value}</span></div>
<div><span style="color: #8b949e;">Min Minutes:</span> <span style="color: #e0e0e0;">{minutes_input.value}</span></div>
<div><span style="color: #8b949e;">Filters Active:</span> <span style="color: #e0e0e0;">{'Yes' if (injury_exclude_out.value or enable_salary_filter.value) else 'No'}</span></div>
</div>
</div>
            """
            dataset_info_pane.object = dataset_info

            # Update performance summary
            summary_text = f"""
## Performance Breakdown

**Elite Players ($8k+)**: {final_summary.get('elite_mape', 'N/A')}% MAPE
**Mid-Tier ($5k-$8k)**: {final_summary.get('mid_mape', 'N/A')}% MAPE
**Budget (<$5k)**: {final_summary.get('budget_mape', 'N/A')}% MAPE

**Total Predictions**: {final_summary.get('total_predictions', 'N/A'):,}
**Models Trained**: {final_summary.get('models_trained', 'N/A'):,}
            """
            summary_pane.object = summary_text

        elif result_event.get('status') == 'finished':
            if runner and runner.error:
                logger.error(f"Backtest failed: {runner.error}")
                status_text.object = f'**Status:** Backtest failed - {runner.error}'
            runner = None


run_button = pn.widgets.Button(name='Run Backtest', button_type='primary', width=300)
run_button.on_click(on_run_backtest)

# Periodic callback
pn.state.add_periodic_callback(update_display, period=1000)

# Layout
sidebar = pn.Column(
    '## Configuration',
    '---',
    '### Data',
    data_dir_input,
    train_start_input,
    train_end_input,
    test_start_input,
    test_end_input,
    '### Model',
    model_type_select,
    feature_config_select,
    model_params_input,
    '### Training',
    per_player_checkbox,
    recalibrate_input,
    minutes_input,
    '### Filters',
    pn.pane.Markdown('**Injury Filter:**'),
    injury_exclude_out,
    injury_exclude_doubtful,
    injury_exclude_questionable,
    pn.pane.Markdown('**Salary Filter:**'),
    enable_salary_filter,
    salary_min,
    '### Output',
    output_dir_input,
    save_predictions_checkbox,
    save_models_checkbox,
    '---',
    run_button,
    width=350,
    styles={
        'background': '#2b2b2b',
        'padding': '20px',
        'border-radius': '12px',
        'margin': '10px'
    }
)

# Header with logo
logo_path = repo_root / 'src' / 'interface' / 'assets' / 'nba_logo.png'
logo = pn.pane.PNG(str(logo_path), width=60, height=60, align='center')
header_text = pn.pane.Markdown('# delapan.ai', margin=(15, 0, 0, 10))
header = pn.Row(logo, header_text, align='center')

main = pn.Column(
    header,
    status_text,
    dataset_info_pane,
    summary_pane,
    '## Execution Logs',
    log_display,
    width=1000,
    styles={
        'padding': '20px',
        'border-radius': '12px',
        'margin': '10px'
    }
)

# Floating terminal panel HTML - injected via raw HTML pane
floating_terminal_html = pn.pane.HTML(
    '''
    <div id="floating-terminal-container" style="display: none;"></div>
    '''
)

# Serve with floating terminal
pn.Row(sidebar, main, floating_terminal_html).servable()
