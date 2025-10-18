"""Streamlit interface for configuring and running walk-forward backtests."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml

from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.walk_forward_backtest import WalkForwardBacktest


LOGGER = logging.getLogger(__name__)
EXPERIMENT_DIR = Path("config/experiments")


@dataclass
class BacktestConfig:
    """Serializable configuration used to run the walk-forward backtest."""

    db_path: str = "nba_dfs.db"
    train_start: str = "20231101"
    train_end: str = "20240229"
    test_start: str = "20240301"
    test_end: str = "20240331"
    model_type: str = "xgboost"
    model_params: Dict[str, Any] = field(default_factory=dict)
    feature_config: str = "default_features"
    output_dir: str = "data/backtest_results"
    data_dir: Optional[str] = None
    per_player_models: bool = False
    recalibrate_days: int = 7
    minutes_threshold: int = 12
    n_jobs: int = 1
    save_models: bool = True
    save_predictions: bool = True
    resume_from_run: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["model_params"] = data["model_params"] or {}
        return data


DEFAULT_CONFIG = BacktestConfig().to_dict()


class QueueLogHandler(logging.Handler):
    """Logging handler that forwards log records to a queue."""

    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive fallback
            message = record.getMessage()
        self.queue.put(
            {
                "type": "log",
                "level": record.levelname,
                "message": message,
                "created": record.created,
            }
        )


class QueueStream(io.TextIOBase):
    """File-like object that writes standard output/error to a queue."""

    def __init__(self, queue: Queue, level: str = "INFO") -> None:
        super().__init__()
        self.queue = queue
        self.level = level

    def write(self, data: str) -> int:  # pragma: no cover - thin wrapper
        if data and not data.isspace():
            self.queue.put(
                {
                    "type": "log",
                    "level": self.level,
                    "message": data.rstrip(),
                    "created": time.time(),
                }
            )
        return len(data)

    def flush(self) -> None:  # pragma: no cover - compatibility
        return None


class StreamingList(list):
    """List that emits an event whenever an item is appended."""

    def __init__(self, queue: Queue, event_type: str) -> None:
        super().__init__()
        self._queue = queue
        self._event_type = event_type

    def append(self, item: Any) -> None:  # pragma: no cover - delegates to list
        super().append(item)
        self._queue.put({"type": self._event_type, "payload": item})


class BacktestRunner:
    """Background worker that executes the backtest and streams updates."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_queue: "Queue[Dict[str, Any]]" = Queue()
        self.result_queue: "Queue[Dict[str, Any]]" = Queue()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    def start(self) -> None:
        if self.is_running():
            raise RuntimeError("Backtest is already running")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.addHandler(handler)
        if previous_level == logging.NOTSET or previous_level > logging.INFO:
            root_logger.setLevel(logging.INFO)

        stdout_stream = QueueStream(self.log_queue, level="INFO")
        stderr_stream = QueueStream(self.log_queue, level="ERROR")

        self.log_queue.put(
            {
                "type": "log",
                "level": "INFO",
                "message": "Starting backtest run...",
                "created": time.time(),
            }
        )

        try:
            backtest = WalkForwardBacktest(**self.config)
            backtest.results = StreamingList(self.result_queue, "daily_result")

            with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(
                stderr_stream
            ):
                results = backtest.run()

            self.result_queue.put({"type": "final_summary", "payload": results})
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.exception("Backtest execution failed")
            self._error = str(exc)
            self.log_queue.put(
                {
                    "type": "log",
                    "level": "ERROR",
                    "message": f"Backtest failed: {exc}",
                    "created": time.time(),
                }
            )
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(previous_level)
            self.result_queue.put(
                {
                    "type": "status",
                    "status": "finished",
                    "error": self._error,
                }
            )

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def consume_logs(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        while True:
            try:
                messages.append(self.log_queue.get_nowait())
            except Empty:
                break
        return messages

    def consume_results(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        while True:
            try:
                events.append(self.result_queue.get_nowait())
            except Empty:
                break
        return events

    @property
    def error(self) -> Optional[str]:
        return self._error


def _list_experiments() -> List[Path]:
    if not EXPERIMENT_DIR.exists():
        return []
    return sorted(p for p in EXPERIMENT_DIR.glob("*.yaml") if p.is_file())


def _load_experiment_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    config = DEFAULT_CONFIG.copy()
    data_section = data.get("data", {})
    for key in ("train_start", "train_end", "test_start", "test_end"):
        if key in data_section:
            config[key] = str(data_section[key])

    model_section = data.get("model", {})
    if model_section:
        if "type" in model_section:
            config["model_type"] = model_section["type"]
        if "params" in model_section and isinstance(model_section["params"], dict):
            config["model_params"] = model_section["params"]

    evaluation_section = data.get("evaluation", {})
    output_dir = evaluation_section.get("output_dir")
    if output_dir:
        config["output_dir"] = output_dir

    return config


def _initialise_session_state() -> None:
    if "backtest_config" not in st.session_state:
        st.session_state.backtest_config = DEFAULT_CONFIG.copy()
    if "model_params_text" not in st.session_state:
        st.session_state.model_params_text = json.dumps(
            st.session_state.backtest_config.get("model_params", {}), indent=2
        )
    if "logs" not in st.session_state:
        st.session_state.logs: List[Dict[str, Any]] = []
    if "daily_results" not in st.session_state:
        st.session_state.daily_results: List[Dict[str, Any]] = []
    if "final_summary" not in st.session_state:
        st.session_state.final_summary: Optional[Dict[str, Any]] = None
    if "runner" not in st.session_state:
        st.session_state.runner: Optional[BacktestRunner] = None
    if "run_error" not in st.session_state:
        st.session_state.run_error: Optional[str] = None
    if "selected_experiment" not in st.session_state:
        st.session_state.selected_experiment = "Manual configuration"
    if "training_sample" not in st.session_state:
        st.session_state.training_sample: Optional[pd.DataFrame] = None
    if "training_sample_error" not in st.session_state:
        st.session_state.training_sample_error: Optional[str] = None
    if "training_sample_config" not in st.session_state:
        st.session_state.training_sample_config: Optional[Dict[str, Any]] = None
    if "training_sample_limit" not in st.session_state:
        st.session_state.training_sample_limit = 25


def _reset_training_sample_state() -> None:
    st.session_state.training_sample = None
    st.session_state.training_sample_error = None
    st.session_state.training_sample_config = None


def _format_timestamp(value: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(value))


def _render_sidebar() -> None:
    st.sidebar.header("Backtest Controls")

    experiments = _list_experiments()
    experiment_labels = ["Manual configuration"] + [p.name for p in experiments]
    try:
        default_index = experiment_labels.index(st.session_state.selected_experiment)
    except ValueError:
        default_index = 0

    selected_label = st.sidebar.selectbox(
        "Experiment preset", experiment_labels, index=default_index
    )

    if selected_label != st.session_state.selected_experiment:
        if selected_label != "Manual configuration":
            selected_path = experiments[experiment_labels.index(selected_label) - 1]
            loaded = _load_experiment_config(selected_path)
            st.session_state.backtest_config.update(loaded)
            st.session_state.model_params_text = json.dumps(
                st.session_state.backtest_config.get("model_params", {}), indent=2
            )
            st.sidebar.success(f"Loaded preset: {selected_label}")
            _reset_training_sample_state()
        st.session_state.selected_experiment = selected_label

    config = st.session_state.backtest_config

    with st.sidebar.form("backtest_config_form"):
        db_path = st.text_input("Database path", value=config["db_path"])
        data_dir = st.text_input(
            "Data directory (optional)",
            value="" if config.get("data_dir") is None else str(config.get("data_dir")),
        )
        train_start = st.text_input("Train start", value=config["train_start"])
        train_end = st.text_input("Train end", value=config["train_end"])
        test_start = st.text_input("Test start", value=config["test_start"])
        test_end = st.text_input("Test end", value=config["test_end"])

        model_type = st.selectbox(
            "Model type",
            options=["xgboost", "random_forest"],
            index=["xgboost", "random_forest"].index(config.get("model_type", "xgboost")),
        )
        feature_config = st.text_input(
            "Feature config name", value=config.get("feature_config", "default_features")
        )
        output_dir = st.text_input(
            "Output directory", value=config.get("output_dir", DEFAULT_CONFIG["output_dir"])
        )
        model_params_text = st.text_area(
            "Model parameters (JSON)",
            value=st.session_state.model_params_text,
            height=160,
        )

        per_player_models = st.checkbox(
            "Train per-player models",
            value=bool(config.get("per_player_models", False)),
        )
        recalibrate_days = st.number_input(
            "Recalibrate cadence (days)",
            min_value=1,
            value=int(config.get("recalibrate_days", 7)),
            step=1,
        )
        minutes_threshold = st.number_input(
            "Minutes threshold",
            min_value=0,
            value=int(config.get("minutes_threshold", 12)),
            step=1,
        )
        n_jobs = st.number_input(
            "Parallel jobs",
            min_value=1,
            value=int(config.get("n_jobs", 1)),
            step=1,
        )
        save_predictions = st.checkbox(
            "Persist predictions",
            value=bool(config.get("save_predictions", True)),
        )
        save_models = st.checkbox(
            "Persist models",
            value=bool(config.get("save_models", True)),
        )
        resume_from_run = st.text_input(
            "Resume from run (optional)",
            value=config.get("resume_from_run") or "",
        )

        submitted = st.form_submit_button("Run Backtest", use_container_width=True)

    if submitted:
        if st.session_state.runner and st.session_state.runner.is_running():
            st.sidebar.warning("A backtest is already running.")
            return

        try:
            model_params = json.loads(model_params_text) if model_params_text.strip() else {}
        except json.JSONDecodeError as exc:
            st.sidebar.error(f"Invalid model parameters JSON: {exc}")
            return

        updated_config = {
            "db_path": db_path.strip() or DEFAULT_CONFIG["db_path"],
            "data_dir": data_dir.strip() or None,
            "train_start": train_start.strip(),
            "train_end": train_end.strip(),
            "test_start": test_start.strip(),
            "test_end": test_end.strip(),
            "model_type": model_type,
            "feature_config": feature_config.strip() or DEFAULT_CONFIG["feature_config"],
            "model_params": model_params,
            "per_player_models": per_player_models,
            "recalibrate_days": int(recalibrate_days),
            "minutes_threshold": int(minutes_threshold),
            "n_jobs": int(n_jobs),
            "save_predictions": bool(save_predictions),
            "save_models": bool(save_models),
            "resume_from_run": resume_from_run.strip() or None,
            "output_dir": output_dir.strip() or DEFAULT_CONFIG["output_dir"],
        }

        st.session_state.backtest_config.update(updated_config)
        st.session_state.model_params_text = json.dumps(model_params, indent=2)
        _reset_training_sample_state()

        runner = BacktestRunner(st.session_state.backtest_config.copy())
        st.session_state.runner = runner
        st.session_state.logs = []
        st.session_state.daily_results = []
        st.session_state.final_summary = None
        st.session_state.run_error = None

        runner.start()
        st.sidebar.success("Backtest started.")


def _render_results_panel() -> None:
    st.subheader("Backtest Results")
    tab_results, tab_training_sample = st.tabs(
        ["Daily Metrics", "Training Input Sample"]
    )

    with tab_results:
        if st.session_state.daily_results:
            df = pd.DataFrame(st.session_state.daily_results)
            df = df.set_index("date") if "date" in df.columns else df
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Results will appear here once available.")

        if st.session_state.final_summary:
            st.markdown("#### Final Summary")
            st.json(st.session_state.final_summary)
        elif st.session_state.runner and st.session_state.runner.is_running():
            st.info("Backtest is running... results will update automatically.")

    with tab_training_sample:
        _render_training_sample_tab()


def _render_training_sample_tab() -> None:
    st.markdown("Preview the feature matrix that powers training.")

    limit = st.slider(
        "Rows to preview",
        min_value=5,
        max_value=200,
        value=int(st.session_state.training_sample_limit),
        step=5,
        key="training_sample_limit_slider",
    )

    if limit != st.session_state.training_sample_limit:
        st.session_state.training_sample_limit = int(limit)
        _reset_training_sample_state()

    if st.button("Load sample", key="load_training_sample_button"):
        with st.spinner("Building training sample..."):
            sample_df, error = _load_training_sample(
                st.session_state.backtest_config,
                limit=int(st.session_state.training_sample_limit),
            )

        if error:
            st.session_state.training_sample = None
            st.session_state.training_sample_error = error
        else:
            st.session_state.training_sample = sample_df
            st.session_state.training_sample_error = None
            st.session_state.training_sample_config = deepcopy(
                st.session_state.backtest_config
            )

    if st.session_state.training_sample_error:
        st.error(st.session_state.training_sample_error)
        return

    if st.session_state.training_sample is None:
        st.info("Click \"Load sample\" to preview the latest training inputs.")
        return

    if st.session_state.training_sample_config != st.session_state.backtest_config:
        st.warning(
            "Configuration has changed since this sample was generated. Reload to"
            " refresh the preview."
        )

    sample_df = st.session_state.training_sample
    if sample_df.empty:
        st.info("No training rows were available for the current configuration.")
        return

    st.dataframe(sample_df, use_container_width=True)
    st.caption(
        "Showing the first "
        f"{min(len(sample_df), st.session_state.training_sample_limit)} rows of"
        " the engineered training dataset."
    )


def _render_logs_panel() -> None:
    st.markdown("---")
    st.subheader("Backtest Logs")
    if not st.session_state.logs:
        st.info("Logs will stream here during execution.")
        return

    formatted = [
        f"[{_format_timestamp(entry['created'])}] {entry['level']}: {entry['message']}"
        for entry in st.session_state.logs
    ]
    st.code("\n".join(formatted), language="text")


def _load_training_sample(
    config: Dict[str, Any], limit: int = 25
) -> Tuple[pd.DataFrame, Optional[str]]:
    preview_config = deepcopy(config)

    try:
        preview_backtest = WalkForwardBacktest(**preview_config)
        training_data = preview_backtest.loader.load_historical_player_logs(
            start_date=preview_backtest.train_start,
            end_date=preview_backtest.train_end,
            num_seasons=preview_backtest.num_seasons,
        )
        if training_data.empty:
            return pd.DataFrame(), "No training data found for the selected window."

        df = training_data.copy()
        df["gameDate"] = pd.to_datetime(
            df["gameDate"], format="%Y%m%d", errors="coerce"
        )

        if "fpts" not in df.columns:
            df["fpts"] = df.apply(calculate_dk_fantasy_points, axis=1)

        df["target"] = df.groupby("playerID")["fpts"].shift(-1)

        features = preview_backtest.feature_pipeline.fit_transform(df)
        features = features.dropna(subset=["target"])

        if features.empty:
            return pd.DataFrame(), "Feature pipeline did not produce any training rows."

        if "gameDate" in features.columns:
            features["gameDate"] = pd.to_datetime(
                features["gameDate"], errors="coerce"
            )
        sort_columns = [
            column for column in ("gameDate", "playerID") if column in features.columns
        ]
        if sort_columns:
            features = features.sort_values(sort_columns)

        sample_df = features.head(limit).copy()
        if "gameDate" in sample_df.columns:
            sample_df["gameDate"] = sample_df["gameDate"].dt.strftime("%Y-%m-%d")

        ordering: List[str] = []
        for col in (
            "playerID",
            "playerName",
            "longName",
            "team",
            "pos",
            "gameDate",
            "target",
        ):
            if col in sample_df.columns:
                ordering.append(col)

        remaining_cols = [
            col for col in sample_df.columns if col not in ordering
        ]
        ordered_df = sample_df[ordering + remaining_cols]

        return ordered_df.reset_index(drop=True), None
    except Exception as exc:  # pragma: no cover - defensive UI helper
        LOGGER.exception("Failed to prepare training sample preview")
        return pd.DataFrame(), f"Unable to load training sample: {exc}"


def _drain_runner_events() -> None:
    runner: Optional[BacktestRunner] = st.session_state.get("runner")
    if not runner:
        return

    for log_event in runner.consume_logs():
        st.session_state.logs.append(log_event)

    for result_event in runner.consume_results():
        event_type = result_event.get("type")
        if event_type == "daily_result":
            payload = result_event.get("payload", {})
            if isinstance(payload, dict):
                st.session_state.daily_results.append(payload)
        elif event_type == "final_summary":
            st.session_state.final_summary = result_event.get("payload")
        elif event_type == "status" and result_event.get("status") == "finished":
            st.session_state.run_error = result_event.get("error")
            st.session_state.runner = None


def main() -> None:
    """Entry point for the Streamlit application."""

    st.set_page_config(page_title="Backtest Control Center", layout="wide")
    _initialise_session_state()

    _drain_runner_events()

    _render_sidebar()

    col_results = st.container()
    with col_results:
        _render_results_panel()

    _render_logs_panel()

    runner = st.session_state.get("runner")
    if runner and runner.is_running():
        time.sleep(1)
        st.experimental_rerun()
    elif st.session_state.run_error:
        st.error(f"Backtest finished with errors: {st.session_state.run_error}")
    elif st.session_state.final_summary:
        st.success("Backtest completed successfully.")


if __name__ == "__main__":
    main()
