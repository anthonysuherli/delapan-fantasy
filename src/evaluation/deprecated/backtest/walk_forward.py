"""
Walk-forward backtesting framework for time series model validation.

This module implements walk-forward validation for NBA fantasy predictions,
supporting both per-player and slate-level models with periodic recalibration.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from src.data.loaders.historical_loader import HistoricalDataLoader
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.evaluation.metrics.accuracy import (
    MAPEMetric, RMSEMetric, MAEMetric, CorrelationMetric,
    CappedMAPEMetric, SMAPEMetric, WMAPEMetric
)
from src.evaluation.benchmarks.season_average import SeasonAverageBenchmark
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.features.pipeline import FeaturePipeline
from src.utils.feature_config import load_feature_config
from src.evaluation.report_generator import BacktestReportGenerator
from src.evaluation.pdf_report_generator import PDFStyleBacktestReportGenerator
from src.filters.base import PlayerFilter
from src.config.paths import (
    PER_PLAYER_MODEL_DIR,
    PER_SLATE_MODEL_DIR,
    PER_PLAYER_TRAINING_INPUTS_DIR,
    PER_SLATE_TRAINING_INPUTS_DIR
)

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtesting framework for NBA fantasy predictions.

    This class implements a comprehensive walk-forward validation strategy with:
    - Periodic model recalibration
    - Per-player or slate-level model support
    - Benchmark comparison
    - Comprehensive metrics evaluation
    - Model and prediction persistence
    - GPU acceleration support
    - Interactive mode for analysis
    """

    def __init__(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        model_type: str = 'xgboost',
        model_params: Optional[Dict] = None,
        feature_config: str = 'default_features',
        output_dir: str = 'data/backtest_results',
        data_dir: str = 'data',
        per_player_models: bool = False,
        min_player_games: int = 10,
        min_games_for_benchmark: int = 5,
        recalibrate_days: int = 7,
        num_seasons: int = 1,
        salary_tiers: Optional[List[int]] = None,
        save_models: bool = True,
        save_predictions: bool = True,
        n_jobs: int = 1,
        rewrite_models: bool = False,
        resume_from_run: Optional[str] = None,
        minutes_threshold: int = 12,
        cmape_cap: float = 8.0,
        wmape_weight: str = 'actual_fpts',
        player_filters: Optional[List[PlayerFilter]] = None,
        gpu_pipeline: Optional[Any] = None,
        enable_feature_caching: bool = True,
        gpu_batch_size: int = 8,
        benchmark_use_all_history: bool = True,
        interactive: bool = False
    ):
        """
        Initialize walk-forward backtest.

        Args:
            train_start: Training start date (YYYYMMDD)
            train_end: Training end date (YYYYMMDD)
            test_start: Test start date (YYYYMMDD)
            test_end: Test end date (YYYYMMDD)
            model_type: Type of model ('xgboost' or 'random_forest')
            model_params: Model hyperparameters
            feature_config: Feature configuration name
            output_dir: Output directory for results
            data_dir: Base data directory
            per_player_models: Whether to train per-player models
            min_player_games: Minimum games for per-player model
            min_games_for_benchmark: Minimum games for benchmark
            recalibrate_days: Days between model recalibrations
            num_seasons: Number of seasons for training data
            salary_tiers: Salary tier breakpoints for analysis
            save_models: Whether to save trained models
            save_predictions: Whether to save predictions
            n_jobs: Number of parallel jobs (-1 for all cores)
            rewrite_models: Whether to rewrite existing models
            resume_from_run: Resume from previous run timestamp
            minutes_threshold: Minimum minutes for evaluation
            cmape_cap: Cap for capped MAPE metric
            wmape_weight: Weight type for weighted MAPE
            player_filters: List of player filters to apply
            gpu_pipeline: GPU pipeline for acceleration
            enable_feature_caching: Whether to cache features
            gpu_batch_size: Batch size for GPU training
            benchmark_use_all_history: Use all history for benchmark
            interactive: Enable interactive mode
        """
        # Convert string dates to integers for comparison
        train_start, train_end, test_start, test_end = (
            int(train_start), int(train_end), int(test_start), int(test_end)
        )

        # Validate date ranges
        if train_start >= train_end:
            raise ValueError(f"train_start ({train_start}) must be < train_end ({train_end})")
        if test_start >= test_end:
            raise ValueError(f"test_start ({test_start}) must be < test_end ({test_end})")
        if train_end > test_start:
            logger.warning(
                f"train_end ({train_end}) is after test_start ({test_start}). "
                "Training data will overlap with test window."
            )

        # Store parameters
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.model_type = model_type
        self.model_params = model_params or {}
        self.feature_config_name = feature_config
        self.data_dir = data_dir
        self.per_player_models = per_player_models
        self.min_player_games = min_player_games
        self.min_games_for_benchmark = min_games_for_benchmark
        self.recalibrate_days = recalibrate_days
        self.num_seasons = num_seasons
        self.salary_tiers = salary_tiers or [0, 4000, 6000, 8000, 15000]
        self.save_models = save_models
        self.save_predictions = save_predictions
        self.n_jobs = n_jobs
        self.rewrite_models = rewrite_models
        self.resume_from_run = resume_from_run
        self.minutes_threshold = int(minutes_threshold)
        self.cmape_cap = float(cmape_cap)
        self.wmape_weight = str(wmape_weight)
        self.player_filters = player_filters or []
        self.gpu_pipeline = gpu_pipeline
        self.enable_feature_caching = enable_feature_caching
        self.gpu_batch_size = gpu_batch_size
        self.benchmark_use_all_history = benchmark_use_all_history
        self.interactive = bool(interactive)
        self._interactive_auto_continue = False

        # Feature and data caching
        self.feature_cache = {}
        self.training_data_cache = None
        self.training_features_cache = None
        self.filtered_player_ids = None
        self.cache_stats = {
            'feature_cache_hits': 0,
            'feature_cache_misses': 0,
            'training_data_loads': 0
        }

        # GPU batch trainer for optimized per-player model training
        self.gpu_batch_trainer = None
        if per_player_models and gpu_batch_size > 1:
            try:
                from src.models.gpu_batch_trainer import GPUBatchTrainer
                gpu_device = model_params.get('device', 'cuda:0') if model_params else 'cuda:0'
                self.gpu_batch_trainer = GPUBatchTrainer(
                    batch_size=gpu_batch_size,
                    gpu_device=gpu_device,
                    max_workers=min(gpu_batch_size, 4)
                )
                logger.info(f"Initialized GPU batch trainer: batch_size={gpu_batch_size}, device={gpu_device}")
            except ImportError:
                logger.warning("GPU batch trainer not available, falling back to sequential training")
                self.gpu_batch_trainer = None

        # Handle relative paths for data_dir and output_dir
        data_path = Path(data_dir)
        output_path_obj = Path(output_dir)

        if not output_path_obj.is_absolute():
            self.output_dir = str(data_path / output_dir)
        else:
            self.output_dir = output_dir

        # Initialize data loader and feature pipeline
        self.loader = HistoricalDataLoader(data_dir=data_dir)
        config = load_feature_config(feature_config)
        self.feature_pipeline = config.build_pipeline(FeaturePipeline)

        # Initialize results containers
        self.results = []
        self.all_predictions = []

        # Initialize metrics
        self.mape_metric = MAPEMetric()
        self.rmse_metric = RMSEMetric()
        self.mae_metric = MAEMetric()
        self.corr_metric = CorrelationMetric()
        self.cmape_metric = CappedMAPEMetric(cap=self.cmape_cap)
        self.smape_metric = SMAPEMetric()
        self.wmape_metric = WMAPEMetric()

        # Initialize benchmark
        self.benchmark = None

        # Model state
        self.current_model = None
        self.player_models = {}
        self.last_training_date = None

        # Configuration dictionary for persistence
        self.config = {
            'data_dir': data_dir,
            'output_dir': self.output_dir,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'model_type': model_type,
            'model_params': model_params or {},
            'feature_config': feature_config,
            'per_player_models': per_player_models,
            'min_player_games': min_player_games,
            'min_games_for_benchmark': min_games_for_benchmark,
            'recalibrate_days': recalibrate_days,
            'num_seasons': num_seasons,
            'salary_tiers': salary_tiers or [0, 4000, 6000, 8000, 15000],
            'n_jobs': n_jobs,
            'save_models': save_models,
            'save_predictions': save_predictions,
            'rewrite_models': rewrite_models,
            'interactive': bool(interactive)
        }

        # Add player filters descriptions
        if player_filters:
            self.config['player_filters'] = [str(pf) for pf in player_filters]

        # Persist evaluation knobs
        self.config['minutes_threshold'] = self.minutes_threshold
        self.config['cmape_cap'] = self.cmape_cap
        self.config['wmape_weight'] = self.wmape_weight

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Log initialization
        self._log_initialization()

    def _log_initialization(self):
        """Log backtest initialization parameters."""
        logger.info("Initialized WalkForwardBacktest")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Training period: {self.train_start} to {self.train_end}")
        logger.info(f"Testing period: {self.test_start} to {self.test_end}")
        logger.info(f"Benchmark: {'Expanding from training start date' if self.benchmark_use_all_history else 'Uses full training period'}")
        logger.info(f"Per-player models: {self.per_player_models}")
        logger.info(f"Feature config: {self.feature_config_name}")
        logger.info(f"Recalibrate every {self.recalibrate_days} days")
        logger.info(f"Rewrite models: {self.rewrite_models}")
        logger.info(f"Save models: {self.save_models}")
        logger.info(f"Save predictions: {self.save_predictions}")
        logger.info(f"Parallel jobs: {self.n_jobs} ({'all cores' if self.n_jobs == -1 else 'sequential' if self.n_jobs == 1 else f'{self.n_jobs} workers'})")
        logger.info("Using YAML-configured feature pipeline")
        logger.info(f"Metrics: minutes_threshold={self.minutes_threshold}, cmape_cap={self.cmape_cap}, wmape_weight={self.wmape_weight}")

        if self.player_filters:
            logger.info(f"Player filters: {len(self.player_filters)} filters")
            for pf in self.player_filters:
                logger.info(f"  - {pf}")

    def run(self) -> Dict[str, Any]:
        """
        Execute walk-forward backtest.

        Returns:
            Dictionary containing aggregated results and metrics
        """
        # Import trainers here to avoid circular imports
        from .trainers.per_player import PerPlayerTrainer
        from .trainers.per_slate import PerSlateTrainer

        # Select appropriate trainer
        if self.per_player_models:
            trainer = PerPlayerTrainer(self)
        else:
            trainer = PerSlateTrainer(self)

        # Execute backtest
        return trainer.run()

    def _should_recalibrate(self, current_date: str) -> bool:
        """
        Check if model should be recalibrated.

        Args:
            current_date: Current test date (YYYYMMDD)

        Returns:
            Whether to recalibrate models
        """
        if self.last_training_date is None:
            return True

        from datetime import datetime
        current = datetime.strptime(str(current_date), '%Y%m%d')
        last = datetime.strptime(str(self.last_training_date), '%Y%m%d')
        days_diff = (current - last).days

        return days_diff >= self.recalibrate_days

    def _format_time(self, seconds: float) -> str:
        """
        Format seconds into human-readable time string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _build_training_features(
        self,
        training_data: pd.DataFrame,
        injuries: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training features using FeaturePipeline.

        Args:
            training_data: Training data
            injuries: Optional injury data

        Returns:
            Tuple of (features, targets)
        """
        if training_data.empty:
            return pd.DataFrame(), pd.Series()

        df = training_data.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['playerID', 'gameDate'])

        if 'fpts' not in df.columns:
            df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

        # Always pass injuries in context to ensure consistent feature columns
        injuries_to_use = injuries if injuries is not None else pd.DataFrame()
        context = {'injuries': injuries_to_use}

        df = self.feature_pipeline.fit_transform(df, context=context)
        df = df.dropna(subset=['target'])

        # Select only numeric columns (int64, float64) to avoid dtype issues with XGBoost
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Remove target from features (it's separate)
        feature_cols = [col for col in numeric_cols if col != 'target']

        X = df[feature_cols].copy()
        X = X.fillna(0).infer_objects(copy=False)

        y = df['target']

        return X, y

    def _build_training_features_cached(
        self,
        training_data: pd.DataFrame,
        injuries: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training features with caching to avoid redundant computation.

        Args:
            training_data: Training data
            injuries: Optional injury data

        Returns:
            Tuple of (features, targets)
        """
        if not self.enable_feature_caching:
            return self._build_training_features(training_data, injuries)

        # Create cache key based on training data characteristics
        data_hash = hash((
            len(training_data),
            training_data['gameDate'].min() if 'gameDate' in training_data.columns else '',
            training_data['gameDate'].max() if 'gameDate' in training_data.columns else '',
            len(injuries) if injuries is not None and not injuries.empty else 0,
            self.feature_config_name
        ))

        cache_key = f"training_features_{data_hash}"

        # Check cache
        if cache_key in self.feature_cache:
            logger.debug(f"Feature cache hit: {cache_key}")
            self.cache_stats['feature_cache_hits'] += 1
            return self.feature_cache[cache_key]

        # Cache miss - compute features
        logger.debug(f"Feature cache miss: {cache_key} - computing features")
        self.cache_stats['feature_cache_misses'] += 1

        X, y = self._build_training_features(training_data, injuries)

        # Cache the result
        self.feature_cache[cache_key] = (X.copy(), y.copy())
        logger.info(f"Cached features for {len(X)} samples (cache size: {len(self.feature_cache)})")

        return X, y

    def _load_training_data_cached(self, player_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load training data once and cache for reuse across slates.

        Args:
            player_ids: Optional list of player IDs to filter for

        Returns:
            Cached training data
        """
        if self.training_data_cache is None:
            logger.info("Loading training data (one-time operation)")
            start_time = time.perf_counter()

            if player_ids:
                self.training_data_cache = self.loader.load_historical_player_logs(
                    end_date=self.train_end,
                    num_seasons=self.num_seasons,
                    player_ids=player_ids
                )
            else:
                self.training_data_cache = self.loader.load_historical_player_logs(
                    end_date=self.train_end,
                    num_seasons=self.num_seasons
                )

            elapsed = time.perf_counter() - start_time
            self.cache_stats['training_data_loads'] += 1
            logger.info(f"Loaded {len(self.training_data_cache)} training samples in {self._format_time(elapsed)}")

        return self.training_data_cache

    def _build_slate_features(
        self,
        slate_data: Dict[str, Any],
        player_training_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build features for slate prediction.

        Args:
            slate_data: Slate data dictionary
            player_training_data: Historical player data

        Returns:
            DataFrame with slate features
        """
        if slate_data.get('dfs_salaries', pd.DataFrame()).empty:
            return pd.DataFrame()

        salaries_df = slate_data['dfs_salaries'].copy()

        if player_training_data.empty:
            # No historical data, return empty
            return pd.DataFrame()

        # Combine historical and slate data
        df = player_training_data.copy()

        if 'gameDate' not in df.columns:
            return pd.DataFrame()

        df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['playerID', 'gameDate'])

        if 'fpts' not in df.columns:
            df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

        # Always pass injuries in context
        injuries = slate_data.get('injuries', pd.DataFrame())
        context = {'injuries': injuries}

        # Fit pipeline on historical data and transform
        df = self.feature_pipeline.fit_transform(df, context=context)

        # Get the latest features for each player
        latest_features = df.groupby('playerID').last().reset_index()

        # Merge with salary data to get current slate info
        slate_features = salaries_df.merge(
            latest_features,
            on='playerID',
            how='left',
            suffixes=('', '_hist')
        )

        # Keep slate metadata from salaries
        for col in ['playerName', 'team', 'pos', 'salary']:
            if col in salaries_df.columns and f'{col}_hist' in slate_features.columns:
                slate_features[col] = salaries_df[col]
                slate_features = slate_features.drop(columns=[f'{col}_hist'])

        return slate_features

    def _generate_projections(self, model, slate_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate projections using a trained model.

        Args:
            model: Trained model
            slate_features: Features for prediction

        Returns:
            DataFrame with projections
        """
        logger.debug(f"Generating projections for {len(slate_features)} players")

        # Base metadata columns (only those that exist)
        potential_metadata = ['playerID', 'playerName', 'longName', 'team', 'pos', 'salary']
        metadata_cols = [col for col in potential_metadata if col in slate_features.columns]

        # Add injury-related columns if they exist
        injury_cols = ['status', 'is_out', 'is_doubtful', 'is_questionable', 'is_injured',
                        'injury_status', 'injury_designation', 'injury_description', 'gameInfo']
        for col in injury_cols:
            if col in slate_features.columns and col not in metadata_cols:
                metadata_cols.append(col)

        # Select only numeric columns for model input (avoid dtype issues with XGBoost)
        numeric_cols = slate_features.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Exclude metadata and target columns
        exclude_cols = metadata_cols + ['target']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        X = slate_features[feature_cols].fillna(0).infer_objects(copy=False)

        if hasattr(model, 'predict'):
            if hasattr(model, 'is_trained') and not model.is_trained:
                raise ValueError("Model must be trained before prediction")
            predictions = model.predict(X)
        else:
            predictions = model.predict(X)

        projections = slate_features[metadata_cols].copy()
        projections['projected_fpts'] = predictions

        # Add value metric if salary is available
        if 'salary' in projections.columns:
            projections['salary'] = pd.to_numeric(projections['salary'], errors='coerce')
            projections['value'] = projections['projected_fpts'] / (projections['salary'] / 1000)
        else:
            projections['value'] = 0.0

        logger.debug(f"Projections: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")

        return projections

    def _evaluate_slate(
        self,
        projections: pd.DataFrame,
        test_date: str,
        model_name: str = 'Model'
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Evaluate slate projections against actuals.

        Args:
            projections: Projections DataFrame
            test_date: Test date
            model_name: Name of the model for logging

        Returns:
            Tuple of (results dict, merged DataFrame)
        """
        logger.debug(f"Evaluating slate {test_date} for {model_name}")
        logger.info(f"_evaluate_slate called with {model_name}: {len(projections)} rows, columns: {list(projections.columns)}")

        # Validate projections DataFrame
        if projections.empty:
            logger.warning(f"No projections available for {test_date}")
            return {
                'date': test_date,
                'num_players': 0,
                'model_mape': np.nan,
                'model_rmse': np.nan,
                'model_mae': np.nan,
                'model_corr': np.nan
            }, pd.DataFrame()

        # Determine which prediction column to use
        pred_col = None
        if 'projected_fpts' in projections.columns:
            pred_col = 'projected_fpts'
        elif 'benchmark_pred' in projections.columns:
            pred_col = 'benchmark_pred'
        else:
            logger.error(f"Projections missing prediction column for {test_date}. Available columns: {list(projections.columns)}")
            return {
                'date': test_date,
                'num_players': 0,
                'model_mape': np.nan,
                'model_rmse': np.nan,
                'model_mae': np.nan,
                'model_corr': np.nan
            }, pd.DataFrame()

        # Load actuals
        actuals = self._load_actuals(test_date)

        if actuals.empty:
            logger.warning(f"No actuals found for {test_date}")
            return {
                'date': test_date,
                'num_players': 0,
                'model_mape': np.nan,
                'model_rmse': np.nan,
                'model_mae': np.nan,
                'model_corr': np.nan
            }, pd.DataFrame()

        # Merge projections with actuals
        merged = projections.merge(
            actuals[['playerID', 'actual_fpts']],
            on='playerID',
            how='inner'
        )

        if 'actual_mins' in actuals.columns:
            merged = merged.merge(
                actuals[['playerID', 'actual_mins']],
                on='playerID',
                how='left'
            )

        logger.debug(f"Matched {len(merged)}/{len(projections)} players with actuals")

        if merged.empty:
            logger.warning(f"No matching players for {test_date}")
            return {
                'date': test_date,
                'num_players': 0,
                'model_mape': np.nan,
                'model_rmse': np.nan,
                'model_mae': np.nan,
                'model_corr': np.nan
            }, pd.DataFrame()

        # Verify required columns exist
        if pred_col not in merged.columns:
            logger.error(f"Missing '{pred_col}' column in merged data for {test_date}. Available columns: {list(merged.columns)}")
            return {
                'date': test_date,
                'num_players': 0,
                'model_mape': np.nan,
                'model_rmse': np.nan,
                'model_mae': np.nan,
                'model_corr': np.nan
            }, pd.DataFrame()

        if 'actual_fpts' not in merged.columns:
            logger.error(f"Missing 'actual_fpts' column in merged data for {test_date}. Available columns: {list(merged.columns)}")
            return {
                'date': test_date,
                'num_players': 0,
                'model_mape': np.nan,
                'model_rmse': np.nan,
                'model_mae': np.nan,
                'model_corr': np.nan
            }, pd.DataFrame()

        y_true = merged['actual_fpts'].values
        y_pred = merged[pred_col].values

        # Calculate metrics
        # Filter out near-zero actuals for MAPE
        mape_filter = merged['actual_fpts'] >= 5.0
        if mape_filter.any():
            model_mape = self.mape_metric.calculate(
                merged.loc[mape_filter, 'actual_fpts'].values,
                merged.loc[mape_filter, pred_col].values
            )
        else:
            model_mape = np.nan

        model_rmse = self.rmse_metric.calculate(y_true, y_pred)
        model_mae = self.mae_metric.calculate(y_true, y_pred)
        model_corr = self.corr_metric.calculate(y_true, y_pred)

        result = {
            'date': test_date,
            'num_players': len(merged),
            'model_mape': model_mape,
            'model_rmse': model_rmse,
            'model_mae': model_mae,
            'model_corr': model_corr,
            'mean_projected': y_pred.mean(),
            'mean_actual': y_true.mean()
        }

        logger.debug(f"Evaluation complete: MAPE={model_mape:.2f}%, RMSE={model_rmse:.2f}, MAE={model_mae:.2f}, Corr={model_corr:.3f}")

        merged['date'] = test_date
        return result, merged

    def _load_actuals(self, date: str) -> pd.DataFrame:
        """
        Load actual results for a date.

        Args:
            date: Date to load actuals for (YYYYMMDD format)

        Returns:
            DataFrame with actual results
        """
        try:
            # Convert date format YYYYMMDD to YYYY/MM/DD for file path
            year = date[:4]
            month = date[4:6]
            day = date[6:8]

            # Try to load from player_logs_extracted (subdirectory of data_dir)
            if hasattr(self, 'data_dir') and self.data_dir:
                base_dir = Path(self.data_dir)
                if not base_dir.is_absolute():
                    # Convert relative path to absolute
                    base_dir = base_dir.resolve()
            else:
                # Fall back to current working directory
                base_dir = Path.cwd()

            file_path = base_dir / 'player_logs_extracted' / year / month / f"{day}.parquet"

            if not file_path.exists():
                logger.warning(f"No actuals file found for {date} at {file_path}")
                return pd.DataFrame()

            df = pd.read_parquet(file_path)

            if df.empty:
                return pd.DataFrame()

            df['actual_fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

            if 'longName' in df.columns and 'playerName' not in df.columns:
                df['playerName'] = df['longName']

            # Include actual minutes if available
            if 'mins' in df.columns:
                df['actual_mins'] = pd.to_numeric(df['mins'], errors='coerce')
            else:
                df['actual_mins'] = np.nan

            return df[['playerID', 'playerName', 'team', 'pos', 'actual_fpts', 'actual_mins']]

        except Exception as e:
            logger.error(f"Failed to load actuals for {date}: {str(e)}")
            return pd.DataFrame()

    def _save_model(self, model, model_file: Path, player_name: str, player_id: str, num_samples: int):
        """
        Save a player model with metadata.

        Args:
            model: Trained model
            model_file: Path to save model
            player_name: Player name
            player_id: Player ID
            num_samples: Number of training samples
        """
        try:
            metadata = {
                'player_name': player_name,
                'player_id': player_id,
                'model_type': self.model_type,
                'num_training_samples': num_samples,
                'feature_config': self.feature_config_name,
                'model_params': self.model_params,
                'trained_at': pd.Timestamp.now().isoformat()
            }

            if hasattr(model, 'save'):
                model.save(str(model_file))
            else:
                import pickle
                with open(model_file, 'wb') as f:
                    pickle.dump({'model': model, 'metadata': metadata}, f)

            # Save metadata separately
            metadata_file = model_file.with_suffix('.meta.json')
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save model for {player_name}: {e}")

    def _interactive_pause(self, test_date: str, daily_results: Dict[str, Any]) -> str:
        """
        Interactive pause for analysis.

        Args:
            test_date: Current test date
            daily_results: Results for the day

        Returns:
            User action ('continue', 'stop', or 'auto')
        """
        if self._interactive_auto_continue:
            return 'continue'

        print("\n" + "="*60)
        print(f"Completed slate: {test_date}")
        print(f"Players: {daily_results.get('num_players', 0)}")
        print(f"MAPE: {daily_results.get('model_mape', np.nan):.2f}%")
        print(f"RMSE: {daily_results.get('model_rmse', np.nan):.2f}")
        print(f"Correlation: {daily_results.get('model_corr', np.nan):.3f}")
        print("="*60)

        while True:
            response = input("\nContinue? [Y]es / [N]o / [A]uto-continue: ").strip().lower()
            if response in ['y', 'yes', '']:
                return 'continue'
            elif response in ['n', 'no']:
                return 'stop'
            elif response in ['a', 'auto']:
                self._interactive_auto_continue = True
                return 'continue'
            else:
                print("Invalid response. Please enter Y, N, or A.")

    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate daily results into final metrics.

        Returns:
            Aggregated results dictionary
        """
        if not self.results:
            return {
                'error': 'No results to aggregate',
                'config': self.config
            }

        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(self.results)

        # Filter out invalid results
        valid_results = results_df[results_df['num_players'] > 0].copy()

        if valid_results.empty:
            return {
                'error': 'No valid results',
                'config': self.config
            }

        # Calculate aggregate metrics
        aggregated = {
            'num_slates': len(valid_results),
            'total_players': valid_results['num_players'].sum(),
            'avg_players_per_slate': valid_results['num_players'].mean(),
            'overall_mape': valid_results['model_mape'].mean(),
            'overall_rmse': valid_results['model_rmse'].mean(),
            'overall_mae': valid_results['model_mae'].mean(),
            'overall_corr': valid_results['model_corr'].mean(),
            'daily_results': self.results,
            'config': self.config
        }

        # Add benchmark comparison if available
        if 'benchmark' in valid_results.columns:
            benchmark_results = []
            for _, row in valid_results.iterrows():
                if isinstance(row.get('benchmark'), dict):
                    benchmark_results.append(row['benchmark'])

            if benchmark_results:
                benchmark_df = pd.DataFrame(benchmark_results)
                aggregated['benchmark_mape'] = benchmark_df['model_mape'].mean()
                aggregated['benchmark_rmse'] = benchmark_df['model_rmse'].mean()
                aggregated['benchmark_corr'] = benchmark_df['model_corr'].mean()

        return aggregated