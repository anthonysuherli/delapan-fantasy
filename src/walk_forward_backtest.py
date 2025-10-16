import pandas as pd
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from joblib import Parallel, delayed

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.evaluation.metrics.accuracy import MAPEMetric, RMSEMetric, MAEMetric, CorrelationMetric, CappedMAPEMetric, SMAPEMetric, WMAPEMetric
from src.evaluation.benchmarks.season_average import SeasonAverageBenchmark
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.features.pipeline import FeaturePipeline
from src.utils.feature_config import load_feature_config
from src.evaluation.report_generator import BacktestReportGenerator
from src.config.paths import (
    PER_PLAYER_MODEL_DIR,
    PER_SLATE_MODEL_DIR,
    PER_PLAYER_TRAINING_INPUTS_DIR,
    PER_SLATE_TRAINING_INPUTS_DIR
)

logger = logging.getLogger(__name__)


def _train_single_player_model(
    player_row: pd.Series,
    training_data: pd.DataFrame,
    feature_pipeline: FeaturePipeline,
    model_type: str,
    model_params: Dict,
    min_player_games: int,
    save_models: bool,
    models_dir: Path,
    inputs_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Worker function for parallel per-player model training.

    Returns dict with projection data or None if player should be skipped.
    """
    player_id = player_row.get('playerID')
    player_name = player_row.get('longName')

    player_training_data = training_data[training_data['playerID'] == player_id].copy()

    if len(player_training_data) < min_player_games:
        return None

    try:
        df = player_training_data.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['playerID', 'gameDate'])

        if 'fpts' not in df.columns:
            df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

        df['target'] = df.groupby('playerID')['fpts'].shift(-1)
        df = feature_pipeline.fit_transform(df)
        df = df.dropna(subset=['target'])

        metadata_cols = [
            'playerID', 'playerName', 'longName', 'team', 'teamAbv', 'teamID',
            'pos', 'gameDate', 'gameID', 'fpts', 'fantasyPoints', 'fantasyPts',
            'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins',
            'tech', 'created_at', 'updated_at'
        ]

        feature_cols = [col for col in df.columns if col not in metadata_cols]

        X_train = df[feature_cols].copy()
        X_train = X_train.fillna(0)

        # Convert any remaining object columns to numeric
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)

        y_train = df['target']

        if X_train.empty or y_train.empty or len(X_train) < 3:
            return None

        safe_player_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in player_name)
        safe_player_name = safe_player_name.replace(' ', '_')
        input_file = inputs_dir / f"player_{safe_player_name}_{player_id}_inputs.parquet"

        if model_type == 'xgboost':
            try:
                model = XGBoostModel(model_params)
                model.train(X_train, y_train, save_inputs=True, input_save_path=str(input_file))
            except ImportError:
                # Fall back to RandomForest if XGBoost not available
                rf_params = {k: v for k, v in model_params.items()
                             if k not in ['learning_rate', 'objective', 'colsample_bytree', 'subsample', 'min_child_weight']}
                model = RandomForestModel(rf_params)
                model.train(X_train, y_train, save_inputs=True, input_save_path=str(input_file))
        elif model_type == 'random_forest':
            # Filter out XGBoost-specific parameters
            rf_params = {k: v for k, v in model_params.items()
                         if k not in ['learning_rate', 'objective', 'colsample_bytree', 'subsample', 'min_child_weight']}
            model = RandomForestModel(rf_params)
            model.train(X_train, y_train, save_inputs=True, input_save_path=str(input_file))
        else:
            return None

        latest_features = X_train.iloc[[-1]]
        prediction = model.predict(latest_features)[0]

        result = {
            'playerID': player_id,
            'playerName': player_name,
            'team': player_row.get('team', ''),
            'pos': player_row.get('pos', ''),
            'salary': player_row.get('salary', 0),
            'projected_fpts': prediction,
            'model': model if save_models else None
        }

        return result

    except Exception as e:
        logger.warning(f"Error training model for {player_name}: {str(e)}")
        return None


class WalkForwardBacktest:

    def __init__(
        self,
        db_path: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        model_type: str = 'xgboost',
        model_params: Optional[Dict] = None,
        feature_config: str = 'default_features',
        output_dir: str = 'data/backtest_results',
        data_dir: Optional[str] = None,
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
        wmape_weight: str = 'actual_fpts'
    ):
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

        if data_dir:
            data_path = Path(data_dir)
            db_path_obj = Path(db_path)
            if not db_path_obj.is_absolute():
                db_path = str(data_path / db_path)

            output_path_obj = Path(output_dir)
            if not output_path_obj.is_absolute():
                self.output_dir = str(data_path / output_dir)
            else:
                self.output_dir = output_dir
        else:
            self.output_dir = output_dir

        storage = SQLiteStorage(db_path)
        self.loader = HistoricalDataLoader(storage)

        config = load_feature_config(feature_config)
        self.feature_pipeline = config.build_pipeline(FeaturePipeline)

        self.results = []
        self.all_predictions = []

        self.mape_metric = MAPEMetric()
        self.rmse_metric = RMSEMetric()
        self.mae_metric = MAEMetric()
        self.corr_metric = CorrelationMetric()
        self.cmape_metric = CappedMAPEMetric(cap=self.cmape_cap)
        self.smape_metric = SMAPEMetric()
        self.wmape_metric = WMAPEMetric()

        self.benchmark = None

        self.current_model = None
        self.player_models = {}
        self.last_training_date = None

        self.config = {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'model_type': model_type,
            'model_params': model_params or {},
            'feature_config': feature_config,
            'per_player_models': per_player_models,
            'min_player_games': min_player_games,
            'recalibrate_days': recalibrate_days,
            'num_seasons': num_seasons,
            'n_jobs': n_jobs,
            'rewrite_models': rewrite_models
        }
        # Persist evaluation knobs
        self.config['minutes_threshold'] = self.minutes_threshold
        self.config['cmape_cap'] = self.cmape_cap
        self.config['wmape_weight'] = self.wmape_weight

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized WalkForwardBacktest")
        if data_dir:
            logger.info(f"Architecture: Separated (data_dir={data_dir})")
            logger.info(f"  Database: {db_path}")
            logger.info(f"  Outputs: {self.output_dir}")
        else:
            logger.info(f"Architecture: Default (data in project directory)")
        logger.info(f"Training period: {train_start} to {train_end}")
        logger.info(f"Testing period: {test_start} to {test_end}")
        logger.info(f"Per-player models: {per_player_models}")
        logger.info(f"Feature config: {feature_config}")
        logger.info(f"Recalibrate every {recalibrate_days} days")
        logger.info(f"Rewrite models: {rewrite_models}")
        logger.info(f"Save models: {save_models}")
        logger.info(f"Save predictions: {save_predictions}")
        logger.info(f"Parallel jobs: {n_jobs} ({'all cores' if n_jobs == -1 else 'sequential' if n_jobs == 1 else f'{n_jobs} workers'})")
        logger.info(f"Using YAML-configured feature pipeline")
        logger.info(f"Metrics: minutes_threshold={self.minutes_threshold}, cmape_cap={self.cmape_cap}, wmape_weight={self.wmape_weight}")

    def _build_training_features(
        self,
        training_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training features using FeaturePipeline.
        """
        if training_data.empty:
            return pd.DataFrame(), pd.Series()

        df = training_data.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
        df = df.sort_values(['playerID', 'gameDate'])

        if 'fpts' not in df.columns:
            df['fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

        df['target'] = df.groupby('playerID')['fpts'].shift(-1)

        df = self.feature_pipeline.fit_transform(df)

        df = df.dropna(subset=['target'])

        metadata_cols = [
            'playerID', 'playerName', 'longName', 'team', 'teamAbv', 'teamID',
            'pos', 'gameDate', 'gameID', 'fpts', 'fantasyPoints', 'fantasyPts',
            'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins',
            'tech', 'created_at', 'updated_at'
        ]

        feature_cols = [col for col in df.columns if col not in metadata_cols]

        X = df[feature_cols].copy()
        X = X.fillna(0)

        # Convert any remaining object columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        y = df['target']

        return X, y

    def _build_slate_features(
        self,
        slate_data: Dict[str, Any],
        training_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build slate features using FeaturePipeline.
        """
        salaries_df = slate_data.get('dfs_salaries', pd.DataFrame()).copy()
        if salaries_df.empty:
            return pd.DataFrame()

        if 'longName' in salaries_df.columns and 'playerName' not in salaries_df.columns:
            salaries_df['playerName'] = salaries_df['longName']

        training_data = training_data.copy()
        training_data['gameDate'] = pd.to_datetime(training_data['gameDate'], format='%Y%m%d', errors='coerce')

        if 'fpts' not in training_data.columns:
            training_data['fpts'] = training_data.apply(calculate_dk_fantasy_points, axis=1)

        training_features = self.feature_pipeline.transform(training_data).copy()

        metadata_cols = [
            'playerID', 'playerName', 'longName', 'team', 'teamAbv', 'teamID',
            'pos', 'gameDate', 'gameID', 'fpts', 'fantasyPoints', 'fantasyPts',
            'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins',
            'tech', 'created_at', 'updated_at'
        ]
        # Convert any non-numeric feature columns to numeric
        for col in training_features.columns:
            if col not in metadata_cols and training_features[col].dtype == 'object':
                training_features[col] = pd.to_numeric(training_features[col], errors='coerce').fillna(0)

        all_features = []

        for _, player_row in salaries_df.iterrows():
            player_id = player_row['playerID']

            player_features = training_features[training_features['playerID'] == player_id]

            if player_features.empty:
                continue

            last_row = player_features.iloc[-1]

            features = {
                'playerID': player_id,
                'playerName': player_row.get('playerName', ''),
                'team': player_row.get('team', ''),
                'pos': player_row.get('pos', ''),
                'salary': player_row.get('salary', 0)
            }

            for col in last_row.index:
                if col not in metadata_cols:
                    features[col] = last_row[col]

            all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        return pd.DataFrame(all_features)

    def _should_recalibrate(self, current_date: str) -> bool:
        from datetime import datetime, timedelta

        if self.rewrite_models:
            return True

        if self.last_training_date is None:
            return True

        current_dt = datetime.strptime(current_date, '%Y%m%d')
        last_train_dt = datetime.strptime(self.last_training_date, '%Y%m%d')
        days_diff = (current_dt - last_train_dt).days

        return days_diff >= self.recalibrate_days

    def run(self) -> Dict[str, Any]:
        from datetime import datetime as dt

        backtest_start_time = time.perf_counter()

        if self.resume_from_run:
            self.run_timestamp = self.resume_from_run
            logger.info(f"RESUMING existing run: {self.run_timestamp}")
        else:
            self.run_timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
            logger.info(f"Starting NEW run: {self.run_timestamp}")

        if self.data_dir:
            data_path = Path(self.data_dir)
            output_path_obj = Path(self.output_dir)
            if not output_path_obj.is_absolute():
                base_output = data_path / self.output_dir
            else:
                base_output = Path(self.output_dir)
            self.run_output_dir = base_output / self.run_timestamp
        else:
            self.run_output_dir = Path('data') / 'outputs' / self.run_timestamp

        self.run_inputs_dir = self.run_output_dir / 'inputs'
        self.run_features_dir = self.run_output_dir / 'features'
        self.run_predictions_dir = self.run_output_dir / 'predictions'
        self.run_checkpoint_dir = self.run_output_dir / 'checkpoints'

        self.run_inputs_dir.mkdir(parents=True, exist_ok=True)
        self.run_features_dir.mkdir(parents=True, exist_ok=True)
        self.run_predictions_dir.mkdir(parents=True, exist_ok=True)
        self.run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("STARTING WALK-FORWARD BACKTEST")
        logger.info("="*80)
        logger.info(f"Run timestamp: {self.run_timestamp}")
        logger.info(f"Output directory: {self.run_output_dir}")
        logger.info(f"Training period: {self.train_start} to {self.train_end}")
        logger.info(f"Testing period: {self.test_start} to {self.test_end}")
        logger.info(f"Model: {self.model_type}")
        logger.info("="*80)

        completed_slates = self._load_checkpoint()
        if completed_slates:
            logger.info(f"RESUMING from checkpoint: {len(completed_slates)} slates already completed")
            logger.info(f"Completed dates: {sorted(completed_slates)}")
            logger.info("="*80)

        slate_dates = self.loader.load_slate_dates(self.test_start, self.test_end)

        if not slate_dates:
            logger.error("No slate dates found")
            return {'error': 'No slate dates found'}

        print(f"\nBacktesting {len(slate_dates)} slates from {self.test_start} to {self.test_end}\n")

        logger.info("="*80)
        logger.info("INITIALIZING BENCHMARK")
        logger.info("="*80)
        benchmark_start_time = time.perf_counter()
        logger.info("Loading training data for benchmark...")
        training_data_full = self.loader.load_historical_player_logs(
            start_date=self.train_start,
            end_date=self.train_end,
            num_seasons=self.num_seasons
        )
        logger.info(f"Loaded {len(training_data_full)} training records")

        logger.info("Building features for benchmark...")
        training_data_sorted = training_data_full.copy()
        training_data_sorted['gameDate'] = pd.to_datetime(training_data_sorted['gameDate'], format='%Y%m%d', errors='coerce')
        training_data_sorted = training_data_sorted.sort_values(['playerID', 'gameDate'])

        if 'fpts' not in training_data_sorted.columns:
            training_data_sorted['fpts'] = training_data_sorted.apply(calculate_dk_fantasy_points, axis=1)
            logger.info("Calculated fantasy points for training data")

        training_features = self.feature_pipeline.fit_transform(training_data_sorted)
        logger.info(f"Generated {len(training_features)} feature rows with {len([c for c in training_features.columns if c.startswith(('rolling_', 'ewma_'))])} features")

        df_qualified = training_features[
            training_features.groupby('playerID')['playerID'].transform('size') >= self.min_games_for_benchmark
        ].copy()
        logger.info(f"Qualified players: {df_qualified['playerID'].nunique()} (min_games={self.min_games_for_benchmark})")

        logger.info("Initializing SeasonAverageBenchmark...")
        self.benchmark = SeasonAverageBenchmark(min_games=self.min_games_for_benchmark)
        self.benchmark.fit(df_qualified)

        logger.info(f"Benchmark fitted successfully for {len(self.benchmark.player_averages)} players")

        top_5 = sorted(self.benchmark.player_averages.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 benchmark averages:")
        for player_id, avg_fpts in top_5:
            if player_id in df_qualified['playerID'].values:
                player_name = df_qualified[df_qualified['playerID'] == player_id]['longName'].iloc[0] if 'longName' in df_qualified.columns else 'Unknown'
                logger.info(f"  {player_name}: {avg_fpts:.2f} fpts")

        benchmark_elapsed = time.perf_counter() - benchmark_start_time
        logger.info(f"Benchmark initialization completed in {self._format_time(benchmark_elapsed)}")
        logger.info("="*80)
        logger.info("")

        slate_times = []
        for i, test_date in enumerate(tqdm(slate_dates, desc="Backtesting slates")):
            if test_date in completed_slates:
                logger.info(f"Skipping slate {i+1}/{len(slate_dates)}: {test_date} (already completed)")
                slate_result = self._load_slate_checkpoint(test_date)
                if slate_result:
                    self.results.append(slate_result['daily_result'])
                    merged_df = pd.read_parquet(self.run_predictions_dir / f"{test_date}_with_actuals.parquet")
                    self.all_predictions.append(merged_df)
                continue

            slate_start_time = time.perf_counter()
            logger.info('')
            logger.info(f"Processing slate {i+1}/{len(slate_dates)}: {test_date}")

            slate_data = self.loader.load_slate_data(test_date)

            salaries_df = slate_data.get('dfs_salaries', pd.DataFrame())
            if salaries_df.empty:
                logger.warning(f"No salary data for {test_date}, skipping")
                continue

            training_data = self.loader.load_historical_player_logs(
                start_date=self.train_start,
                end_date=self.train_end,
                num_seasons=self.num_seasons
            )

            if self.per_player_models:
                projections = self._generate_per_player_projections(
                    slate_data,
                    training_data,
                    test_date
                )

            else:
                should_train = self._should_recalibrate(test_date)

                if should_train:
                    X_train, y_train = self._build_training_features(training_data)

                    if X_train.empty or y_train.empty:
                        logger.warning(f"Feature generation failed for {test_date}")
                        continue

                    input_file = self.run_inputs_dir / f"slate_training_inputs_{test_date}.parquet"

                    model = self._train_model(
                        X_train,
                        y_train,
                        save_inputs=True,
                        input_save_path=str(input_file)
                    )
                    self.current_model = model
                    self.last_training_date = test_date
                    logger.info(f"Model trained and cached for {test_date}")
                    logger.info(f"Saved training inputs to {input_file}")

                    models_dir = Path(PER_SLATE_MODEL_DIR)
                    models_dir.mkdir(parents=True, exist_ok=True)
                    model_file = models_dir / f"{self.model_type}_{test_date}.pkl"

                    if hasattr(model, 'save'):
                        self._save_slate_model(model, model_file, test_date, len(X_train))
                        logger.info(f"Saved per-slate model to {model_file}")
                else:
                    model = self.current_model
                    logger.info(f"Reusing model from {self.last_training_date}")

                slate_features = self._build_slate_features(slate_data, training_data)

                if slate_features.empty:
                    logger.warning(f"No slate features for {test_date}")
                    continue

                projections = self._generate_projections(model, slate_features)

            if projections.empty:
                logger.warning(f"No projections generated for {test_date}")
                continue

            logger.info(f"Adding benchmark predictions...")
            projections['benchmark_pred'] = projections['playerID'].map(
                self.benchmark.player_averages
            ).fillna(0)
            has_benchmark = (projections['benchmark_pred'] > 0).sum()
            logger.info(f"Benchmark predictions: {has_benchmark}/{len(projections)} players")

            if self.save_predictions:
                logger.info("Saving predictions to parquet...")
                predictions_path = self.run_predictions_dir / f"{test_date}.parquet"
                projections.to_parquet(predictions_path)
                logger.info(f"Saved predictions: {predictions_path}")
                logger.info(f"  Players: {len(projections)}, Columns: {len(projections.columns)}")

            actuals = self._load_actuals(test_date)

            if actuals.empty:
                logger.warning(f"No actual results for {test_date}")
                continue

            logger.info("Evaluating predictions against actuals...")
            daily_results, merged_df = self._evaluate_slate(test_date, projections, actuals)

            if self.save_predictions and not merged_df.empty:
                logger.info("Saving results with actuals...")
                results_with_actuals_path = self.run_predictions_dir / f"{test_date}_with_actuals.parquet"
                merged_df.to_parquet(results_with_actuals_path)
                logger.info(f"Saved results with actuals: {results_with_actuals_path}")

            logger.info("")
            logger.info(f"Daily Evaluation Results for {test_date}:")
            logger.info(f"  Players evaluated: {daily_results['num_players']}")
            logger.info(f"  Model Performance:")
            logger.info(f"    MAPE: {daily_results['model_mape']:.2f}%")
            logger.info(f"    RMSE: {daily_results['model_rmse']:.2f}")
            logger.info(f"    MAE:  {daily_results['model_mae']:.2f}")
            logger.info(f"    Correlation: {daily_results['model_corr']:.3f}")
            if not np.isnan(daily_results.get('benchmark_mape', np.nan)):
                logger.info(f"  Benchmark Performance:")
                logger.info(f"    MAPE: {daily_results['benchmark_mape']:.2f}%")
                logger.info(f"    RMSE: {daily_results['benchmark_rmse']:.2f}")
                improvement = daily_results['benchmark_mape'] - daily_results['model_mape']
                status = "BETTER" if improvement > 0 else "WORSE"
                logger.info(f"  Improvement: {improvement:+.2f}% ({status})")
            logger.info(f"  Fantasy Points:")
            logger.info(f"    Mean Actual:     {daily_results['mean_actual']:.2f}")
            logger.info(f"    Mean Projected:  {daily_results['mean_projected']:.2f}")
            logger.info(f"    Mean Benchmark:  {daily_results['mean_benchmark']:.2f}")
            logger.info(f"    Mean Error:      {daily_results['mean_projected'] - daily_results['mean_actual']:+.2f}")

            if not merged_df.empty and 'salary_bin' in merged_df.columns:
                logger.info(f"  Salary Tier Breakdown:")
                for tier in merged_df['salary_bin'].unique():
                    if pd.notna(tier):
                        tier_data = merged_df[merged_df['salary_bin'] == tier]
                        tier_mape = self.mape_metric.calculate(tier_data['actual_fpts'], tier_data['projected_fpts'])
                        logger.info(f"    {tier}: {tier_mape:.1f}% (n={len(tier_data)})")

            self.results.append(daily_results)
            self.all_predictions.append(merged_df)

            self._save_slate_checkpoint(test_date, daily_results, merged_df)
            logger.info(f"Checkpoint saved for {test_date}")

            slate_elapsed = time.perf_counter() - slate_start_time
            slate_times.append(slate_elapsed)
            avg_slate_time = sum(slate_times) / len(slate_times)
            remaining_slates = len(slate_dates) - (i + 1)
            eta = remaining_slates * avg_slate_time
            logger.info(f"Slate completed in {self._format_time(slate_elapsed)} (avg: {self._format_time(avg_slate_time)}/slate, ETA: {self._format_time(eta)})")

        logger.info(f"Walk-forward backtest complete: {len(self.results)} slates processed")

        backtest_elapsed = time.perf_counter() - backtest_start_time
        logger.info(f"Total backtest time: {self._format_time(backtest_elapsed)}")

        results = self._aggregate_results()

        logger.info("="*80)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("="*80)

        try:
            report_generator = BacktestReportGenerator(self.run_output_dir)
            report_path = report_generator.generate_report(
                results=results,
                config=self.config,
                run_timestamp=self.run_timestamp
            )
            logger.info(f"Report generated: {report_path}")
            results['report_path'] = str(report_path)
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")

        return results

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        save_inputs: bool = False,
        input_save_path: str = None
    ):
        logger.debug(f"Training {self.model_type} model on {len(X_train)} samples")

        if self.model_type == 'xgboost':
            try:
                model = XGBoostModel(self.model_params)
                model.train(X_train, y_train, save_inputs=save_inputs, input_save_path=input_save_path)
                return model
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                self.model_type = 'random_forest'

        if self.model_type == 'random_forest':
            # Filter out XGBoost-specific parameters
            rf_params = {k: v for k, v in self.model_params.items()
                         if k not in ['learning_rate', 'objective', 'colsample_bytree', 'subsample', 'min_child_weight']}
            model = RandomForestModel(rf_params)
            model.train(X_train, y_train, save_inputs=save_inputs, input_save_path=input_save_path)
            return model

        if self.model_type == 'linear':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            return model

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _generate_projections(self, model, slate_features: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Generating projections for {len(slate_features)} players")

        metadata_cols = ['playerID', 'playerName', 'team', 'pos', 'salary']
        feature_cols = [col for col in slate_features.columns if col not in metadata_cols]

        X = slate_features[feature_cols].fillna(0)

        if hasattr(model, 'predict'):
            if hasattr(model, 'is_trained') and not model.is_trained:
                raise ValueError("Model must be trained before prediction")
            predictions = model.predict(X)
        else:
            predictions = model.predict(X)

        projections = slate_features[metadata_cols].copy()
        projections['projected_fpts'] = predictions
        projections['salary'] = pd.to_numeric(projections['salary'], errors='coerce')
        projections['value'] = projections['projected_fpts'] / (projections['salary'] / 1000)

        logger.debug(f"Projections: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")

        return projections

    def _load_actuals(self, date: str) -> pd.DataFrame:
        try:
            filters = {'start_date': date, 'end_date': date}
            df = self.loader.storage.load('box_scores', filters)

            if df.empty:
                return pd.DataFrame()

            df['actual_fpts'] = df.apply(calculate_dk_fantasy_points, axis=1)

            if 'longName' in df.columns and 'playerName' not in df.columns:
                df['playerName'] = df['longName']

            # Include actual minutes if available for filtering/weighting
            if 'mins' in df.columns:
                df['actual_mins'] = pd.to_numeric(df['mins'], errors='coerce')
            else:
                df['actual_mins'] = np.nan

            return df[['playerID', 'playerName', 'team', 'pos', 'actual_fpts', 'actual_mins']]

        except Exception as e:
            logger.error(f"Failed to load actuals for {date}: {str(e)}")
            return pd.DataFrame()

    def _evaluate_slate(
        self,
        test_date: str,
        projections: pd.DataFrame,
        actuals: pd.DataFrame
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        logger.debug(f"Evaluating slate {test_date}")

        logger.debug(f"Merging projections with actuals...")
        merged = projections.merge(
            actuals[['playerID', 'actual_fpts']],
            on='playerID',
            how='inner'
        )
        # Merge minutes if present
        if 'actual_mins' in actuals.columns and 'actual_mins' not in merged.columns:
            merged = merged.merge(actuals[['playerID', 'actual_mins']], on='playerID', how='left')
        logger.debug(f"Matched {len(merged)}/{len(projections)} players with actuals")

        if merged.empty:
            logger.warning(f"No matching players for {test_date}")
            return {
                'date': test_date,
                'num_players': 0,
                'model_mape': np.nan,
                'model_rmse': np.nan,
                'model_mae': np.nan,
                'model_corr': np.nan,
                'benchmark_mape': np.nan,
                'benchmark_rmse': np.nan,
                'mean_projected': np.nan,
                'mean_actual': np.nan,
                'mean_benchmark': np.nan
            }, pd.DataFrame()

        y_true = merged['actual_fpts'].values
        y_pred = merged['projected_fpts'].values

        logger.debug("Calculating model metrics...")
        model_mape = self.mape_metric.calculate(y_true, y_pred)
        model_rmse = self.rmse_metric.calculate(y_true, y_pred)
        model_mae = self.mae_metric.calculate(y_true, y_pred)
        model_corr = self.corr_metric.calculate(y_true, y_pred)
        model_cmape = self.cmape_metric.calculate(y_true, y_pred)
        model_smape = self.smape_metric.calculate(y_true, y_pred)
        # Build weights for WMAPE
        weights_series = None
        if self.wmape_weight == 'actual_fpts':
            weights_series = merged['actual_fpts'].clip(lower=1.0)
        elif self.wmape_weight == 'actual_mins' and 'actual_mins' in merged.columns:
            weights_series = merged['actual_mins'].fillna(0).clip(lower=1.0)
        elif self.wmape_weight == 'expected_mins' and 'expected_mins' in merged.columns:
            weights_series = merged['expected_mins'].fillna(0).clip(lower=1.0)
        else:
            # Fallback
            weights_series = merged['actual_fpts'].clip(lower=1.0)
        model_wmape = self.wmape_metric.calculate(y_true, y_pred, weights_series.values)

        logger.debug("Calculating benchmark metrics...")
        has_benchmark = (merged['benchmark_pred'] > 0)
        benchmark_mape = self.mape_metric.calculate(
            merged[has_benchmark]['actual_fpts'],
            merged[has_benchmark]['benchmark_pred']
        ) if has_benchmark.any() else np.nan

        benchmark_rmse = self.rmse_metric.calculate(
            merged[has_benchmark]['actual_fpts'],
            merged[has_benchmark]['benchmark_pred']
        ) if has_benchmark.any() else np.nan

        benchmark_cmape = self.cmape_metric.calculate(
            merged[has_benchmark]['actual_fpts'],
            merged[has_benchmark]['benchmark_pred']
        ) if has_benchmark.any() else np.nan

        benchmark_wmape = self.wmape_metric.calculate(
            merged[has_benchmark]['actual_fpts'],
            merged[has_benchmark]['benchmark_pred'],
            (weights_series[has_benchmark]).values if has_benchmark.any() else None
        ) if has_benchmark.any() else np.nan

        logger.debug(f"Creating salary tiers (bins: {self.salary_tiers})...")
        merged['salary'] = merged['salary'].astype(int)
        merged['salary_bin'] = pd.cut(
            merged['salary'],
            bins=self.salary_tiers,
            labels=['Low', 'Mid', 'High', 'Elite'][:len(self.salary_tiers)-1]
        )

        # Minutes-based filtering (use expected_mins if present, else actual_mins)
        minute_col = 'expected_mins' if 'expected_mins' in merged.columns else 'actual_mins' if 'actual_mins' in merged.columns else None
        if minute_col is not None:
            filt_mask = merged[minute_col].fillna(0) >= self.minutes_threshold
        else:
            filt_mask = pd.Series([True] * len(merged))

        if filt_mask.any():
            y_true_f = merged.loc[filt_mask, 'actual_fpts'].values
            y_pred_f = merged.loc[filt_mask, 'projected_fpts'].values
            weights_f = (weights_series[filt_mask]).values if len(weights_series) == len(merged) else None
            model_cmape_f = self.cmape_metric.calculate(y_true_f, y_pred_f)
            model_smape_f = self.smape_metric.calculate(y_true_f, y_pred_f)
            model_wmape_f = self.wmape_metric.calculate(y_true_f, y_pred_f, weights_f)
        else:
            model_cmape_f = np.nan
            model_smape_f = np.nan
            model_wmape_f = np.nan

        result = {
            'date': test_date,
            'num_players': len(merged),
            'model_mape': model_mape,
            'model_rmse': model_rmse,
            'model_mae': model_mae,
            'model_corr': model_corr,
            'model_cmape': model_cmape,
            'model_smape': model_smape,
            'model_wmape': model_wmape,
            'model_cmape_filtered': model_cmape_f,
            'model_smape_filtered': model_smape_f,
            'model_wmape_filtered': model_wmape_f,
            'benchmark_mape': benchmark_mape,
            'benchmark_rmse': benchmark_rmse,
            'benchmark_cmape': benchmark_cmape,
            'benchmark_wmape': benchmark_wmape,
            'mean_projected': merged['projected_fpts'].mean(),
            'mean_actual': merged['actual_fpts'].mean(),
            'mean_benchmark': merged['benchmark_pred'].mean(),
            'minutes_threshold': self.minutes_threshold,
            'wmape_weight': self.wmape_weight,
            'cmape_cap': self.cmape_cap,
            'num_players_filtered': int(filt_mask.sum()),
            'num_low_minutes': int((~filt_mask).sum())
        }

        logger.debug(f"Evaluation complete: MAPE={model_mape:.2f}%, RMSE={model_rmse:.2f}, MAE={model_mae:.2f}, Corr={model_corr:.3f}")

        return result, merged

    def _generate_per_player_projections(
        self,
        slate_data: Dict[str, Any],
        training_data: pd.DataFrame,
        test_date: str
    ) -> pd.DataFrame:
        logger.info("Generating per-player model projections")

        salaries_df = slate_data['dfs_salaries'].copy()
        all_projections = []

        total_players = len(salaries_df)
        players_with_models = 0
        models_trained = 0
        models_reused = 0

        should_recalibrate = self._should_recalibrate(test_date)

        models_dir = Path(PER_PLAYER_MODEL_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        if should_recalibrate:
            logger.info(f"Recalibrating player models, saving to: {models_dir}")
        else:
            logger.info(f"Reusing cached player models from {self.last_training_date}")

        per_player_start_time = time.perf_counter()

        if should_recalibrate and self.n_jobs != 1:
            logger.info(f"Training models in parallel with {self.n_jobs} workers")

            player_rows = [row for _, row in salaries_df.iterrows()]

            results = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(_train_single_player_model)(
                    player_row,
                    training_data,
                    self.feature_pipeline,
                    self.model_type,
                    self.model_params,
                    self.min_player_games,
                    self.save_models,
                    models_dir,
                    self.run_inputs_dir
                )
                for player_row in player_rows
            )

            for result in results:
                if result is not None:
                    player_id = result['playerID']
                    model = result.pop('model')

                    if model is not None:
                        self.player_models[player_id] = model
                        models_trained += 1

                        if self.save_models:
                            player_name = result['playerName']
                            safe_player_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in player_name)
                            safe_player_name = safe_player_name.replace(' ', '_')
                            model_file = models_dir / f"{safe_player_name}_{player_id}.pkl"

                            if hasattr(model, 'save'):
                                self._save_model(model, model_file, player_name, player_id, 0)

                    all_projections.append(pd.DataFrame([result]))
                    players_with_models += 1

            logger.info(f"Parallel training complete: {models_trained} models trained")

        else:
            model_train_times = []
            log_interval = max(1, total_players // 10)

            for idx, player_row in tqdm(salaries_df.iterrows(), total=len(salaries_df), desc="Per-player models", leave=False):
                player_id = player_row.get('playerID')
                player_name = player_row.get('longName')
                logger.debug(f"Processing {player_name} ({player_id}) for {test_date}")

                player_training_data = training_data[training_data['playerID'] == player_id].copy()

                if len(player_training_data) < self.min_player_games:
                    logger.debug(f"Skipping {player_name}: only {len(player_training_data)} games (need {self.min_player_games})")
                    continue

                try:
                    if should_recalibrate or player_id not in self.player_models:
                        model_start_time = time.perf_counter()

                        X_train, y_train = self._build_training_features(player_training_data)

                        if X_train.empty or y_train.empty or len(X_train) < 3:
                            logger.debug(f"Insufficient features for {player_name}")
                            continue

                        safe_player_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in player_name)
                        safe_player_name = safe_player_name.replace(' ', '_')

                        input_file = self.run_inputs_dir / f"player_{safe_player_name}_{player_id}_inputs.parquet"

                        model = self._train_model(
                            X_train,
                            y_train,
                            save_inputs=True,
                            input_save_path=str(input_file)
                        )
                        self.player_models[player_id] = model
                        models_trained += 1

                        model_elapsed = time.perf_counter() - model_start_time
                        model_train_times.append(model_elapsed)

                        if models_trained % log_interval == 0:
                            avg_model_time = sum(model_train_times) / len(model_train_times)
                            models_per_sec = 1 / avg_model_time if avg_model_time > 0 else 0
                            remaining_models = total_players - (idx + 1)
                            eta_models = remaining_models * avg_model_time
                            logger.info(f"  Progress: {models_trained} models trained ({models_trained/total_players*100:.1f}%) - {models_per_sec:.2f} models/sec - ETA: {self._format_time(eta_models)}")

                        model_file = models_dir / f"{safe_player_name}_{player_id}.pkl"

                        if hasattr(model, 'save'):
                            self._save_model(model, model_file, player_name, player_id, len(X_train))
                            logger.debug(f"Saved model for {player_name} to {model_file}")
                            logger.debug(f"Saved training inputs to {input_file}")
                    else:
                        model = self.player_models[player_id]
                        models_reused += 1
                        logger.debug(f"Reusing cached model for {player_name}")

                    slate_data_single = {
                        'dfs_salaries': salaries_df.iloc[[idx]],
                        'date': slate_data['date'],
                        'schedule': slate_data.get('schedule', pd.DataFrame()),
                        'betting_odds': slate_data.get('betting_odds', pd.DataFrame()),
                        'injuries': slate_data.get('injuries', pd.DataFrame())
                    }

                    slate_features = self._build_slate_features(slate_data_single, player_training_data)

                    if slate_features.empty:
                        logger.debug(f"No features generated for {player_name}")
                        continue

                    projection = self._generate_projections(model, slate_features)
                    all_projections.append(projection)
                    players_with_models += 1

                except Exception as e:
                    logger.warning(f"Error generating projection for {player_name}: {str(e)}")
                    continue

            if model_train_times:
                avg_train_time = sum(model_train_times) / len(model_train_times)
                logger.info(f"Average model training time: {self._format_time(avg_train_time)} ({1/avg_train_time:.2f} models/sec)")

        if should_recalibrate:
            self.last_training_date = test_date

        if not all_projections:
            logger.warning("No player projections generated")
            return pd.DataFrame()

        projections_df = pd.concat(all_projections, ignore_index=True)

        per_player_elapsed = time.perf_counter() - per_player_start_time
        logger.info(f"Generated projections for {players_with_models}/{total_players} players in {self._format_time(per_player_elapsed)}")
        logger.info(f"Models trained: {models_trained}, reused: {models_reused}")

        return projections_df

    def _save_model(self, model, model_file: Path, player_name: str, player_id: str, num_samples: int):
        import json

        if self.save_models:
            model_data = {
                'model': model,
                'player_name': player_name,
                'player_id': player_id,
                'num_training_samples': num_samples,
                'model_type': self.model_type,
                'feature_config': self.feature_config_name
            }

            if hasattr(model, 'save'):
                model.save(str(model_file))
            else:
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)

            metadata_file = model_file.with_suffix('.json')
            metadata = {
                'player_name': player_name,
                'player_id': player_id,
                'num_training_samples': num_samples,
                'model_type': self.model_type,
                'feature_config': self.feature_config_name,
                'model_file': str(model_file.name)
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Saved model and metadata for {player_name}")

    def _save_slate_model(self, model, model_file: Path, test_date: str, num_samples: int):
        import json

        if self.save_models:
            if hasattr(model, 'save'):
                model.save(str(model_file))
            else:
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)

            metadata_file = model_file.with_suffix('.json')
            metadata = {
                'training_date': test_date,
                'num_training_samples': num_samples,
                'model_type': self.model_type,
                'feature_config': self.feature_config_name,
                'model_file': str(model_file.name)
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Saved per-slate model and metadata for {test_date}")

    def _format_time(self, seconds: float) -> str:
        """Format elapsed time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.0f}s"

    def _save_slate_checkpoint(self, test_date: str, daily_results: Dict[str, Any], merged_df: pd.DataFrame):
        """Save checkpoint after each slate completion."""
        import json

        checkpoint_file = self.run_checkpoint_dir / f"{test_date}.json"
        checkpoint_data = {
            'test_date': test_date,
            'completed_at': pd.Timestamp.now().isoformat(),
            'daily_result': daily_results,
            'num_players': daily_results.get('num_players', 0),
            'model_mape': daily_results.get('model_mape', None),
            'benchmark_mape': daily_results.get('benchmark_mape', None)
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        progress_file = self.run_checkpoint_dir / 'progress.json'
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {
                'run_timestamp': self.run_timestamp,
                'test_start': self.test_start,
                'test_end': self.test_end,
                'completed_slates': [],
                'last_updated': None
            }

        if test_date not in progress['completed_slates']:
            progress['completed_slates'].append(test_date)
        progress['last_updated'] = pd.Timestamp.now().isoformat()
        progress['total_completed'] = len(progress['completed_slates'])

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)

        logger.debug(f"Saved checkpoint: {checkpoint_file}")

    def _load_checkpoint(self) -> set:
        """Load completed slate dates from checkpoint directory."""
        import json

        if not self.run_checkpoint_dir.exists():
            return set()

        completed_slates = set()
        for checkpoint_file in self.run_checkpoint_dir.glob('*.json'):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    completed_slates.add(checkpoint_data['test_date'])
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {str(e)}")

        return completed_slates

    def _load_slate_checkpoint(self, test_date: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data for a specific slate."""
        import json

        checkpoint_file = self.run_checkpoint_dir / f"{test_date}.json"
        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint for {test_date}: {str(e)}")
            return None

    def _aggregate_results(self) -> Dict[str, Any]:
        logger.info("Aggregating backtest results")

        if not self.results:
            return {
                'error': 'No results to aggregate',
                'num_slates': 0
            }

        daily_df = pd.DataFrame(self.results)
        all_predictions_df = pd.concat(self.all_predictions, ignore_index=True) if self.all_predictions else pd.DataFrame()

        logger.info("="*80)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("="*80)
        logger.info(f"Number of Slates: {len(daily_df)}")
        logger.info(f"Date Range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        logger.info(f"Total Players Evaluated: {daily_df['num_players'].sum():.0f}")
        logger.info(f"Average Players per Slate: {daily_df['num_players'].mean():.1f}")
        logger.info("")
        logger.info("Model Performance:")
        logger.info(f"  Mean MAPE: {daily_df['model_mape'].mean():.2f}%")
        if 'model_cmape' in daily_df.columns:
            logger.info(f"  Mean cMAPE: {daily_df['model_cmape'].mean():.2f}%")
        if 'model_wmape' in daily_df.columns:
            logger.info(f"  Mean WMAPE: {daily_df['model_wmape'].mean():.2f}%")
        logger.info(f"  Median MAPE: {daily_df['model_mape'].median():.2f}%")
        logger.info(f"  Std MAPE: {daily_df['model_mape'].std():.2f}%")
        logger.info(f"  Mean RMSE: {daily_df['model_rmse'].mean():.2f}")
        logger.info(f"  Std RMSE: {daily_df['model_rmse'].std():.2f}")
        logger.info(f"  Mean MAE: {daily_df['model_mae'].mean():.2f}")
        logger.info(f"  Mean Correlation: {daily_df['model_corr'].mean():.3f}")
        logger.info(f"  Std Correlation: {daily_df['model_corr'].std():.3f}")
        logger.info("")
        logger.info("Benchmark Performance:")
        logger.info(f"  Mean MAPE: {daily_df['benchmark_mape'].mean():.2f}%")
        logger.info(f"  Median MAPE: {daily_df['benchmark_mape'].median():.2f}%")
        logger.info("")
        mape_improvement = daily_df['benchmark_mape'].mean() - daily_df['model_mape'].mean()
        logger.info("Improvement (Model vs Benchmark):")
        logger.info(f"  MAPE Improvement: {mape_improvement:+.2f}% {'(Model better)' if mape_improvement > 0 else '(Benchmark better)'}")
        logger.info("")

        logger.info("")
        logger.info("Performing benchmark comparison analysis...")
        comparison_df = all_predictions_df[
            (all_predictions_df['projected_fpts'] > 0) &
            (all_predictions_df['benchmark_pred'] > 0)
        ].copy()
        logger.info(f"Valid comparisons: {len(comparison_df)} players")

        if not comparison_df.empty and self.benchmark is not None:
            logger.info("Comparing model vs benchmark performance...")
            comparison_results = self.benchmark.compare_with_model(
                actual=comparison_df['actual_fpts'],
                model_pred=comparison_df['projected_fpts'],
                benchmark_pred=comparison_df['benchmark_pred']
            )

            logger.info("="*80)
            logger.info("BENCHMARK COMPARISON")
            logger.info("="*80)
            logger.info(comparison_results['summary'])

            logger.info("")
            logger.info("Analyzing performance by salary tier...")
            tier_data = comparison_df.rename(columns={
                'projected_fpts': 'model_pred',
                'actual_fpts': 'actual'
            })
            tier_data['salary'] = pd.to_numeric(tier_data['salary'], errors='coerce')
            tier_data = tier_data[tier_data['salary'].notna()]
            logger.info(f"Players with valid salary data: {len(tier_data)}")

            tier_comparison = self.benchmark.compare_by_salary_tier(tier_data, self.salary_tiers)

            logger.info("")
            logger.info("Performance by Salary Tier:")
            logger.info("="*80)
            for _, row in tier_comparison.iterrows():
                improvement = row['mape_improvement']
                status = 'BETTER' if improvement > 0 else 'WORSE'
                logger.info(f"\n{str(row['salary_tier']):20} {improvement:+6.1f}% {status:8} "
                           f"(Model: {row['model_mape']:.1f}%, Benchmark: {row['benchmark_mape']:.1f}%)")
                logger.info(f"{'':20}   Players: {row['count']}")

            logger.info("")
            logger.info("Performing statistical significance testing...")
            comparison_df['model_error'] = np.abs(comparison_df['projected_fpts'] - comparison_df['actual_fpts'])
            comparison_df['benchmark_error'] = np.abs(comparison_df['benchmark_pred'] - comparison_df['actual_fpts'])

            model_errors = comparison_df['model_error'].values
            benchmark_errors = comparison_df['benchmark_error'].values
            logger.info(f"Comparing {len(model_errors)} paired errors")

            logger.info("Running paired t-test...")
            t_stat, p_value = stats.ttest_rel(model_errors, benchmark_errors)

            logger.info("")
            logger.info("Statistical Significance Test (Paired t-test):")
            logger.info(f"  t-statistic: {t_stat:.4f}")
            logger.info(f"  p-value: {p_value:.6f}")
            logger.info(f"  Significance level: 0.05")
            if p_value < 0.05:
                if t_stat < 0:
                    logger.info("  Result: Model is SIGNIFICANTLY BETTER than benchmark (p < 0.05)")
                else:
                    logger.info("  Result: Model is SIGNIFICANTLY WORSE than benchmark (p < 0.05)")
            else:
                logger.info("  Result: No significant difference between model and benchmark (p >= 0.05)")

            logger.info("")
            logger.info("Calculating effect size (Cohen's d)...")
            cohens_d = (model_errors.mean() - benchmark_errors.mean()) / np.sqrt((model_errors.std()**2 + benchmark_errors.std()**2) / 2)
            logger.info(f"  Cohen's d: {cohens_d:.4f}")

            if abs(cohens_d) < 0.2:
                effect_size = 'negligible'
            elif abs(cohens_d) < 0.5:
                effect_size = 'small'
            elif abs(cohens_d) < 0.8:
                effect_size = 'medium'
            else:
                effect_size = 'large'
            logger.info(f"  Effect size: {effect_size}")
            logger.info(f"  Interpretation: {'d < 0.2' if abs(cohens_d) < 0.2 else 'd < 0.5' if abs(cohens_d) < 0.5 else 'd < 0.8' if abs(cohens_d) < 0.8 else 'd >= 0.8'}")

            logger.info("")
            logger.info("Error Statistics:")
            logger.info(f"  Model MAE: {model_errors.mean():.2f}")
            logger.info(f"  Benchmark MAE: {benchmark_errors.mean():.2f}")
            logger.info(f"  Difference: {benchmark_errors.mean() - model_errors.mean():.2f} (positive = model better)")
            logger.info(f"  Model Std: {model_errors.std():.2f}")
            logger.info(f"  Benchmark Std: {benchmark_errors.std():.2f}")

        # Low-min cohort summary
        low_minutes_metrics = None
        if 'num_low_minutes' in daily_df.columns and daily_df['num_low_minutes'].sum() > 0:
            # Build across-all-slates combined merged predictions
            all_preds = pd.concat(self.all_predictions, ignore_index=True) if self.all_predictions else pd.DataFrame()
            if not all_preds.empty:
                minute_col = 'expected_mins' if 'expected_mins' in all_preds.columns else 'actual_mins' if 'actual_mins' in all_preds.columns else None
                if minute_col is not None:
                    mask_lm = all_preds[minute_col].fillna(0) < self.minutes_threshold
                    if mask_lm.any():
                        y_true_lm = all_preds.loc[mask_lm, 'actual_fpts'].values
                        y_pred_lm = all_preds.loc[mask_lm, 'projected_fpts'].values
                        weights_lm = None
                        if self.wmape_weight == 'actual_fpts':
                            weights_lm = np.maximum(all_preds.loc[mask_lm, 'actual_fpts'].values, 1.0)
                        elif self.wmape_weight == 'actual_mins' and 'actual_mins' in all_preds.columns:
                            weights_lm = np.maximum(all_preds.loc[mask_lm, 'actual_mins'].fillna(0).values, 1.0)
                        elif self.wmape_weight == 'expected_mins' and 'expected_mins' in all_preds.columns:
                            weights_lm = np.maximum(all_preds.loc[mask_lm, 'expected_mins'].fillna(0).values, 1.0)

                        low_minutes_metrics = {
                            'count': int(mask_lm.sum()),
                            'cmape': self.cmape_metric.calculate(y_true_lm, y_pred_lm),
                            'smape': self.smape_metric.calculate(y_true_lm, y_pred_lm),
                            'wmape': self.wmape_metric.calculate(y_true_lm, y_pred_lm, weights_lm) if weights_lm is not None else np.nan
                        }

        summary = {
            'num_slates': len(daily_df),
            'date_range': f"{daily_df['date'].min()} to {daily_df['date'].max()}",
            'model_mean_mape': daily_df['model_mape'].mean(),
            'model_median_mape': daily_df['model_mape'].median(),
            'model_std_mape': daily_df['model_mape'].std(),
            'model_mean_cmape': daily_df['model_cmape'].mean() if 'model_cmape' in daily_df.columns else np.nan,
            'model_mean_smape': daily_df['model_smape'].mean() if 'model_smape' in daily_df.columns else np.nan,
            'model_mean_wmape': daily_df['model_wmape'].mean() if 'model_wmape' in daily_df.columns else np.nan,
            'model_mean_rmse': daily_df['model_rmse'].mean(),
            'model_std_rmse': daily_df['model_rmse'].std(),
            'model_mean_mae': daily_df['model_mae'].mean(),
            'model_mean_correlation': daily_df['model_corr'].mean(),
            'model_std_correlation': daily_df['model_corr'].std(),
            'benchmark_mean_mape': daily_df['benchmark_mape'].mean(),
            'benchmark_median_mape': daily_df['benchmark_mape'].median(),
            'benchmark_mean_cmape': daily_df['benchmark_cmape'].mean() if 'benchmark_cmape' in daily_df.columns else np.nan,
            'benchmark_mean_wmape': daily_df['benchmark_wmape'].mean() if 'benchmark_wmape' in daily_df.columns else np.nan,
            'mape_improvement': mape_improvement,
            'total_players_evaluated': daily_df['num_players'].sum(),
            'avg_players_per_slate': daily_df['num_players'].mean(),
            'daily_results': daily_df,
            'all_predictions': all_predictions_df
        }

        if low_minutes_metrics is not None:
            summary['low_minutes_metrics'] = low_minutes_metrics

        if not comparison_df.empty and self.benchmark is not None:
            summary['benchmark_comparison'] = comparison_results
            summary['tier_comparison'] = tier_comparison
            summary['statistical_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_size': effect_size
            }

        logger.info("="*80)
        logger.info(f"Aggregation complete: {summary['num_slates']} slates")
        logger.info(f"  Model MAPE: {summary['model_mean_mape']:.2f}%")
        logger.info(f"  Benchmark MAPE: {summary['benchmark_mean_mape']:.2f}%")
        logger.info(f"  Improvement: {summary['mape_improvement']:+.2f}%")
        logger.info(f"  Predictions saved: {len(summary['all_predictions'])} rows")
        if 'statistical_test' in summary:
            logger.info(f"  Statistical test: p={summary['statistical_test']['p_value']:.6f}")
        logger.info("="*80)

        return summary
