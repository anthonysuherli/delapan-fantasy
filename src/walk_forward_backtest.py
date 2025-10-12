import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.loaders.historical_loader import HistoricalDataLoader
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.evaluation.metrics.accuracy import MAPEMetric, RMSEMetric, CorrelationMetric
from src.utils.fantasy_points import calculate_dk_fantasy_points
from src.features.pipeline import FeaturePipeline
from src.features.transformers.rolling_stats import RollingStatsTransformer
from src.features.transformers.ewma import EWMATransformer
from src.config.paths import (
    PER_PLAYER_MODEL_DIR,
    PER_SLATE_MODEL_DIR,
    PER_PLAYER_TRAINING_INPUTS_DIR,
    PER_SLATE_TRAINING_INPUTS_DIR
)

logger = logging.getLogger(__name__)


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
        rolling_window_sizes: Optional[List[int]] = None,
        output_dir: str = 'data/backtest_results',
        per_player_models: bool = False,
        min_player_games: int = 10,
        recalibrate_days: int = 7,
        num_seasons: int = 2
    ):
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.model_type = model_type
        self.model_params = model_params or {}
        self.rolling_window_sizes = rolling_window_sizes or [3, 5, 10]
        self.output_dir = output_dir
        self.per_player_models = per_player_models
        self.min_player_games = min_player_games
        self.recalibrate_days = recalibrate_days
        self.num_seasons = num_seasons
        
        storage = SQLiteStorage(db_path)
        self.loader = HistoricalDataLoader(storage)

        self.feature_pipeline = FeaturePipeline()
        self.feature_pipeline.add(RollingStatsTransformer(
            windows=self.rolling_window_sizes,
            stats=['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins'],
            include_std=True
        ))
        self.feature_pipeline.add(EWMATransformer(
            span=5,
            stats=['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins']
        ))

        self.results = []

        self.mape_metric = MAPEMetric()
        self.rmse_metric = RMSEMetric()
        self.corr_metric = CorrelationMetric()

        self.current_model = None
        self.player_models = {}
        self.last_training_date = None

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized WalkForwardBacktest")
        logger.info(f"Training period: {train_start} to {train_end}")
        logger.info(f"Testing period: {test_start} to {test_end}")
        logger.info(f"Per-player models: {per_player_models}")
        logger.info(f"Recalibrate every {recalibrate_days} days")
        logger.info(f"Using new registry-based architecture with FeaturePipeline")

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

        metadata_cols = ['playerID', 'playerName', 'team', 'pos', 'gameDate', 'fpts', 'target', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins']

        for col in df.columns:
            if col not in metadata_cols and df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        feature_cols = [col for col in df.columns if col not in metadata_cols]

        X = df[feature_cols].fillna(0)
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
        salaries_df = slate_data.get('dfs_salaries', pd.DataFrame())
        if salaries_df.empty:
            return pd.DataFrame()

        if 'longName' in salaries_df.columns and 'playerName' not in salaries_df.columns:
            salaries_df['playerName'] = salaries_df['longName']

        training_data = training_data.copy()
        training_data['gameDate'] = pd.to_datetime(training_data['gameDate'], format='%Y%m%d', errors='coerce')

        if 'fpts' not in training_data.columns:
            training_data['fpts'] = training_data.apply(calculate_dk_fantasy_points, axis=1)

        training_features = self.feature_pipeline.transform(training_data)

        metadata_cols = ['playerID', 'playerName', 'team', 'pos', 'gameDate', 'fpts', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'mins']
        for col in training_features.columns:
            if col not in metadata_cols and training_features[col].dtype == 'object':
                training_features[col] = training_features[col].astype('category')

        all_features = []

        for _, player_row in salaries_df.iterrows():
            player_id = player_row['playerID']

            player_features = training_features[training_features['playerID'] == player_id]

            if len(player_features) < min(self.rolling_window_sizes):
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

        if self.last_training_date is None:
            return True

        current_dt = datetime.strptime(current_date, '%Y%m%d')
        last_train_dt = datetime.strptime(self.last_training_date, '%Y%m%d')
        days_diff = (current_dt - last_train_dt).days

        return days_diff >= self.recalibrate_days

    def run(self) -> Dict[str, Any]:
        logger.info("="*80)
        logger.info("STARTING WALK-FORWARD BACKTEST")
        logger.info("="*80)
        logger.info(f"Training period: {self.train_start} to {self.train_end}")
        logger.info(f"Testing period: {self.test_start} to {self.test_end}")
        logger.info(f"Model: {self.model_type}")
        logger.info("="*80)

        slate_dates = self.loader.load_slate_dates(self.test_start, self.test_end)

        if not slate_dates:
            logger.error("No slate dates found")
            return {'error': 'No slate dates found'}

        print(f"\nBacktesting {len(slate_dates)} slates from {self.test_start} to {self.test_end}\n")

        for i, test_date in enumerate(tqdm(slate_dates, desc="Backtesting slates")):
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

                    inputs_dir = Path(PER_SLATE_TRAINING_INPUTS_DIR)
                    inputs_dir.mkdir(parents=True, exist_ok=True)
                    input_file = inputs_dir / f"training_inputs_{test_date}.parquet"

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

            actuals = self._load_actuals(test_date)

            if actuals.empty:
                logger.warning(f"No actual results for {test_date}")
                continue

            daily_results = self._evaluate_slate(test_date, projections, actuals)

            logger.info(f"Daily evaluation for {test_date}:")
            logger.info(f"  Players evaluated: {daily_results['num_players']}")
            logger.info(f"  MAPE: {daily_results['mape']:.2f}%")
            logger.info(f"  RMSE: {daily_results['rmse']:.2f}")
            logger.info(f"  Correlation: {daily_results['correlation']:.3f}")
            logger.info(f"  Mean projected: {daily_results['mean_projected']:.2f}")
            logger.info(f"  Mean actual: {daily_results['mean_actual']:.2f}")

            self.results.append(daily_results)

        logger.info(f"Walk-forward backtest complete: {len(self.results)} slates processed")

        return self._aggregate_results()

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
            model = RandomForestModel(self.model_params)
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

            return df[['playerID', 'playerName', 'team', 'pos', 'actual_fpts']]

        except Exception as e:
            logger.error(f"Failed to load actuals for {date}: {str(e)}")
            return pd.DataFrame()

    def _evaluate_slate(
        self,
        test_date: str,
        projections: pd.DataFrame,
        actuals: pd.DataFrame
    ) -> Dict[str, Any]:
        logger.debug(f"Evaluating slate {test_date}")

        merged = projections.merge(
            actuals[['playerID', 'actual_fpts']],
            on='playerID',
            how='inner'
        )

        if merged.empty:
            logger.warning(f"No matching players for {test_date}")
            return {
                'date': test_date,
                'num_players': 0,
                'mape': np.nan,
                'rmse': np.nan,
                'correlation': np.nan
            }

        y_true = merged['actual_fpts'].values
        y_pred = merged['projected_fpts'].values

        mape = self.mape_metric.calculate(y_true, y_pred)
        rmse = self.rmse_metric.calculate(y_true, y_pred)
        corr = self.corr_metric.calculate(y_true, y_pred)

        result = {
            'date': test_date,
            'num_players': len(merged),
            'mape': mape,
            'rmse': rmse,
            'correlation': corr,
            'mean_projected': merged['projected_fpts'].mean(),
            'mean_actual': merged['actual_fpts'].mean()
        }

        logger.debug(f"Slate evaluation: MAPE={mape:.2f}, RMSE={rmse:.2f}, Corr={corr:.3f}")

        return result

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

        for idx, player_row in tqdm(salaries_df.iterrows(), total=len(salaries_df), desc="Per-player models", leave=False):
            player_id = player_row.get('playerID')
            player_name = player_row.get('longName')
            logger.info(f"Processing {player_name} ({player_id}) for {test_date}")

            player_training_data = training_data[training_data['playerID'] == player_id].copy()

            if len(player_training_data) < self.min_player_games:
                logger.debug(f"Skipping {player_name}: only {len(player_training_data)} games (need {self.min_player_games})")
                continue

            try:
                if should_recalibrate or player_id not in self.player_models:
                    X_train, y_train = self._build_training_features(player_training_data)

                    if X_train.empty or y_train.empty or len(X_train) < 3:
                        logger.debug(f"Insufficient features for {player_name}")
                        continue

                    safe_player_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in player_name)
                    safe_player_name = safe_player_name.replace(' ', '_')

                    inputs_dir = Path(PER_PLAYER_TRAINING_INPUTS_DIR)
                    inputs_dir.mkdir(parents=True, exist_ok=True)
                    input_file = inputs_dir / f"{safe_player_name}_{player_id}_inputs.parquet"

                    model = self._train_model(
                        X_train,
                        y_train,
                        save_inputs=True,
                        input_save_path=str(input_file)
                    )
                    self.player_models[player_id] = model
                    models_trained += 1

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

        if should_recalibrate:
            self.last_training_date = test_date

        if not all_projections:
            logger.warning("No player projections generated")
            return pd.DataFrame()

        projections_df = pd.concat(all_projections, ignore_index=True)
        logger.info(f"Generated projections for {players_with_models}/{total_players} players")
        logger.info(f"Models trained: {models_trained}, reused: {models_reused}")

        return projections_df

    def _save_model(self, model, model_file: Path, player_name: str, player_id: str, num_samples: int):
        import json

        model_data = {
            'model': model,
            'player_name': player_name,
            'player_id': player_id,
            'num_training_samples': num_samples,
            'model_type': self.model_type,
            'window_sizes': self.rolling_window_sizes
        }

        if hasattr(model, 'save'):
            model.save(str(model_file))
        else:
            import pickle
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)

        metadata_file = model_file.with_suffix('.json')
        metadata = {
            'player_name': player_name,
            'player_id': player_id,
            'num_training_samples': num_samples,
            'model_type': self.model_type,
            'window_sizes': self.rolling_window_sizes,
            'model_file': str(model_file.name)
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved model and metadata for {player_name}")

    def _save_slate_model(self, model, model_file: Path, test_date: str, num_samples: int):
        import json

        if hasattr(model, 'save'):
            model.save(str(model_file))
        else:
            import pickle
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

        metadata_file = model_file.with_suffix('.json')
        metadata = {
            'training_date': test_date,
            'num_training_samples': num_samples,
            'model_type': self.model_type,
            'window_sizes': self.rolling_window_sizes,
            'model_file': str(model_file.name)
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved per-slate model and metadata for {test_date}")

    def _aggregate_results(self) -> Dict[str, Any]:
        logger.info("Aggregating backtest results")

        if not self.results:
            return {
                'error': 'No results to aggregate',
                'num_slates': 0
            }

        daily_df = pd.DataFrame(self.results)

        summary = {
            'num_slates': len(daily_df),
            'date_range': f"{daily_df['date'].min()} to {daily_df['date'].max()}",
            'mean_mape': daily_df['mape'].mean(),
            'median_mape': daily_df['mape'].median(),
            'std_mape': daily_df['mape'].std(),
            'mean_rmse': daily_df['rmse'].mean(),
            'std_rmse': daily_df['rmse'].std(),
            'mean_correlation': daily_df['correlation'].mean(),
            'std_correlation': daily_df['correlation'].std(),
            'total_players_evaluated': daily_df['num_players'].sum(),
            'avg_players_per_slate': daily_df['num_players'].mean(),
            'daily_results': daily_df
        }

        logger.info(f"Aggregation complete: {summary['num_slates']} slates, MAPE={summary['mean_mape']:.2f}%")

        return summary
