import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from src.historical_data_loader import HistoricalDataLoader
from src.feature_builder_v2 import FeatureBuilder
from src.metrics import calculate_mape, calculate_rmse, calculate_correlation

logger = logging.getLogger(__name__)


class WalkForwardBacktest:

    def __init__(
        self,
        db_path: str,
        start_date: str,
        end_date: str,
        lookback_days: int = 90,
        model_type: str = 'xgboost',
        model_params: Optional[Dict] = None,
        rolling_window_sizes: Optional[List[int]] = None,
        output_dir: str = 'data/backtest_results',
        per_player_models: bool = False,
        min_player_games: int = 10
    ):
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_days = lookback_days
        self.model_type = model_type
        self.model_params = model_params or {}
        self.rolling_window_sizes = rolling_window_sizes or [3, 5, 10]
        self.output_dir = output_dir
        self.per_player_models = per_player_models
        self.min_player_games = min_player_games

        self.loader = HistoricalDataLoader(db_path)
        self.feature_builder = FeatureBuilder()
        self.results = []

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized WalkForwardBacktest: {start_date} to {end_date}")
        logger.info(f"Per-player models: {per_player_models}")

    def run(self) -> Dict[str, Any]:
        logger.info("="*80)
        logger.info("STARTING WALK-FORWARD BACKTEST")
        logger.info("="*80)
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Lookback: {self.lookback_days} days")
        logger.info(f"Model: {self.model_type}")
        logger.info("="*80)

        slate_dates = self.loader.load_slate_dates(self.start_date, self.end_date)

        if not slate_dates:
            logger.error("No slate dates found")
            return {'error': 'No slate dates found'}

        print(f"\nBacktesting {len(slate_dates)} slates from {self.start_date} to {self.end_date}\n")

        for i, test_date in enumerate(slate_dates):
            print(f"\n[{i+1}/{len(slate_dates)}] Processing {test_date}...")
            logger.info(f"Processing slate {i+1}/{len(slate_dates)}: {test_date}")

            try:
                slate_data = self.loader.load_slate_data(test_date)

                if slate_data['salaries'].empty:
                    print(f"  Skipping (no salary data)")
                    logger.warning(f"No salary data for {test_date}, skipping")
                    continue

                training_data = self.loader.load_historical_player_logs(
                    end_date=test_date,
                    lookback_days=self.lookback_days
                )

                if len(training_data) < 500:
                    print(f"  Skipping (insufficient training data: {len(training_data)} games)")
                    logger.warning(f"Insufficient training data for {test_date}")
                    continue

                print(f"  Building features from {len(training_data)} training games...")

                if self.per_player_models:
                    print(f"  Using per-player models...")
                    projections = self._generate_per_player_projections(
                        slate_data,
                        training_data,
                        test_date
                    )

                    if projections.empty:
                        print(f"  Skipping (no per-player projections generated)")
                        logger.warning(f"No per-player projections for {test_date}")
                        continue

                else:
                    X_train, y_train = self.feature_builder.build_training_features(
                        training_data,
                        window_sizes=self.rolling_window_sizes
                    )

                    if X_train.empty or y_train.empty:
                        print(f"  Skipping (feature generation failed)")
                        logger.warning(f"Feature generation failed for {test_date}")
                        continue

                    print(f"  Training {self.model_type} model...")
                    model = self._train_model(X_train, y_train)

                    print(f"  Building slate features...")
                    slate_features = self.feature_builder.build_slate_features(
                        slate_data,
                        training_data,
                        window_sizes=self.rolling_window_sizes
                    )

                    if slate_features.empty:
                        print(f"  Skipping (no slate features generated)")
                        logger.warning(f"No slate features for {test_date}")
                        continue

                    print(f"  Generating projections for {len(slate_features)} players...")
                    projections = self._generate_projections(model, slate_features)

                    if projections.empty:
                        print(f"  Skipping (no projections generated)")
                        logger.warning(f"No projections generated for {test_date}")
                        continue

                print(f"  Loading actual results...")
                actuals = self._load_actuals(test_date)

                if actuals.empty:
                    print(f"  Skipping (no actual results available)")
                    logger.warning(f"No actual results for {test_date}")
                    continue

                print(f"  Evaluating...")
                daily_results = self._evaluate_slate(test_date, projections, actuals)

                self.results.append(daily_results)

                print(f"  MAPE: {daily_results['mape']:.1f}%  |  RMSE: {daily_results['rmse']:.2f}  |  Corr: {daily_results['correlation']:.3f}")

            except Exception as e:
                logger.error(f"Error processing {test_date}: {str(e)}", exc_info=True)
                print(f"  ERROR: {str(e)}")
                continue

        logger.info(f"Walk-forward backtest complete: {len(self.results)} slates processed")

        return self._aggregate_results()

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.debug(f"Training {self.model_type} model on {len(X_train)} samples")

        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(**self.model_params)
                model.fit(X_train, y_train, verbose=False)
                return model
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                self.model_type = 'random_forest'

        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=self.model_params.get('n_estimators', 200),
                max_depth=self.model_params.get('max_depth', 6),
                random_state=self.model_params.get('random_state', 42),
                n_jobs=-1
            )
            model.fit(X_train, y_train)
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
        predictions = model.predict(X)

        projections = slate_features[metadata_cols].copy()
        projections['projected_fpts'] = predictions
        projections['value'] = projections['projected_fpts'] / (projections['salary'] / 1000)

        logger.debug(f"Projections: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")

        return projections

    def _load_actuals(self, date: str) -> pd.DataFrame:
        import sqlite3

        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT playerID, longName as playerName, team, pos,
                       pts, reb, ast, stl, blk, TOV, mins
                FROM player_logs_extracted
                WHERE gameDate = ?
            """
            df = pd.read_sql_query(query, conn, params=(date,))
            conn.close()

            if df.empty:
                return pd.DataFrame()

            df['actual_fpts'] = df.apply(self.feature_builder.calculate_dk_fantasy_points, axis=1)

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

        mape = calculate_mape(merged['actual_fpts'], merged['projected_fpts'])
        rmse = calculate_rmse(merged['actual_fpts'], merged['projected_fpts'])
        corr = calculate_correlation(merged['actual_fpts'], merged['projected_fpts'])

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

        salaries_df = slate_data['salaries'].copy()
        all_projections = []

        total_players = len(salaries_df)
        players_with_models = 0

        year = test_date[:4]
        month = test_date[4:6]
        day = test_date[6:8]
        models_dir = Path('data') / 'models' / year / month / day
        models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving models to: {models_dir}")

        for idx, player_row in salaries_df.iterrows():
            player_id = player_row.get('playerID')
            player_name = player_row.get('longName') or player_row.get('playerName', '')

            player_training_data = training_data[training_data['playerID'] == player_id].copy()

            if len(player_training_data) < self.min_player_games:
                logger.debug(f"Skipping {player_name}: only {len(player_training_data)} games (need {self.min_player_games})")
                continue

            try:
                X_train, y_train = self.feature_builder.build_training_features(
                    player_training_data,
                    window_sizes=self.rolling_window_sizes
                )

                if X_train.empty or y_train.empty or len(X_train) < 3:
                    logger.debug(f"Insufficient features for {player_name}")
                    continue

                model = self._train_model(X_train, y_train)

                safe_player_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in player_name)
                safe_player_name = safe_player_name.replace(' ', '_')

                model_file = models_dir / f"{safe_player_name}_{player_id}.pkl"

                self._save_model(model, model_file, player_name, player_id, len(X_train))
                logger.debug(f"Saved model for {player_name} to {model_file}")

                slate_data_single = {
                    'salaries': salaries_df.iloc[[idx]],
                    'date': slate_data['date']
                }

                slate_features = self.feature_builder.build_slate_features(
                    slate_data_single,
                    player_training_data,
                    window_sizes=self.rolling_window_sizes
                )

                if slate_features.empty:
                    logger.debug(f"No features generated for {player_name}")
                    continue

                projection = self._generate_projections(model, slate_features)
                all_projections.append(projection)
                players_with_models += 1

            except Exception as e:
                logger.warning(f"Error generating projection for {player_name}: {str(e)}")
                continue

        if not all_projections:
            logger.warning("No player projections generated")
            return pd.DataFrame()

        projections_df = pd.concat(all_projections, ignore_index=True)
        logger.info(f"Generated projections for {players_with_models}/{total_players} players using individual models")
        logger.info(f"Saved {players_with_models} player models to {models_dir}")
        print(f"  Generated {players_with_models}/{total_players} player-specific projections")
        print(f"  Models saved to: {models_dir}")

        return projections_df

    def _save_model(self, model, model_file: Path, player_name: str, player_id: str, num_samples: int):
        import pickle
        import json

        model_data = {
            'model': model,
            'player_name': player_name,
            'player_id': player_id,
            'num_training_samples': num_samples,
            'model_type': self.model_type,
            'window_sizes': self.rolling_window_sizes,
            'lookback_days': self.lookback_days
        }

        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

        metadata_file = model_file.with_suffix('.json')
        metadata = {
            'player_name': player_name,
            'player_id': player_id,
            'num_training_samples': num_samples,
            'model_type': self.model_type,
            'window_sizes': self.rolling_window_sizes,
            'lookback_days': self.lookback_days,
            'model_file': str(model_file.name)
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved model and metadata for {player_name}")

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
