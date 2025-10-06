import pandas as pd
import numpy as np
import logging
import sqlite3
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from src.evaluation.backtest_config import BacktestConfig
from src.evaluation.data_loader import HistoricalDataLoader
from src.evaluation.feature_builder import FeatureBuilder
from src.evaluation.metrics import calculate_mape, calculate_rmse, calculate_correlation

logger = logging.getLogger(__name__)


class WalkForwardBacktest:

    def __init__(self, config: BacktestConfig, db_path: str = None):
        self.config = config
        self.db_path = db_path or "nba_dfs.db"
        self.loader = HistoricalDataLoader(config, self.db_path)
        self.feature_builder = FeatureBuilder()
        self.results = []

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        logger.info("="*80)
        logger.info("STARTING WALK-FORWARD BACKTEST")
        logger.info("="*80)
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Lookback: {self.config.lookback_days} days")
        logger.info(f"Model: {self.config.model_type}")
        logger.info("="*80)

        slate_dates = self.loader.load_slate_dates()

        if not slate_dates:
            logger.error("No slate dates found")
            return {'error': 'No slate dates found'}

        print(f"\nBacktesting {len(slate_dates)} slates from {self.config.start_date} to {self.config.end_date}\n")

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
                    lookback_days=self.config.lookback_days
                )

                if len(training_data) < self.config.min_training_games:
                    print(f"  Skipping (insufficient training data: {len(training_data)} games)")
                    logger.warning(f"Insufficient training data for {test_date}: {len(training_data)} < {self.config.min_training_games}")
                    continue

                print(f"  Building features from {len(training_data)} training games...")
                X_train, y_train = self.feature_builder.build_training_features(training_data)

                if X_train.empty or y_train.empty:
                    print(f"  Skipping (feature generation failed)")
                    logger.warning(f"Feature generation failed for {test_date}")
                    continue

                print(f"  Training {self.config.model_type} model...")
                model = self.train_model(X_train, y_train)

                print(f"  Building slate features...")
                slate_features = self.feature_builder.build_slate_features(slate_data, training_data)

                if slate_features.empty:
                    print(f"  Skipping (no slate features generated)")
                    logger.warning(f"No slate features for {test_date}")
                    continue

                print(f"  Generating projections for {len(slate_features)} players...")
                projections = self.generate_projections(model, slate_features)

                print(f"  Loading actual results...")
                actuals = self.load_actual_results(test_date)

                if actuals.empty:
                    print(f"  Skipping (no actual results available)")
                    logger.warning(f"No actual results for {test_date}")
                    continue

                print(f"  Evaluating...")
                daily_results = self.evaluate_slate(
                    test_date,
                    projections,
                    actuals
                )

                self.results.append(daily_results)

                print(f"  MAPE: {daily_results['mape']:.1f}%  |  RMSE: {daily_results['rmse']:.2f}  |  Corr: {daily_results['correlation']:.3f}")

                if self.config.save_daily_results:
                    self._save_daily_results(test_date, daily_results, projections, actuals)

            except Exception as e:
                logger.error(f"Error processing {test_date}: {str(e)}", exc_info=True)
                print(f"  ERROR: {str(e)}")
                continue

        logger.info(f"Walk-forward backtest complete: {len(self.results)} slates processed")

        return self.aggregate_results()

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.debug(f"Training {self.config.model_type} model on {len(X_train)} samples")

        if self.config.model_type == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(**self.config.model_params)
                model.fit(X_train, y_train, verbose=False)
                return model
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                self.config.model_type = 'random_forest'

        if self.config.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=self.config.model_params.get('n_estimators', 200),
                max_depth=self.config.model_params.get('max_depth', 6),
                random_state=self.config.model_params.get('random_state', 42),
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            return model

        if self.config.model_type == 'linear':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            return model

        raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def generate_projections(
        self,
        model,
        slate_features: pd.DataFrame
    ) -> pd.DataFrame:
        logger.debug(f"Generating projections for {len(slate_features)} players")

        feature_cols = [col for col in slate_features.columns
                       if col not in ['playerID', 'playerName', 'team', 'pos', 'salary']]

        X = slate_features[feature_cols].fillna(0)

        predictions = model.predict(X)

        projections = slate_features[['playerID', 'playerName', 'team', 'pos', 'salary']].copy()
        projections['projected_fpts'] = predictions

        projections['value'] = projections['projected_fpts'] / (projections['salary'] / 1000)

        projections['floor'] = projections['projected_fpts'] * 0.7
        projections['ceiling'] = projections['projected_fpts'] * 1.3

        logger.debug(f"Projections: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")

        return projections

    def load_actual_results(self, date: str) -> pd.DataFrame:
        logger.debug(f"Loading actual results for {date}")

        try:
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT playerID, longName as playerName, team, pos,
                       pts, reb, ast, stl, blk, TOV, mins
                FROM player_logs
                WHERE gameDate = ?
            """

            df = pd.read_sql_query(query, conn, params=(date,))
            conn.close()

            if df.empty:
                logger.warning(f"No actual results found for {date}")
                return pd.DataFrame()

            df['actual_fpts'] = df.apply(self.feature_builder.calculate_dk_fantasy_points, axis=1)

            logger.debug(f"Loaded actual results for {len(df)} players")

            return df[['playerID', 'playerName', 'team', 'pos', 'actual_fpts']]

        except Exception as e:
            logger.error(f"Failed to load actual results for {date}: {str(e)}")
            return pd.DataFrame()

    def evaluate_slate(
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
            logger.warning(f"No matching players between projections and actuals for {test_date}")
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

    def aggregate_results(self) -> Dict[str, Any]:
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

    def _save_daily_results(
        self,
        date: str,
        metrics: Dict,
        projections: pd.DataFrame,
        actuals: pd.DataFrame
    ):
        try:
            daily_dir = Path(self.config.output_dir) / 'daily' / date
            daily_dir.mkdir(parents=True, exist_ok=True)

            with open(daily_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

            merged = projections.merge(
                actuals[['playerID', 'actual_fpts']],
                on='playerID',
                how='left'
            )

            merged['error'] = merged['projected_fpts'] - merged['actual_fpts']
            merged['abs_error'] = merged['error'].abs()
            merged['pct_error'] = (merged['error'] / merged['actual_fpts'].replace(0, np.nan)) * 100

            merged.to_csv(daily_dir / 'projections_vs_actuals.csv', index=False)

            logger.debug(f"Saved daily results for {date}")

        except Exception as e:
            logger.warning(f"Failed to save daily results for {date}: {str(e)}")
