import sys
from pathlib import Path

repo_root = "C:\\Users\\antho\\OneDrive\\Documents\\Repositories\\delapan-fantasy"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
import logging
import sqlite3

from src.evaluation.base import BacktestStrategy, ValidationStrategy
from src.evaluation.metrics import evaluate_predictions

logger = logging.getLogger(__name__)

DB_PATH = "C:\\Users\\antho\\OneDrive\\Documents\\Repositories\\delapan-fantasy\\nba_dfs.db"


class DailyBacktest:

    def __init__(
        self,
        data_prep,
        storage,
        seasons: Optional[List[str]] = None
    ):
        self.data_prep = data_prep
        self.storage = storage

        if seasons is None:
            current_year = datetime.now().year
            self.seasons = [str(current_year), str(current_year - 1)]
        else:
            self.seasons = seasons

        self.results = []

    def prepare_slate_data(self, game_date: str) -> Dict[str, Any]:
        """
        Prepare all data for specific slate date.

        Calls BacktestDataPrep to:
        1. Get teams playing on date
        2. Get players for teams
        3. Collect missing historical data

        Args:
            game_date: Date in YYYYMMDD format

        Returns:
            Dict with data preparation summary
        """
        logger.info(f"Preparing slate data for {game_date}")
        prep_summary = self.data_prep.prepare_data_for_date(
            game_date=game_date,
            seasons=self.seasons
        )
        logger.info(f"Data preparation complete: {prep_summary.get('total_players', 0)} players found")

        return prep_summary

    def verify_slate_readiness(self, game_date: str) -> Dict[str, Any]:
        """
        Verify all required data exists for slate.

        Args:
            game_date: Date in YYYYMMDD format

        Returns:
            Dict with verification status
        """
        logger.info(f"Verifying data completeness for {game_date}")
        verification = self.data_prep.verify_data_completeness(game_date)
        logger.info(f"Verification complete: {verification['players_with_data']}/{verification.get('total_players', 0)} players have data")

        return verification

    def prepare_player_features(self, game_date: str, players: List[Dict]) -> pd.DataFrame:
        """
        Load and prepare player features for lineup optimization.

        Args:
            game_date: Date in YYYYMMDD format
            players: List of player dicts with playerID, playerName, team, pos

        Returns:
            DataFrame with player features
        """
        logger.info(f"Preparing features for {len(players)} players")
        player_features = []
        players_with_data = 0
        players_without_data = 0

        for player in players:
            player_id = player['playerID']

            player_data = self._load_player_historical_data(player_id)

            if player_data.empty:
                players_without_data += 1
                logger.debug(f"No historical data for player {player_id}")
                continue

            features = self._calculate_player_features(
                player_data=player_data,
                player_id=player_id,
                player_name=player.get('playerName', ''),
                team=player.get('team', ''),
                pos=player.get('pos', ''),
                as_of_date=game_date
            )

            if features:
                player_features.append(features)
                players_with_data += 1
            else:
                players_without_data += 1
                logger.debug(f"Insufficient data to calculate features for player {player_id}")

        logger.info(f"Feature preparation complete: {players_with_data} with features, {players_without_data} without")

        return pd.DataFrame(player_features)

    def _load_player_historical_data(self, player_id: str) -> pd.DataFrame:
        """
        Load player historical data from database.

        Args:
            player_id: Player ID

        Returns:
            DataFrame with player game logs
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            query = """
                SELECT * FROM player_logs
                WHERE playerID = ?
                ORDER BY gameDate DESC
            """
            df = pd.read_sql_query(query, conn, params=(player_id,))
            conn.close()

            logger.debug(f"Loaded {len(df)} game logs for player {player_id} from database")
            return df

        except Exception as e:
            logger.warning(f"Failed to load data for player {player_id} from database: {str(e)}")
            return pd.DataFrame()

    def _calculate_player_features(
        self,
        player_data: pd.DataFrame,
        player_id: str,
        player_name: str,
        team: str,
        pos: str,
        as_of_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate rolling features from historical data.

        Args:
            player_data: DataFrame with player game logs
            player_id: Player ID
            player_name: Player name
            team: Team abbreviation
            pos: Position
            as_of_date: Date in YYYYMMDD format (exclude games on/after this date)

        Returns:
            Dict with player features or None if insufficient data
        """
        if player_data.empty:
            logger.debug(f"Empty player data for {player_id}")
            return None

        df = player_data.copy()

        if 'gameDate' in df.columns:
            df['gameDate'] = pd.to_datetime(df['gameDate'], format='%Y%m%d', errors='coerce')
            as_of_datetime = pd.to_datetime(as_of_date, format='%Y%m%d')
            df = df[df['gameDate'] < as_of_datetime]

        if df.empty or len(df) < 3:
            logger.debug(f"Insufficient games for {player_id}: {len(df)} games (need 3+)")
            return None

        logger.debug(f"Calculating features for {player_id} using {len(df)} games")
        stats_columns = ['mins', 'pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'fpts']

        features = {
            'playerID': player_id,
            'playerName': player_name,
            'team': team,
            'pos': pos,
            'games_played': len(df)
        }

        for col in stats_columns:
            if col not in df.columns:
                logger.debug(f"Column {col} not found for player {player_id}")
                continue

            values = pd.to_numeric(df[col], errors='coerce').dropna()

            if len(values) == 0:
                logger.debug(f"No valid values for {col} for player {player_id}")
                continue

            features[f'{col}_avg_3'] = values.rolling(3).mean()
            features[f'{col}_std_3'] = values.rolling(3).std()

            features[f'{col}_avg_5'] = values.rolling(5).mean() if len(values) >= 5 else values.mean()
            features[f'{col}_std_5'] = values.rolling(5).std() if len(values) >= 5 else values.std()

            features[f'{col}_avg_10'] = values.rolling(10).mean() if len(values) >= 10 else values.mean()
            features[f'{col}_std_10'] = values.rolling(10).std() if len(values) >= 10 else values.std()

        return features

    def drun_daily_backtest(
        self,
        game_date: str,
        model_fn=None,
        optimizer_fn=None
    ) -> Dict[str, Any]:
        """
        Run complete backtest for single date.

        Process:
        1. Prepare data (collect missing historical data)
        2. Verify data completeness
        3. Load player features
        4. Generate projections (if model provided)
        5. Optimize lineup (if optimizer provided)
        6. Compare to actuals (if available)

        Args:
            game_date: Date in YYYYMMDD format
            model_fn: Optional function that returns trained model
            optimizer_fn: Optional function that returns lineup optimizer

        Returns:
            Dict with backtest results
        """
        logger.info(f"Starting daily backtest for {game_date}")
        print(f"\n=== Running Backtest for {game_date} ===\n")

        prep_summary = self.prepare_slate_data(game_date)

        verification = self.verify_slate_readiness(game_date)

        if not verification['is_complete']:
            logger.warning(f"{verification['players_missing_data']} players missing data for {game_date}")
            print(f"WARNING: {verification['players_missing_data']} players missing data")

        players = prep_summary.get('all_players', [])
        logger.info(f"Processing {len(players)} players")

        player_features = self.prepare_player_features(game_date, players)

        print(f"\nPrepared features for {len(player_features)} players")

        result = {
            'game_date': game_date,
            'teams': prep_summary.get('teams', []),
            'total_players': prep_summary.get('total_players', 0),
            'players_with_data': verification['players_with_data'],
            'players_with_features': len(player_features),
            'data_complete': verification['is_complete']
        }

        if model_fn is not None:
            logger.info("Generating projections")
            print("\nGenerating projections...")
            projections = self._generate_projections(player_features, model_fn)
            result['projections'] = projections
            logger.info(f"Generated {len(projections)} projections")
            print(f"Generated {len(projections)} projections")

        if optimizer_fn is not None and model_fn is not None:
            logger.info("Optimizing lineup")
            print("\nOptimizing lineup...")
            lineup = self._optimize_lineup(projections, optimizer_fn)
            result['lineup'] = lineup
            logger.info(f"Optimized lineup with {len(lineup)} players")
            print(f"Optimized lineup with {len(lineup)} players")

        self.results.append(result)
        logger.info(f"Backtest complete for {game_date}")

        return result

    def _generate_projections(
        self,
        player_features: pd.DataFrame,
        model_fn
    ) -> pd.DataFrame:
        """
        Generate fantasy point projections using model.

        Args:
            player_features: DataFrame with player features
            model_fn: Function that returns trained model

        Returns:
            DataFrame with projections
        """
        if player_features.empty:
            logger.warning("No player features to generate projections")
            return pd.DataFrame()

        logger.debug(f"Generating projections for {len(player_features)} players")
        model = model_fn()

        feature_cols = [col for col in player_features.columns
                       if col not in ['playerID', 'playerName', 'team', 'pos']]

        logger.debug(f"Using {len(feature_cols)} feature columns for predictions")
        X = player_features[feature_cols].fillna(0)

        predictions = model.predict(X)
        logger.debug(f"Predictions generated: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")

        projections = player_features[['playerID', 'playerName', 'team', 'pos']].copy()
        projections['projected_fpts'] = predictions

        return projections

    def _optimize_lineup(
        self,
        projections: pd.DataFrame,
        optimizer_fn
    ) -> pd.DataFrame:
        """
        Optimize lineup using projections.

        Args:
            projections: DataFrame with player projections
            optimizer_fn: Function that returns lineup optimizer

        Returns:
            DataFrame with optimal lineup
        """
        logger.debug(f"Optimizing lineup from {len(projections)} player projections")
        optimizer = optimizer_fn()

        lineup = optimizer.optimize(projections)
        logger.debug(f"Lineup optimized with {len(lineup)} players")

        return lineup

    def run_multi_date_backtest(
        self,
        game_dates: List[str],
        model_fn=None,
        optimizer_fn=None
    ) -> pd.DataFrame:
        """
        Run backtest across multiple dates.

        Args:
            game_dates: List of dates in YYYYMMDD format
            model_fn: Optional function that returns trained model
            optimizer_fn: Optional function that returns lineup optimizer

        Returns:
            DataFrame with results for all dates
        """
        logger.info(f"Starting multi-date backtest for {len(game_dates)} dates")
        print(f"\n=== Running Multi-Date Backtest ===")
        print(f"Dates to process: {len(game_dates)}\n")

        successful = 0
        failed = 0

        for idx, game_date in enumerate(game_dates, 1):
            print(f"\n[{idx}/{len(game_dates)}] Processing {game_date}")
            logger.info(f"Processing date {idx}/{len(game_dates)}: {game_date}")

            try:
                self.run_daily_backtest(
                    game_date=game_date,
                    model_fn=model_fn,
                    optimizer_fn=optimizer_fn
                )
                successful += 1
            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {game_date}: {str(e)}", exc_info=True)
                print(f"ERROR: Failed to process {game_date}: {str(e)}")
                continue

        logger.info(f"Multi-date backtest complete: {successful} successful, {failed} failed")

        return self.get_results()

    def get_results(self) -> pd.DataFrame:
        """Get backtest results as DataFrame."""
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame(self.results)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all backtest runs."""
        if not self.results:
            return {}

        df = self.get_results()

        summary = {
            'num_slates': len(df),
            'total_players_processed': df['players_with_features'].sum(),
            'avg_players_per_slate': df['players_with_features'].mean(),
            'data_completeness_rate': (df['data_complete'].sum() / len(df)) * 100
        }

        return summary


class WalkForwardValidator(ValidationStrategy):

    def __init__(
        self,
        train_window: int = 30,
        test_window: int = 1,
        step_size: int = 1
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def split(
        self,
        data: pd.DataFrame,
        date_column: str = 'date'
    ) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
        logger.info(f"Creating walk-forward splits: train_window={self.train_window}, test_window={self.test_window}, step_size={self.step_size}")

        data = data.sort_values(date_column).reset_index(drop=True)
        dates = pd.to_datetime(data[date_column])
        unique_dates = sorted(dates.unique())

        logger.debug(f"Data spans {len(unique_dates)} unique dates from {unique_dates[0]} to {unique_dates[-1]}")

        splits = []
        start_idx = 0

        while start_idx + self.train_window + self.test_window <= len(unique_dates):
            train_end_idx = start_idx + self.train_window
            test_end_idx = train_end_idx + self.test_window

            train_dates = unique_dates[start_idx:train_end_idx]
            test_dates = unique_dates[train_end_idx:test_end_idx]

            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)

            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()

            splits.append((train_data, test_data))
            logger.debug(f"Split {len(splits)}: train {train_dates[0]} to {train_dates[-1]} ({len(train_data)} rows), test {test_dates[0]} to {test_dates[-1]} ({len(test_data)} rows)")

            start_idx += self.step_size

        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits


class BacktestRunner(BacktestStrategy):

    def __init__(
        self,
        model_fn: Callable,
        storage,
        validator: Optional[ValidationStrategy] = None
    ):
        self.model_fn = model_fn
        self.storage = storage
        self.validator = validator or WalkForwardValidator()
        self.results = []

    def run(
        self,
        start_date: str,
        end_date: str,
        feature_columns: List[str],
        target_column: str = 'fantasyPoints',
        date_column: str = 'date',
        **kwargs
    ) -> Dict[str, Any]:
        logger.info(f"Starting BacktestRunner from {start_date} to {end_date}")
        logger.debug(f"Feature columns: {feature_columns}")
        logger.debug(f"Target column: {target_column}")

        data = self._load_data(start_date, end_date, **kwargs)

        if data.empty:
            logger.error("No data available for date range")
            return {'error': 'No data available for date range'}

        logger.info(f"Loaded {len(data)} rows of data")

        splits = self.validator.split(data, date_column)

        for idx, (train_data, test_data) in enumerate(splits):
            if train_data.empty or test_data.empty:
                logger.warning(f"Skipping fold {idx}: empty train or test data")
                continue

            logger.info(f"Processing fold {idx}: train_size={len(train_data)}, test_size={len(test_data)}")

            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]

            logger.debug(f"Training model for fold {idx}")
            model = self.model_fn()
            model.fit(X_train, y_train)

            logger.debug(f"Generating predictions for fold {idx}")
            predictions = model.predict(X_test)

            metrics = evaluate_predictions(y_test, pd.Series(predictions))
            logger.info(f"Fold {idx} metrics: MAPE={metrics['mape']:.2f}, RMSE={metrics['rmse']:.2f}, Corr={metrics['correlation']:.3f}")

            self.results.append({
                'fold': idx,
                'train_start': train_data[date_column].min(),
                'train_end': train_data[date_column].max(),
                'test_start': test_data[date_column].min(),
                'test_end': test_data[date_column].max(),
                'train_size': len(train_data),
                'test_size': len(test_data),
                **metrics
            })

        logger.info(f"BacktestRunner complete: processed {len(self.results)} folds")

        return self._aggregate_results()

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def _load_data(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(DB_PATH)
            query = """
                SELECT * FROM player_logs
                WHERE gameDate BETWEEN ? AND ?
                ORDER BY gameDate
            """
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()

            logger.info(f"Loaded {len(df)} rows from database for date range {start_date} to {end_date}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data from database: {str(e)}")
            return pd.DataFrame()

    def _aggregate_results(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        aggregated = {
            'num_folds': len(self.results),
            'avg_mape': df['mape'].mean(),
            'std_mape': df['mape'].std(),
            'avg_rmse': df['rmse'].mean(),
            'std_rmse': df['rmse'].std(),
            'avg_correlation': df['correlation'].mean(),
            'std_correlation': df['correlation'].std()
        }

        if 'mae' in df.columns:
            aggregated['avg_mae'] = df['mae'].mean()
            aggregated['std_mae'] = df['mae'].std()

        if 'r2' in df.columns:
            aggregated['avg_r2'] = df['r2'].mean()
            aggregated['std_r2'] = df['r2'].std()

        return aggregated


class SlateBacktester:

    def __init__(self, storage):
        self.storage = storage
        self.results = []

    def backtest_slate(
        self,
        date: str,
        projections: pd.DataFrame,
        actuals: pd.DataFrame,
        lineup_optimizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        logger.info(f"Backtesting slate for {date}")
        logger.debug(f"Projections: {len(projections)} players, Actuals: {len(actuals)} players")

        merged = projections.merge(
            actuals,
            on='playerID',
            suffixes=('_proj', '_actual')
        )

        logger.debug(f"Merged {len(merged)} players with both projections and actuals")

        projection_metrics = evaluate_predictions(
            merged['fantasyPoints_actual'],
            merged['fantasyPoints_proj']
        )

        logger.info(f"Slate {date} projection metrics: MAPE={projection_metrics['mape']:.2f}, RMSE={projection_metrics['rmse']:.2f}, Corr={projection_metrics['correlation']:.3f}")

        result = {
            'date': date,
            'num_players': len(merged),
            'projection_mape': projection_metrics['mape'],
            'projection_rmse': projection_metrics['rmse'],
            'projection_correlation': projection_metrics['correlation']
        }

        if lineup_optimizer is not None:
            logger.debug("Optimizing lineup")
            lineup = lineup_optimizer.optimize(projections)
            lineup_actual = merged[merged['playerID'].isin(lineup['playerID'])]

            result['lineup_projected_score'] = lineup['fantasyPoints_proj'].sum()
            result['lineup_actual_score'] = lineup_actual['fantasyPoints_actual'].sum()
            result['lineup_error'] = (
                result['lineup_actual_score'] - result['lineup_projected_score']
            )

            logger.info(f"Lineup: projected={result['lineup_projected_score']:.2f}, actual={result['lineup_actual_score']:.2f}, error={result['lineup_error']:.2f}")

        self.results.append(result)
        return result

    def run_multi_slate(
        self,
        start_date: str,
        end_date: str,
        projection_fn: Callable,
        lineup_optimizer: Optional[Any] = None
    ) -> pd.DataFrame:
        logger.info(f"Running multi-slate backtest from {start_date} to {end_date}")
        dates = pd.date_range(start_date, end_date, freq='D')
        logger.info(f"Processing {len(dates)} dates")

        successful = 0
        skipped = 0
        failed = 0

        for date in dates:
            date_str = date.strftime('%Y%m%d')

            try:
                logger.debug(f"Loading projections and actuals for {date_str}")
                projections = projection_fn(date_str)
                actuals = self._load_actuals(date_str)

                if projections.empty or actuals.empty:
                    logger.warning(f"Skipping {date_str}: empty projections or actuals")
                    skipped += 1
                    continue

                self.backtest_slate(
                    date_str,
                    projections,
                    actuals,
                    lineup_optimizer
                )
                successful += 1

            except Exception as e:
                failed += 1
                logger.error(f"Error processing {date_str}: {str(e)}", exc_info=True)
                print(f"Error processing {date_str}: {str(e)}")
                continue

        logger.info(f"Multi-slate backtest complete: {successful} successful, {skipped} skipped, {failed} failed")

        return pd.DataFrame(self.results)

    def _load_actuals(self, date: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(DB_PATH)
            query = """
                SELECT playerID, fantasyPoints, pts, reb, ast, stl, blk, TOV
                FROM player_logs
                WHERE gameDate = ?
            """
            df = pd.read_sql_query(query, conn, params=(date,))
            conn.close()

            logger.debug(f"Loaded actuals for {len(df)} players on {date}")
            return df

        except Exception as e:
            logger.error(f"Failed to load actuals for {date}: {str(e)}")
            return pd.DataFrame()

    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        summary = {
            'num_slates': len(df),
            'avg_projection_mape': df['projection_mape'].mean(),
            'avg_projection_rmse': df['projection_rmse'].mean(),
            'avg_projection_correlation': df['projection_correlation'].mean()
        }

        if 'lineup_actual_score' in df.columns:
            summary['avg_lineup_score'] = df['lineup_actual_score'].mean()
            summary['avg_lineup_error'] = df['lineup_error'].mean()
            summary['std_lineup_error'] = df['lineup_error'].std()

        return summary
